"""P3-A: Conv/Pooling/BatchNorm Real Forward Logic Tests.

Comprehensive tests covering the actual forward computation of:
- Convolution layers (1x1, 3x3, padding, stride, groups, bias)
- Pooling layers (MAX, AVE, global pooling, CEIL/FLOOR rounding)
- BatchNorm layers (global stats, epsilon, scale factor)

Each test includes numpy reference implementations and detailed perf_trace
logging of forward time, memory peaks, and exception details.

Run with:
    pytest tests/python/test_p3a_conv_pool_bn.py -v
    pytest tests/python/test_p3a_conv_pool_bn.py -v -s  # verbose with [PERF] logs
"""
from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import net_param_from_string, net_from_param
from .conftest import require_cpp_extension, perf_trace


# ═══════════════════════════════════════════════════════════════════════
# Numpy Reference Implementations
# ═══════════════════════════════════════════════════════════════════════

def conv2d_np(x, W, b=None, stride=1, pad=0, dilation=1, groups=1):
    """Numpy reference for 2D convolution (NCHW format).
    
    Args:
        x: Input tensor (N, C_in, H, W)
        W: Weight tensor (C_out, C_in/groups, kH, kW)
        b: Bias tensor (C_out,) or None
        stride: int or (stride_h, stride_w)
        pad: int or (pad_h, pad_w)
        dilation: int or (dilation_h, dilation_w)
        groups: number of groups
    Returns:
        Output tensor (N, C_out, H_out, W_out)
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(pad, int):
        pad = (pad, pad)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    stride_h, stride_w = stride
    pad_h, pad_w = pad
    dilation_h, dilation_w = dilation
    
    N, C_in, H, W_in = x.shape
    C_out, C_in_g, kH, kW = W.shape
    assert C_in % groups == 0
    assert C_out % groups == 0
    assert C_in_g == C_in // groups
    
    H_out = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
    
    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    
    for n in range(N):
        for g in range(groups):
            c_in_start = g * C_in_g
            c_in_end = (g + 1) * C_in_g
            c_out_start = g * (C_out // groups)
            c_out_end = (g + 1) * (C_out // groups)
            
            for c_out in range(c_out_start, c_out_end):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride_h
                        w_start = w_out * stride_w
                        for c_in in range(c_in_start, c_in_end):
                            for kh in range(kH):
                                for kw in range(kW):
                                    h_pad = h_start + kh * dilation_h
                                    w_pad = w_start + kw * dilation_w
                                    output[n, c_out, h_out, w_out] += (
                                        x_padded[n, c_in, h_pad, w_pad]
                                        * W[c_out, c_in - c_in_start, kh, kw]
                                    )
    
    if b is not None:
        output += b.reshape(1, -1, 1, 1)
    
    return output


def pooling2d_np(x, kernel_size, stride=None, pad=0, pool_type='MAX',
                 ceil_mode=False, global_pooling=False):
    """Numpy reference for 2D pooling (NCHW format).
    
    Args:
        x: Input tensor (N, C, H, W)
        kernel_size: int or (kH, kW)
        stride: int or (stride_h, stride_w), defaults to kernel_size
        pad: int or (pad_h, pad_w)
        pool_type: 'MAX' or 'AVE'
        ceil_mode: use CEIL rounding instead of FLOOR
        global_pooling: pool over entire spatial dimensions
    Returns:
        Output tensor (N, C, H_out, W_out)
    """
    N, C, H, W_in = x.shape
    
    if global_pooling:
        kH, kW = H, W_in
        stride_h, stride_w = 1, 1
        pad_h, pad_w = 0, 0
    else:
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        if isinstance(pad, int):
            pad_h = pad_w = pad
        else:
            pad_h, pad_w = pad
    
    # Compute output size (matching Caffe's CEIL/FLOOR logic)
    if ceil_mode:
        H_out = int(np.ceil(float(H + 2 * pad_h - kH) / stride_h)) + 1
        W_out = int(np.ceil(float(W_in + 2 * pad_w - kW) / stride_w)) + 1
    else:
        H_out = int(np.floor(float(H + 2 * pad_h - kH) / stride_h)) + 1
        W_out = int(np.floor(float(W_in + 2 * pad_w - kW) / stride_w)) + 1
    
    # Caffe's post-correction for padding
    if pad_h > 0 or pad_w > 0:
        if (H_out - 1) * stride_h >= H + pad_h:
            H_out -= 1
        if (W_out - 1) * stride_w >= W_in + pad_w:
            W_out -= 1
    
    if global_pooling:
        H_out = W_out = 1
    
    output = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    
    for n in range(N):
        for c in range(C):
            for ph in range(H_out):
                for pw in range(W_out):
                    hstart = ph * stride_h - pad_h
                    wstart = pw * stride_w - pad_w
                    hend = min(hstart + kH, H)
                    wend = min(wstart + kW, W_in)
                    hstart = max(hstart, 0)
                    wstart = max(wstart, 0)
                    
                    patch = x[n, c, hstart:hend, wstart:wend]
                    if pool_type == 'MAX':
                        output[n, c, ph, pw] = np.max(patch) if patch.size > 0 else -np.inf
                    elif pool_type == 'AVE':
                        output[n, c, ph, pw] = np.mean(patch) if patch.size > 0 else 0.0
    
    return output


def batchnorm_np(x, mean, variance, scale_factor=1.0, eps=1e-5,
                 use_global_stats=True):
    """Numpy reference for BatchNorm (NCHW format).
    
    Args:
        x: Input tensor (N, C, H, W) or (N, C)
        mean: Channel-wise accumulated mean (C,) — blobs_[0] in caffe-ffi
        variance: Channel-wise accumulated variance (C,) — blobs_[1] in caffe-ffi
        scale_factor: accumulation count (scalar, blobs_[2][0])
        eps: epsilon for numerical stability
        use_global_stats: if True, use stored mean/var; if False, compute from batch
    Returns:
        Normalized output tensor (same shape as x)
    """
    # Match C++ logic: scale_factor_use = 1/scale_factor (0 -> 1.0)
    if scale_factor != 0.0:
        sf = 1.0 / scale_factor
    else:
        sf = 1.0
    
    if x.ndim == 4:
        N, C, H, W = x.shape
        spatial_dim = H * W
        # Reshape mean/var for broadcasting: (1, C, 1, 1)
        mean_bc = mean.reshape(1, C, 1, 1) * sf
        var_bc = variance.reshape(1, C, 1, 1) * sf
    elif x.ndim == 2:
        N, C = x.shape
        spatial_dim = 1
        mean_bc = mean.reshape(1, C) * sf
        var_bc = variance.reshape(1, C) * sf
    elif x.ndim == 1:
        C = x.shape[0]
        spatial_dim = 1
        mean_bc = mean * sf
        var_bc = variance * sf
    else:
        raise ValueError(f"Unsupported ndim: {x.ndim}")
    
    x_norm = (x - mean_bc) / np.sqrt(np.maximum(var_bc, 0.0) + eps)
    return x_norm.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Convolution Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestConvolutionLayers:
    """Tests for Convolution layer forward correctness."""
    
    def test_conv_1x1_identity(self, ptrace):
        """1x1 convolution with identity weights should preserve input."""
        prototxt = """name: "conv_1x1_test"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 3 dim: 4 dim: 4 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 3
    kernel_size: 1
    bias_term: false
  }
}
"""
        with ptrace("Net(conv 1x1 identity)") as t:
            net = net_from_param(net_param_from_string(prototxt))
            t['layers'] = len(net.layers_array())
        
        # Identity weight: C_out x C_in x 1 x 1 = 3x3x1x1 identity matrix
        # caffe-ffi stores weights as 2D (num_output, kernel_dim); keep 4D for numpy ref
        W4d = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        W = W4d.reshape(3, -1)  # flatten to 2D for caffe-ffi blob
        conv_layer = net.layer_by_name("conv")
        with ptrace("load 1x1 identity weights") as t:
            conv_layer.blobs[0].from_numpy(W)
            t['W_shape'] = f"{W.shape}"
        
        # Input: random data
        rng = np.random.RandomState(42)
        inp = rng.randn(1, 3, 4, 4).astype(np.float32)
        
        with ptrace("conv 1x1 forward") as t:
            out = net.forward({"data": inp})
            t['input_shape'] = f"{inp.shape}"
            t['output_shape'] = f"{out['conv'].shape}"
        
        expected = conv2d_np(inp, W4d, stride=1, pad=0)
        np.testing.assert_allclose(out["conv"], expected, rtol=1e-5, atol=1e-6)
    
    def test_conv_1x1_with_bias(self, ptrace):
        """1x1 convolution with bias adds bias correctly."""
        prototxt = """name: "conv_1x1_bias"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 2 dim: 3 dim: 3 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 4
    kernel_size: 1
    bias_term: true
  }
}
"""
        with ptrace("Net(conv 1x1 with bias)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(123)
        W = rng.randn(4, 2, 1, 1).astype(np.float32) * 0.1
        b = rng.randn(4).astype(np.float32) * 0.01
        
        conv_layer = net.layer_by_name("conv")
        with ptrace("load 1x1 W+b"):
            conv_layer.blobs[0].from_numpy(W)
            conv_layer.blobs[1].from_numpy(b)
        
        inp = rng.randn(2, 2, 3, 3).astype(np.float32)
        
        with ptrace("conv 1x1+b forward") as t:
            out = net.forward({"data": inp})
            t['kernel'] = '1x1'
            t['has_bias'] = True
        
        expected = conv2d_np(inp, W, b, stride=1, pad=0)
        np.testing.assert_allclose(out["conv"], expected, rtol=1e-5, atol=1e-6)
    
    def test_conv_3x3_no_padding(self, ptrace):
        """3x3 convolution with no padding, stride=1."""
        prototxt = """name: "conv_3x3"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 5 dim: 5 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 1
    kernel_size: 3
    pad: 0
    stride: 1
    bias_term: false
  }
}
"""
        with ptrace("Net(conv 3x3 no pad)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        # Simple edge-detection-like kernel
        W = np.array([[[[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]]]], dtype=np.float32)
        
        conv_layer = net.layer_by_name("conv")
        with ptrace("load 3x3 Laplacian kernel"):
            conv_layer.blobs[0].from_numpy(W)
        
        # Input: a simple pattern
        inp = np.zeros((1, 1, 5, 5), dtype=np.float32)
        inp[0, 0, 2, 2] = 1.0  # Center pixel
        
        with ptrace("conv 3x3 forward") as t:
            out = net.forward({"data": inp})
            t['kernel'] = '3x3'
            t['pad'] = 0
            t['input_center'] = 1.0
        
        expected = conv2d_np(inp, W, stride=1, pad=0)
        np.testing.assert_allclose(out["conv"], expected, rtol=1e-5, atol=1e-6)
        # Center of output should be 4.0 (Laplacian response)
        assert out["conv"][0, 0, 1, 1] == pytest.approx(4.0, abs=1e-6)
    
    def test_conv_3x3_with_padding(self, ptrace):
        """3x3 convolution with padding=1 (same convolution, size preserved)."""
        prototxt = """name: "conv_3x3_pad"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 2 dim: 4 dim: 4 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 2
    kernel_size: 3
    pad: 1
    stride: 1
    bias_term: true
  }
}
"""
        with ptrace("Net(conv 3x3 pad=1)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(456)
        W = rng.randn(2, 2, 3, 3).astype(np.float32) * 0.1
        b = rng.randn(2).astype(np.float32) * 0.01
        
        conv_layer = net.layer_by_name("conv")
        with ptrace("load 3x3 pad=1 W+b"):
            conv_layer.blobs[0].from_numpy(W)
            conv_layer.blobs[1].from_numpy(b)
        
        inp = rng.randn(1, 2, 4, 4).astype(np.float32)
        
        with ptrace("conv 3x3 pad=1 forward") as t:
            out = net.forward({"data": inp})
            t['kernel'] = '3x3'
            t['pad'] = 1
            t['output_preserved'] = out['conv'].shape[2] == inp.shape[2]
        
        assert out["conv"].shape == (1, 2, 4, 4), "Same conv should preserve spatial dims"
        expected = conv2d_np(inp, W, b, stride=1, pad=1)
        np.testing.assert_allclose(out["conv"], expected, rtol=1e-5, atol=1e-6)
    
    def test_conv_stride_2(self, ptrace):
        """Convolution with stride=2 downsamples by 2x."""
        prototxt = """name: "conv_stride2"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 6 dim: 6 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 1
    kernel_size: 3
    pad: 1
    stride: 2
    bias_term: false
  }
}
"""
        with ptrace("Net(conv stride=2)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        W = np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0  # Box blur
        
        conv_layer = net.layer_by_name("conv")
        with ptrace("load 3x3 box blur"):
            conv_layer.blobs[0].from_numpy(W)
        
        inp = np.ones((1, 1, 6, 6), dtype=np.float32)
        
        with ptrace("conv stride=2 forward") as t:
            out = net.forward({"data": inp})
            t['stride'] = 2
            t['expected_H'] = 3
        
        assert out["conv"].shape == (1, 1, 3, 3), "Stride 2 should halve spatial dims"
        expected = conv2d_np(inp, W, stride=2, pad=1)
        np.testing.assert_allclose(out["conv"], expected, rtol=1e-5, atol=1e-6)
    
    def test_conv_group_2(self, ptrace):
        """Group convolution with groups=2 splits channels."""
        prototxt = """name: "conv_group2"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 4 dim: 3 dim: 3 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 2
    kernel_size: 3
    pad: 1
    group: 2
    bias_term: false
  }
}
"""
        with ptrace("Net(conv groups=2)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(789)
        # Weight shape for groups=2: (C_out=2, C_in/groups=2, 3, 3)
        W = rng.randn(2, 2, 3, 3).astype(np.float32) * 0.1
        
        conv_layer = net.layer_by_name("conv")
        with ptrace("load group conv weights"):
            conv_layer.blobs[0].from_numpy(W)
        
        inp = rng.randn(1, 4, 3, 3).astype(np.float32)
        
        with ptrace("conv groups=2 forward") as t:
            out = net.forward({"data": inp})
            t['groups'] = 2
            t['C_in'] = 4
            t['C_in_per_group'] = 2
        
        expected = conv2d_np(inp, W, stride=1, pad=1, groups=2)
        np.testing.assert_allclose(out["conv"], expected, rtol=1e-5, atol=1e-6)
    
    def test_conv_repeated_forward_stable(self, ptrace):
        """Multiple forward passes should produce identical results (determinism)."""
        prototxt = """name: "conv_repeat"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 2 dim: 4 dim: 4 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 2
    kernel_size: 3
    pad: 1
    bias_term: true
  }
}
"""
        with ptrace("Net(conv repeated forward)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(100)
        W = rng.randn(2, 2, 3, 3).astype(np.float32) * 0.1
        b = rng.randn(2).astype(np.float32) * 0.01
        
        conv_layer = net.layer_by_name("conv")
        with ptrace("load conv weights"):
            conv_layer.blobs[0].from_numpy(W)
            conv_layer.blobs[1].from_numpy(b)
        
        inp = rng.randn(2, 2, 4, 4).astype(np.float32)
        
        n_iters = 10
        outputs = []
        with ptrace(f"conv forward x{n_iters}") as t:
            for i in range(n_iters):
                out = net.forward({"data": inp})
                outputs.append(out["conv"].copy())
            t['iterations'] = n_iters
        
        # All outputs should be identical
        for i in range(1, n_iters):
            np.testing.assert_array_equal(outputs[0], outputs[i])
        
        expected = conv2d_np(inp, W, b, stride=1, pad=1)
        np.testing.assert_allclose(outputs[0], expected, rtol=1e-5, atol=1e-6)
    
    def test_conv_weights_unchanged_after_forward(self, ptrace):
        """Forward pass should not modify weight blobs."""
        prototxt = """name: "conv_weights"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 3 dim: 3 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 1
    kernel_size: 3
    pad: 0
    bias_term: true
  }
}
"""
        with ptrace("Net(conv weights check)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        W = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=np.float32)
        b = np.array([0.5], dtype=np.float32)
        
        conv_layer = net.layer_by_name("conv")
        with ptrace("load Sobel-like kernel"):
            conv_layer.blobs[0].from_numpy(W)
            conv_layer.blobs[1].from_numpy(b)
        
        inp = np.random.randn(1, 1, 3, 3).astype(np.float32)
        
        with ptrace("conv forward (weights check)"):
            net.forward({"data": inp})
        
        with ptrace("verify weights unchanged"):
            np.testing.assert_array_equal(conv_layer.blobs[0].to_numpy(), W)
            np.testing.assert_array_equal(conv_layer.blobs[1].to_numpy(), b)


# ═══════════════════════════════════════════════════════════════════════
# Pooling Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestPoolingLayers:
    """Tests for Pooling layer forward correctness."""
    
    def test_max_pooling_2x2_stride2(self, ptrace):
        """MAX pooling 2x2 stride=2: picks maximum in each 2x2 window."""
        prototxt = """name: "maxpool_test"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 4 dim: 4 } }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
"""
        with ptrace("Net(MAX pool 2x2 s2)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        # Known input with clear maxima
        inp = np.array([[[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]]]], dtype=np.float32)
        
        with ptrace("MAX pool 2x2 s2 forward") as t:
            out = net.forward({"data": inp})
            t['pool'] = 'MAX'
            t['kernel'] = '2x2'
        
        expected = pooling2d_np(inp, 2, stride=2, pool_type='MAX')
        np.testing.assert_array_equal(out["pool"], expected)
        # Expected: [[6, 8], [14, 16]]
        assert out["pool"][0, 0, 0, 0] == 6.0
        assert out["pool"][0, 0, 0, 1] == 8.0
        assert out["pool"][0, 0, 1, 0] == 14.0
        assert out["pool"][0, 0, 1, 1] == 16.0
    
    def test_ave_pooling_2x2_stride2(self, ptrace):
        """AVE pooling 2x2 stride=2: computes average in each window."""
        prototxt = """name: "avepool_test"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 4 dim: 4 } }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
"""
        with ptrace("Net(AVE pool 2x2 s2)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        inp = np.ones((1, 1, 4, 4), dtype=np.float32)
        
        with ptrace("AVE pool 2x2 s2 forward") as t:
            out = net.forward({"data": inp})
            t['pool'] = 'AVE'
        
        expected = pooling2d_np(inp, 2, stride=2, pool_type='AVE')
        np.testing.assert_allclose(out["pool"], expected, rtol=1e-6)
        # Average of 1s = 1.0
        np.testing.assert_allclose(out["pool"], np.ones((1, 1, 2, 2), dtype=np.float32))
    
    def test_max_pooling_3x3_pad1(self, ptrace):
        """MAX pooling 3x3 with pad=1, stride=1 (same pooling)."""
        prototxt = """name: "maxpool_3x3"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 5 dim: 5 } }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {
    pool: MAX
    kernel_size: 3
    pad: 1
    stride: 1
  }
}
"""
        with ptrace("Net(MAX pool 3x3 pad=1)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        inp = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
        
        with ptrace("MAX pool 3x3 pad=1 forward") as t:
            out = net.forward({"data": inp})
            t['pool'] = 'MAX'
            t['pad'] = 1
        
        expected = pooling2d_np(inp, 3, stride=1, pad=1, pool_type='MAX')
        np.testing.assert_allclose(out["pool"], expected, rtol=1e-6)
        assert out["pool"].shape == (1, 1, 5, 5)
    
    def test_global_max_pooling(self, ptrace):
        """Global MAX pooling reduces spatial dims to 1x1."""
        prototxt = """name: "global_maxpool"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 dim: 8 dim: 8 } }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {
    pool: MAX
    global_pooling: true
  }
}
"""
        with ptrace("Net(global MAX pool)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(42)
        inp = rng.randn(2, 3, 8, 8).astype(np.float32)
        
        with ptrace("global MAX pool forward") as t:
            out = net.forward({"data": inp})
            t['pool'] = 'MAX'
            t['global'] = True
        
        assert out["pool"].shape == (2, 3, 1, 1), "Global pool should output 1x1"
        expected = pooling2d_np(inp, None, pool_type='MAX', global_pooling=True)
        np.testing.assert_allclose(out["pool"], expected, rtol=1e-6)
        # Verify against manual max
        for n in range(2):
            for c in range(3):
                assert out["pool"][n, c, 0, 0] == pytest.approx(
                    np.max(inp[n, c]), abs=1e-6)
    
    def test_global_ave_pooling(self, ptrace):
        """Global AVE pooling computes spatial average."""
        prototxt = """name: "global_avepool"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 2 dim: 4 dim: 4 } }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
"""
        with ptrace("Net(global AVE pool)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        inp = np.ones((1, 2, 4, 4), dtype=np.float32)
        
        with ptrace("global AVE pool forward") as t:
            out = net.forward({"data": inp})
            t['pool'] = 'AVE'
            t['global'] = True
        
        assert out["pool"].shape == (1, 2, 1, 1)
        expected = pooling2d_np(inp, None, pool_type='AVE', global_pooling=True)
        np.testing.assert_allclose(out["pool"], expected, rtol=1e-6)
        np.testing.assert_allclose(out["pool"], np.ones((1, 2, 1, 1), dtype=np.float32))
    
    def test_ave_pooling_padding_boundary(self, ptrace):
        """AVE pooling with padding correctly counts valid pixels only."""
        prototxt = """name: "avepool_pad"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 3 dim: 3 } }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {
    pool: AVE
    kernel_size: 3
    pad: 1
    stride: 2
  }
}
"""
        with ptrace("Net(AVE pool 3x3 pad=1 s2)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        inp = np.ones((1, 1, 3, 3), dtype=np.float32)
        
        with ptrace("AVE pool pad boundary forward") as t:
            out = net.forward({"data": inp})
            t['pool'] = 'AVE'
            t['pad_boundary'] = True
        
        expected = pooling2d_np(inp, 3, stride=2, pad=1, pool_type='AVE')
        np.testing.assert_allclose(out["pool"], expected, rtol=1e-5, atol=1e-6)
    
    def test_pooling_repeated_forward_stable(self, ptrace):
        """Pooling is deterministic: repeated forwards give same result."""
        prototxt = """name: "pool_repeat"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 2 dim: 4 dim: 4 } }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
"""
        with ptrace("Net(MAX pool repeat)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(999)
        inp = rng.randn(2, 2, 4, 4).astype(np.float32)
        
        n_iters = 10
        outputs = []
        with ptrace(f"MAX pool forward x{n_iters}") as t:
            for _ in range(n_iters):
                out = net.forward({"data": inp})
                outputs.append(out["pool"].copy())
            t['iterations'] = n_iters
        
        for i in range(1, n_iters):
            np.testing.assert_array_equal(outputs[0], outputs[i])
        
        expected = pooling2d_np(inp, 2, stride=2, pool_type='MAX')
        np.testing.assert_allclose(outputs[0], expected, rtol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# BatchNorm Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestBatchNormLayers:
    """Tests for BatchNorm layer forward correctness."""
    
    def test_batchnorm_zero_mean_unit_var(self, ptrace):
        """BatchNorm with zero mean and unit variance should be approximately identity for normalized input."""
        prototxt = """name: "bn_test"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 4 dim: 2 dim: 3 dim: 3 } }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "data"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
"""
        with ptrace("Net(BatchNorm zero/unit)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        # Set mean=0, variance=1, scale_factor=1
        mean = np.zeros(2, dtype=np.float32)
        var = np.ones(2, dtype=np.float32)
        scale_factor = np.array([1.0], dtype=np.float32)
        
        bn_layer = net.layer_by_name("bn")
        with ptrace("load BN stats (mean=0, var=1)"):
            bn_layer.blobs[0].from_numpy(mean)
            bn_layer.blobs[1].from_numpy(var)
            bn_layer.blobs[2].from_numpy(scale_factor)
        
        inp = np.array([[[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=np.float32)
        inp = np.broadcast_to(inp, (4, 2, 3, 3)).copy()
        
        with ptrace("BN forward (zero/unit)") as t:
            out = net.forward({"data": inp})
            t['mean'] = '0'
            t['var'] = '1'
        
        expected = batchnorm_np(inp, mean, var, eps=1e-5)
        np.testing.assert_allclose(out["bn"], expected, rtol=1e-5, atol=1e-6)
    
    def test_batchnorm_normalizes(self, ptrace):
        """BatchNorm normalizes input to zero mean and unit variance (with stored stats)."""
        prototxt = """name: "bn_normalize"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 2 dim: 2 dim: 2 } }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "data"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
"""
        with ptrace("Net(BatchNorm normalize)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        # Known mean=2.0, var=1.0
        mean = np.array([2.0, 3.0], dtype=np.float32)
        var = np.array([1.0, 4.0], dtype=np.float32)
        scale_factor = np.array([1.0], dtype=np.float32)
        
        bn_layer = net.layer_by_name("bn")
        with ptrace("load BN stats (mean=2/3, var=1/4)"):
            bn_layer.blobs[0].from_numpy(mean)
            bn_layer.blobs[1].from_numpy(var)
            bn_layer.blobs[2].from_numpy(scale_factor)
        
        inp = np.array([[[[2.0, 4.0],
                          [0.0, 2.0]],
                         [[3.0, 7.0],
                          [-1.0, 3.0]]]], dtype=np.float32)
        
        with ptrace("BN forward (normalize)") as t:
            out = net.forward({"data": inp})
            t['channels'] = 2
        
        expected = batchnorm_np(inp, mean, var, eps=1e-5)
        np.testing.assert_allclose(out["bn"], expected, rtol=1e-5, atol=1e-6)
        # Channel 0: (x - 2) / sqrt(1 + eps) should give [0, 2, -2, 0]
        np.testing.assert_allclose(out["bn"][0, 0], 
                                   np.array([[0, 2], [-2, 0]], dtype=np.float32),
                                   rtol=1e-4)
    
    def test_batchnorm_epsilon_stability(self, ptrace):
        """BatchNorm with small epsilon handles near-zero variance without division by zero."""
        prototxt = """name: "bn_eps"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 2 dim: 2 } }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "data"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-3
  }
}
"""
        with ptrace("Net(BatchNorm eps stability)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        # Zero variance - epsilon prevents div by zero
        mean = np.array([0.0], dtype=np.float32)
        var = np.array([0.0], dtype=np.float32)
        scale_factor = np.array([1.0], dtype=np.float32)
        
        bn_layer = net.layer_by_name("bn")
        with ptrace("load BN zero variance"):
            bn_layer.blobs[0].from_numpy(mean)
            bn_layer.blobs[1].from_numpy(var)
            bn_layer.blobs[2].from_numpy(scale_factor)
        
        inp = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        
        with ptrace("BN forward (zero var)") as t:
            out = net.forward({"data": inp})
            t['eps'] = 1e-3
            t['zero_var'] = True
            t['output_finite'] = np.all(np.isfinite(out["bn"]))
        
        assert np.all(np.isfinite(out["bn"])), "Output should be finite with eps>0"
        expected = batchnorm_np(inp, mean, var, eps=1e-3)
        np.testing.assert_allclose(out["bn"], expected, rtol=1e-5, atol=1e-6)
    
    def test_batchnorm_scale_factor(self, ptrace):
        """BatchNorm scale_factor divides mean and variance correctly."""
        prototxt = """name: "bn_scale"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 1 dim: 4 } }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "data"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
"""
        with ptrace("Net(BatchNorm scale factor)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        # Accumulated stats: blobs_[0]=mean_acc, blobs_[1]=var_acc, blobs_[2]=2.0
        # C++ computes: effective_mean = mean_acc / 2.0 = 4.0/2 = 2.0
        #                effective_var  = var_acc  / 2.0 = 6.0/2 = 3.0
        # Input is all 2.0, so normalized = (2.0 - 2.0) / sqrt(3.0 + eps) = 0.0
        mean_acc = np.array([4.0], dtype=np.float32)
        var_acc = np.array([6.0], dtype=np.float32)
        scale_factor = np.array([2.0], dtype=np.float32)
        
        bn_layer = net.layer_by_name("bn")
        with ptrace("load BN with scale_factor=2"):
            bn_layer.blobs[0].from_numpy(mean_acc)
            bn_layer.blobs[1].from_numpy(var_acc)
            bn_layer.blobs[2].from_numpy(scale_factor)
        
        inp = np.array([[[[2.0, 2.0, 2.0, 2.0]]]], dtype=np.float32)
        
        with ptrace("BN forward (scale factor)") as t:
            out = net.forward({"data": inp})
            t['scale_factor'] = 2.0
        
        expected = batchnorm_np(inp, mean_acc, var_acc, scale_factor=2.0, eps=1e-5)
        np.testing.assert_allclose(out["bn"], expected, rtol=1e-5, atol=1e-6)
    
    def test_batchnorm_2d_input(self, ptrace):
        """BatchNorm works with 2D input (N, C) for inner-product outputs."""
        prototxt = """name: "bn_2d"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 3 dim: 4 } }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "data"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
"""
        with ptrace("Net(BatchNorm 2D)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        mean = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        var = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        scale_factor = np.array([1.0], dtype=np.float32)
        
        bn_layer = net.layer_by_name("bn")
        with ptrace("load BN 2D stats"):
            bn_layer.blobs[0].from_numpy(mean)
            bn_layer.blobs[1].from_numpy(var)
            bn_layer.blobs[2].from_numpy(scale_factor)
        
        inp = np.array([[0.0, 1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0, 4.0],
                        [2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        
        with ptrace("BN 2D forward") as t:
            out = net.forward({"data": inp})
            t['input_ndim'] = 2
        
        assert out["bn"].shape == inp.shape
        expected = batchnorm_np(inp, mean, var, eps=1e-5)
        np.testing.assert_allclose(out["bn"], expected, rtol=1e-5, atol=1e-6)
    
    def test_batchnorm_repeated_forward_stable(self, ptrace):
        """BatchNorm is deterministic: repeated forwards give same result."""
        prototxt = """name: "bn_repeat"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "data"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
"""
        with ptrace("Net(BN repeat)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(77)
        mean = rng.randn(3).astype(np.float32)
        var = np.abs(rng.randn(3).astype(np.float32)) + 0.1
        scale_factor = np.array([1.0], dtype=np.float32)
        
        bn_layer = net.layer_by_name("bn")
        with ptrace("load BN random stats"):
            bn_layer.blobs[0].from_numpy(mean)
            bn_layer.blobs[1].from_numpy(var)
            bn_layer.blobs[2].from_numpy(scale_factor)
        
        inp = rng.randn(2, 3, 4, 4).astype(np.float32)
        
        n_iters = 10
        outputs = []
        with ptrace(f"BN forward x{n_iters}") as t:
            for _ in range(n_iters):
                out = net.forward({"data": inp})
                outputs.append(out["bn"].copy())
            t['iterations'] = n_iters
        
        for i in range(1, n_iters):
            np.testing.assert_array_equal(outputs[0], outputs[i])
    
    def test_batchnorm_weights_unchanged_after_forward(self, ptrace):
        """Forward pass should not modify BN stats blobs."""
        prototxt = """name: "bn_weights"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 2 dim: 2 dim: 2 } }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "data"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
"""
        with ptrace("Net(BN weights check)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        mean = np.array([0.5, -0.5], dtype=np.float32)
        var = np.array([1.0, 2.0], dtype=np.float32)
        scale_factor = np.array([1.0], dtype=np.float32)
        
        bn_layer = net.layer_by_name("bn")
        with ptrace("load BN initial stats"):
            bn_layer.blobs[0].from_numpy(mean)
            bn_layer.blobs[1].from_numpy(var)
            bn_layer.blobs[2].from_numpy(scale_factor)
        
        inp = np.random.randn(1, 2, 2, 2).astype(np.float32)
        
        with ptrace("BN forward (weights check)"):
            net.forward({"data": inp})
        
        with ptrace("verify BN stats unchanged"):
            np.testing.assert_array_equal(bn_layer.blobs[0].to_numpy(), mean)
            np.testing.assert_array_equal(bn_layer.blobs[1].to_numpy(), var)
            np.testing.assert_array_equal(bn_layer.blobs[2].to_numpy(), scale_factor)


# ═══════════════════════════════════════════════════════════════════════
# End-to-End Layer Combination Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestConvPoolBNCombination:
    """End-to-end tests combining Conv -> Pooling -> BatchNorm pipeline."""
    
    def test_conv_pool_bn_pipeline(self, ptrace):
        """Conv(3x3) -> MAX Pool(2x2) -> BatchNorm pipeline runs correctly."""
        prototxt = """name: "conv_pool_bn"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 dim: 8 dim: 8 } }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 4
    kernel_size: 3
    pad: 1
    stride: 1
    bias_term: true
  }
}
layer {
  name: "pool"
  type: "Pooling"
  bottom: "conv"
  top: "pool"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "bn"
  type: "BatchNorm"
  bottom: "pool"
  top: "bn"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
"""
        with ptrace("Net(Conv->Pool->BN pipeline)") as t:
            net = net_from_param(net_param_from_string(prototxt))
            t['layers'] = len(net.layers_array())
        
        rng = np.random.RandomState(42)
        # Conv weights: 4 out, 3 in, 3x3 kernel
        W = rng.randn(4, 3, 3, 3).astype(np.float32) * 0.1
        b_conv = rng.randn(4).astype(np.float32) * 0.01
        # BN stats: 4 channels
        bn_mean = rng.randn(4).astype(np.float32)
        bn_var = np.abs(rng.randn(4).astype(np.float32)) + 0.1
        bn_scale = np.array([1.0], dtype=np.float32)
        
        with ptrace("load pipeline weights"):
            net.layer_by_name("conv").blobs[0].from_numpy(W)
            net.layer_by_name("conv").blobs[1].from_numpy(b_conv)
            net.layer_by_name("bn").blobs[0].from_numpy(bn_mean)
            net.layer_by_name("bn").blobs[1].from_numpy(bn_var)
            net.layer_by_name("bn").blobs[2].from_numpy(bn_scale)
        
        inp = rng.randn(2, 3, 8, 8).astype(np.float32)
        
        with ptrace("Conv->Pool->BN forward") as t:
            out = net.forward({"data": inp})
            t['input_shape'] = f"{inp.shape}"
            t['bn_output_finite'] = np.all(np.isfinite(out["bn"]))
        
        # Shape checks
        assert out["conv"].shape == (2, 4, 8, 8), "Conv with pad=1 should preserve size"
        assert out["pool"].shape == (2, 4, 4, 4), "Pool 2x2 s2 should halve size"
        assert out["bn"].shape == (2, 4, 4, 4), "BN should preserve shape"
        
        # Numpy reference pipeline
        conv_out = conv2d_np(inp, W, b_conv, stride=1, pad=1)
        pool_out = pooling2d_np(conv_out, 2, stride=2, pool_type='MAX')
        bn_out = batchnorm_np(pool_out, bn_mean, bn_var, eps=1e-5)
        
        np.testing.assert_allclose(out["bn"], bn_out, rtol=1e-4, atol=1e-5)
    
    def test_pipeline_repeated_forward_no_crash(self, ptrace):
        """Conv->Pool->BN pipeline should be stable over many forwards (segfault/OOM check)."""
        prototxt = """name: "pipeline_stress"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 4 dim: 8 dim: 16 dim: 16 } }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 16
    kernel_size: 3
    pad: 1
    stride: 1
    bias_term: true
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "bn1"
  type: "BatchNorm"
  bottom: "pool1"
  top: "bn1"
  batch_norm_param {
    use_global_stats: true
    eps: 1e-5
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "bn1"
  top: "conv2"
  convolution_param {
    num_output: 8
    kernel_size: 1
    pad: 0
    stride: 1
    bias_term: true
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
"""
        with ptrace("Net(pipeline stress)"):
            net = net_from_param(net_param_from_string(prototxt))
        
        rng = np.random.RandomState(12345)
        
        def _init_conv(layer_name, C_out, C_in, kH, kW):
            W = rng.randn(C_out, C_in, kH, kW).astype(np.float32) * 0.05
            b = rng.randn(C_out).astype(np.float32) * 0.001
            layer = net.layer_by_name(layer_name)
            layer.blobs[0].from_numpy(W)
            layer.blobs[1].from_numpy(b)
        
        def _init_bn(layer_name, C):
            mean = rng.randn(C).astype(np.float32) * 0.1
            var = np.abs(rng.randn(C).astype(np.float32)) + 0.5
            sf = np.array([1.0], dtype=np.float32)
            layer = net.layer_by_name(layer_name)
            layer.blobs[0].from_numpy(mean)
            layer.blobs[1].from_numpy(var)
            layer.blobs[2].from_numpy(sf)
        
        with ptrace("initialize stress pipeline weights"):
            _init_conv("conv1", 16, 8, 3, 3)
            _init_bn("bn1", 16)
            _init_conv("conv2", 8, 16, 1, 1)
        
        n_iters = 20
        outputs = []
        with ptrace(f"pipeline forward x{n_iters} (segfault/OOM check)") as t:
            for i in range(n_iters):
                inp = rng.randn(4, 8, 16, 16).astype(np.float32)
                out = net.forward({"data": inp})
                outputs.append(out["pool2"].copy())
                assert np.all(np.isfinite(out["pool2"])), f"Non-finite output at iter {i}"
            t['iterations'] = n_iters
            t['last_shape'] = f"{outputs[-1].shape}"
        
        # Final output should be (4, 8, 1, 1) due to global pooling
        assert outputs[-1].shape == (4, 8, 1, 1)
