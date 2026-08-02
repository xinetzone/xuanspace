"""P3-D: Slice/Crop/Deconvolution/LRN Real Forward Tests.

Supplementary tests covering the remaining 4 C++ layers to bring
coverage from 84% (21/25) to 100% (25/25):
- Slice layer (tensor splitting along axis, equal divisions, slice_points, N=1 identity)
- Crop layer (spatial cropping with offset, 2-bottom reference shape)
- Deconvolution layer (transposed convolution, 1x1 kernel, shape verification, bias)
- LRN layer (Local Response Normalization across channels, AlexNet defaults)

Each test includes numpy reference implementations where applicable.

Run with:
    pytest tests/python/test_p3d_slice_crop_deconv_lrn.py -v
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

def slice_np(x, num_outs, axis=1, slice_points=None):
    """Numpy reference for Slice layer.

    Splits tensor along axis into num_outs parts.
    If slice_points is None, divides evenly; otherwise uses explicit split points.
    """
    if slice_points is not None:
        points = [0] + list(slice_points) + [x.shape[axis]]
        slices = []
        for i in range(len(points) - 1):
            s = [slice(None)] * x.ndim
            s[axis] = slice(points[i], points[i+1])
            slices.append(x[tuple(s)])
        return slices
    else:
        size = x.shape[axis]
        assert size % num_outs == 0, f"Axis size {size} not divisible by {num_outs}"
        part = size // num_outs
        return np.split(x, num_outs, axis=axis)


def crop_np(x, ref_shape, axis=2, offsets=None):
    """Numpy reference for Crop layer.

    Crops x to match ref_shape starting at axis, with optional offsets.
    """
    ndim = x.ndim
    slices = [slice(None)] * ndim
    if offsets is None:
        offsets = [0] * (ndim - axis)
    elif len(offsets) == 1:
        offsets = [offsets[0]] * (ndim - axis)
    for i in range(ndim - axis):
        d = axis + i
        off = offsets[i] if i < len(offsets) else 0
        slices[d] = slice(off, off + ref_shape[d])
    return x[tuple(slices)]


def lrn_np(x, size=5, alpha=1e-4, beta=0.75, k=1.0):
    """Numpy reference for LRN (ACROSS_CHANNELS mode).

    y = x / (k + alpha/size * sum_{c in window} x_c^2)^beta

    The normalization window spans adjacent channels with pre_pad=(size-1)//2.
    """
    N, C, H, W = x.shape
    pre_pad = (size - 1) // 2
    # Pad channels with zeros
    padded = np.zeros((N, C + size - 1, H, W), dtype=np.float64)
    padded[:, pre_pad:pre_pad + C, :, :] = x.astype(np.float64)

    # Compute scale for each channel
    scale = np.ones_like(x, dtype=np.float64) * k
    for c in range(C):
        window = padded[:, c:c + size, :, :]
        scale[:, c, :, :] += (alpha / size) * np.sum(window ** 2, axis=1)

    return (x.astype(np.float64) / (scale ** beta)).astype(np.float32)


def deconv1x1_np(x, weight, bias=None):
    """Numpy reference for 1x1 Deconvolution (no bias, no padding, stride=1).

    Deconv with 1x1 kernel is equivalent to: y[c_out, h, w] = sum_c_in W[c_in, c_out] * x[c_in, h, w] + bias[c_out]
    i.e., matrix multiplication in channel dimension: W^T @ x + b
    Weight shape for 1x1 deconv: (num_input_channels, num_output_channels, 1, 1)
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[1]
    W2d = weight.reshape(C_in, C_out)  # (C_in, C_out)
    # x: (N, C_in, H, W) -> (N*H*W, C_in)
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C_in)
    y_flat = x_flat @ W2d  # (N*H*W, C_out)
    if bias is not None:
        y_flat = y_flat + bias.astype(np.float64)
    return y_flat.reshape(N, H, W, C_out).transpose(0, 3, 1, 2).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Test Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_slice_prototxt(num_tops, axis=1, slice_points=None):
    """Create minimal Input->Slice prototxt."""
    tops_str = "\n".join(f'  top: "slice_{i}"' for i in range(num_tops))
    sp = ""
    if slice_points:
        sp_lines = "\n".join(f"    slice_point: {p}" for p in slice_points)
        sp = f"\n{sp_lines}"
    return f"""name: "test_slice"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 1 dim: 6 dim: 3 dim: 3 }} }}
}}
layer {{
  name: "slice"
  type: "Slice"
  bottom: "data"
{tops_str}
  slice_param {{
    axis: {axis}{sp}
  }}
}}
"""


def _make_crop_prototxt(inp_shape, ref_shape, axis=2, offsets=None):
    """Create Input(data) -> Input(shape_ref) -> Crop prototxt."""
    inp_dims = " ".join(f"dim: {d}" for d in inp_shape)
    ref_dims = " ".join(f"dim: {d}" for d in ref_shape)
    off_str = ""
    if offsets:
        off_vals = "\n    ".join(f"offset: {o}" for o in offsets)
        off_str = f"\n    {off_vals}"
    return f"""name: "test_crop"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ {inp_dims} }} }}
}}
layer {{
  name: "shape_ref"
  type: "Input"
  top: "shape_ref"
  input_param {{ shape {{ {ref_dims} }} }}
}}
layer {{
  name: "crop"
  type: "Crop"
  bottom: "data"
  bottom: "shape_ref"
  top: "cropped"
  crop_param {{
    axis: {axis}{off_str}
  }}
}}
"""


def _make_lrn_prototxt(batch=1, channels=6, h=4, w=4, size=5, alpha=1e-4, beta=0.75, k=1.0):
    """Create Input->LRN prototxt."""
    return f"""name: "test_lrn"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: {batch} dim: {channels} dim: {h} dim: {w} }} }}
}}
layer {{
  name: "lrn"
  type: "LRN"
  bottom: "data"
  top: "lrn_out"
  lrn_param {{
    local_size: {size}
    alpha: {alpha}
    beta: {beta}
    k: {k}
  }}
}}
"""


def _make_deconv1x1_prototxt(batch, c_in, h, w, c_out, bias_term=False):
    """Create Input->Deconvolution(1x1) prototxt."""
    bias_str = "    bias_term: true" if bias_term else "    bias_term: false"
    return f"""name: "test_deconv"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: {batch} dim: {c_in} dim: {h} dim: {w} }} }}
}}
layer {{
  name: "deconv"
  type: "Deconvolution"
  bottom: "data"
  top: "deconv_out"
  convolution_param {{
    num_output: {c_out}
    kernel_size: 1
    stride: 1
    pad: 0
{bias_str}
  }}
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Tests: Slice Layer
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSliceLayers:
    """Slice layer forward tests: equal division, explicit slice_points, N=1 identity, axis selection."""

    def test_slice_equal_2way_channel(self, ptrace):
        """Slice N=2 along channel axis (6ch -> 3ch+3ch)."""
        proto = _make_slice_prototxt(num_tops=2, axis=1)
        with ptrace("Net(slice 2-way channel)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(42)
        inp = rng.randn(1, 6, 3, 3).astype(np.float32)
        with ptrace("slice 2-way forward"):
            out = net.forward({"data": inp})

        expected = slice_np(inp, 2, axis=1)
        assert len(out) == 2, f"Expected 2 outputs, got {len(out)}"
        np.testing.assert_allclose(out["slice_0"], expected[0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(out["slice_1"], expected[1], rtol=1e-6, atol=1e-8)

    def test_slice_equal_3way_channel(self, ptrace):
        """Slice N=3 along channel axis (6ch -> 2ch+2ch+2ch)."""
        proto = _make_slice_prototxt(num_tops=3, axis=1)
        with ptrace("Net(slice 3-way)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(123)
        inp = rng.randn(1, 6, 2, 2).astype(np.float32)
        with ptrace("slice 3-way forward"):
            out = net.forward({"data": inp})

        expected = slice_np(inp, 3, axis=1)
        for i in range(3):
            np.testing.assert_allclose(out[f"slice_{i}"], expected[i], rtol=1e-6, atol=1e-8)

    def test_slice_explicit_points(self, ptrace):
        """Slice with explicit slice_points: 6ch -> 1ch+2ch+3ch."""
        proto = _make_slice_prototxt(num_tops=3, axis=1, slice_points=[1, 3])
        with ptrace("Net(slice with slice_points)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(7)
        inp = rng.randn(1, 6, 2, 2).astype(np.float32)
        with ptrace("slice points forward"):
            out = net.forward({"data": inp})

        expected = slice_np(inp, 3, axis=1, slice_points=[1, 3])
        assert out["slice_0"].shape == (1, 1, 2, 2)
        assert out["slice_1"].shape == (1, 2, 2, 2)
        assert out["slice_2"].shape == (1, 3, 2, 2)
        for i in range(3):
            np.testing.assert_allclose(out[f"slice_{i}"], expected[i], rtol=1e-6, atol=1e-8)

    def test_slice_n1_identity(self, ptrace):
        """N=1 Slice is identity/passthrough (shares data with bottom)."""
        proto = _make_slice_prototxt(num_tops=1, axis=1)
        with ptrace("Net(slice N=1)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(99)
        inp = rng.randn(1, 6, 2, 2).astype(np.float32)
        with ptrace("slice N=1 forward"):
            out = net.forward({"data": inp})

        np.testing.assert_array_equal(out["slice_0"], inp)

    def test_slice_output_shapes(self, ptrace):
        """Verify output shapes match expected dimensions for various configurations."""
        # N=2: 6ch -> 3ch+3ch
        proto = _make_slice_prototxt(num_tops=2, axis=1)
        net = net_from_param(net_param_from_string(proto))
        inp = np.random.RandomState(1).randn(1, 6, 4, 4).astype(np.float32)
        out = net.forward({"data": inp})
        assert out["slice_0"].shape == (1, 3, 4, 4)
        assert out["slice_1"].shape == (1, 3, 4, 4)


# ═══════════════════════════════════════════════════════════════════════
# Tests: Crop Layer
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestCropLayers:
    """Crop layer forward tests: center crop, offset crop, axis selection."""

    def test_crop_center_hw(self, ptrace):
        """Crop spatial dims (axis=2): 4x4 -> 2x2 with offset (1,1)."""
        inp_shape = (1, 2, 4, 4)
        ref_shape = (1, 2, 2, 2)
        proto = _make_crop_prototxt(inp_shape, ref_shape, axis=2, offsets=[1, 1])
        with ptrace("Net(crop center 4x4->2x2)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(42)
        data = rng.randn(*inp_shape).astype(np.float32)
        ref = np.zeros(ref_shape, dtype=np.float32)
        with ptrace("crop forward"):
            out = net.forward({"data": data, "shape_ref": ref})

        expected = crop_np(data, ref_shape, axis=2, offsets=[1, 1])
        np.testing.assert_allclose(out["cropped"], expected, rtol=1e-6, atol=1e-8)

    def test_crop_no_offset(self, ptrace):
        """Crop with default offset=0: 5x5 -> 3x3 from top-left."""
        inp_shape = (1, 3, 5, 5)
        ref_shape = (1, 3, 3, 3)
        proto = _make_crop_prototxt(inp_shape, ref_shape, axis=2)
        with ptrace("Net(crop no offset)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(7)
        data = rng.randn(*inp_shape).astype(np.float32)
        ref = np.zeros(ref_shape, dtype=np.float32)
        with ptrace("crop no-offset forward"):
            out = net.forward({"data": data, "shape_ref": ref})

        expected = crop_np(data, ref_shape, axis=2, offsets=[0, 0])
        np.testing.assert_allclose(out["cropped"], expected, rtol=1e-6, atol=1e-8)
        assert out["cropped"].shape == ref_shape

    def test_crop_axis1_channels(self, ptrace):
        """Crop along channel axis (axis=1): 6ch -> 3ch with offset=2 for C, 0 for H,W.

        C++ Crop layer semantics: when axis=1, offsets[0] applies to axis 1 (C),
        offsets[1] to axis 2 (H), offsets[2] to axis 3 (W). Single offset is
        broadcast to all dims from axis onward, so we must provide per-axis offsets
        when we only want to crop channels.
        """
        inp_shape = (1, 6, 2, 2)
        ref_shape = (1, 3, 2, 2)
        proto = _make_crop_prototxt(inp_shape, ref_shape, axis=1, offsets=[2, 0, 0])
        with ptrace("Net(crop axis=1)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(13)
        data = rng.randn(*inp_shape).astype(np.float32)
        ref = np.zeros(ref_shape, dtype=np.float32)
        with ptrace("crop axis=1 forward"):
            out = net.forward({"data": data, "shape_ref": ref})

        expected = crop_np(data, ref_shape, axis=1, offsets=[2, 0, 0])
        np.testing.assert_allclose(out["cropped"], expected, rtol=1e-6, atol=1e-8)

    def test_crop_single_offset_all_dims(self, ptrace):
        """Crop with single offset value broadcast to all dims from axis=1.

        With axis=1 and offsets=[1], the same offset=1 is applied to C, H, W.
        Input: (1,4,6,6), Ref: (1,3,5,5) -> offset=1 on C (4-1=3), H (6-1=5), W (6-1=5).
        """
        inp_shape = (1, 4, 6, 6)
        ref_shape = (1, 3, 5, 5)
        proto = _make_crop_prototxt(inp_shape, ref_shape, axis=1, offsets=[1])
        with ptrace("Net(crop single offset broadcast)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(55)
        data = rng.randn(*inp_shape).astype(np.float32)
        ref = np.zeros(ref_shape, dtype=np.float32)
        with ptrace("crop single-offset forward"):
            out = net.forward({"data": data, "shape_ref": ref})

        # Single offset=1 broadcast to all dims from axis=1 (C, H, W)
        expected = crop_np(data, ref_shape, axis=1, offsets=[1, 1, 1])
        np.testing.assert_allclose(out["cropped"], expected, rtol=1e-6, atol=1e-8)

    def test_crop_output_shape(self, ptrace):
        """Verify output shape matches reference shape exactly."""
        inp_shape = (2, 4, 8, 8)
        ref_shape = (2, 4, 3, 5)
        proto = _make_crop_prototxt(inp_shape, ref_shape, axis=2, offsets=[2, 1])
        net = net_from_param(net_param_from_string(proto))
        data = np.random.randn(*inp_shape).astype(np.float32)
        ref = np.zeros(ref_shape, dtype=np.float32)
        out = net.forward({"data": data, "shape_ref": ref})
        assert out["cropped"].shape == ref_shape


# ═══════════════════════════════════════════════════════════════════════
# Tests: LRN Layer
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestLRNLayers:
    """LRN layer forward tests: AlexNet defaults, custom params, uniform input."""

    def test_lrn_alexnet_defaults(self, ptrace):
        """LRN with AlexNet defaults: size=5, alpha=1e-4, beta=0.75, k=1."""
        proto = _make_lrn_prototxt(batch=1, channels=6, h=4, w=4)
        with ptrace("Net(LRN alexnet defaults)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(42)
        inp = rng.randn(1, 6, 4, 4).astype(np.float32) * 0.1
        with ptrace("LRN forward"):
            out = net.forward({"data": inp})

        expected = lrn_np(inp, size=5, alpha=1e-4, beta=0.75, k=1.0)
        # LRN involves pow/sum operations; use relaxed tolerance (1e-3) for float32
        np.testing.assert_allclose(out["lrn_out"], expected, rtol=1e-3, atol=1e-4)

    def test_lrn_custom_params(self, ptrace):
        """LRN with custom parameters: size=3, alpha=0.001, beta=0.5, k=2."""
        proto = _make_lrn_prototxt(batch=1, channels=4, h=3, w=3,
                                   size=3, alpha=0.001, beta=0.5, k=2.0)
        with ptrace("Net(LRN custom)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(99)
        inp = rng.randn(1, 4, 3, 3).astype(np.float32) * 0.5
        with ptrace("LRN custom forward"):
            out = net.forward({"data": inp})

        expected = lrn_np(inp, size=3, alpha=0.001, beta=0.5, k=2.0)
        np.testing.assert_allclose(out["lrn_out"], expected, rtol=1e-3, atol=1e-4)

    def test_lrn_uniform_input(self, ptrace):
        """LRN with uniform input x=c: y = c / (k + alpha*c^2)^beta."""
        proto = _make_lrn_prototxt(batch=1, channels=6, h=2, w=2,
                                   size=5, alpha=1e-4, beta=0.75, k=1.0)
        with ptrace("Net(LRN uniform)"):
            net = net_from_param(net_param_from_string(proto))

        c = 0.5
        inp = np.full((1, 6, 2, 2), c, dtype=np.float32)
        with ptrace("LRN uniform forward"):
            out = net.forward({"data": inp})

        # For uniform input with enough channels, scale ≈ k + alpha * c^2 (full window)
        # Edge channels have fewer neighbors in window
        # Check output is finite and shape correct
        assert out["lrn_out"].shape == inp.shape
        assert np.all(np.isfinite(out["lrn_out"]))

    def test_lrn_zero_input(self, ptrace):
        """LRN with zero input should produce zero output."""
        proto = _make_lrn_prototxt(batch=1, channels=4, h=3, w=3)
        with ptrace("Net(LRN zero)"):
            net = net_from_param(net_param_from_string(proto))

        inp = np.zeros((1, 4, 3, 3), dtype=np.float32)
        with ptrace("LRN zero forward"):
            out = net.forward({"data": inp})

        expected = lrn_np(inp)
        np.testing.assert_allclose(out["lrn_out"], expected, rtol=1e-6, atol=1e-8)

    def test_lrn_output_shape(self, ptrace):
        """LRN output shape matches input shape."""
        proto = _make_lrn_prototxt(batch=2, channels=8, h=5, w=5)
        net = net_from_param(net_param_from_string(proto))
        inp = np.random.randn(2, 8, 5, 5).astype(np.float32)
        out = net.forward({"data": inp})
        assert out["lrn_out"].shape == inp.shape


# ═══════════════════════════════════════════════════════════════════════
# Tests: Deconvolution Layer (1x1 kernel, shape verification, bias)
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestDeconvolutionLayers:
    """Deconvolution layer tests: 1x1 kernel correctness, shape verification, bias."""

    def test_deconv_1x1_identity_uprojection(self, ptrace):
        """1x1 Deconv with identity-ish weights: C_in=3 -> C_out=3 should preserve spatial.

        For 1x1 deconv with stride=1, pad=0: output H=H_in, W=W_in (same as input spatial).
        The 1x1 deconv computes y = W^T @ x per spatial location.
        """
        proto = _make_deconv1x1_prototxt(batch=1, c_in=3, h=3, w=3, c_out=3, bias_term=False)
        with ptrace("Net(deconv 1x1 3->3)"):
            net = net_from_param(net_param_from_string(proto))

        # Set W to identity matrix: W shape (C_in, C_out, 1, 1) = (3, 3, 1, 1)
        W = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1)
        deconv_layer = net.layer_by_name("deconv")
        with ptrace("load identity W"):
            deconv_layer.blobs[0].from_numpy(W.reshape(3, -1))

        rng = np.random.RandomState(42)
        inp = rng.randn(1, 3, 3, 3).astype(np.float32) * 0.1
        with ptrace("deconv 1x1 forward"):
            out = net.forward({"data": inp})

        expected = deconv1x1_np(inp, W)
        # Deconv involves GEMM; rtol=1e-4 for float32 accumulation
        np.testing.assert_allclose(out["deconv_out"], expected, rtol=1e-4, atol=1e-5)

    def test_deconv_1x1_channel_projection(self, ptrace):
        """1x1 Deconv projects channels: C_in=2 -> C_out=4 with random weights."""
        proto = _make_deconv1x1_prototxt(batch=1, c_in=2, h=2, w=2, c_out=4, bias_term=False)
        with ptrace("Net(deconv 1x1 2->4)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(123)
        W = rng.randn(2, 4, 1, 1).astype(np.float32) * 0.1
        deconv_layer = net.layer_by_name("deconv")
        with ptrace("load random W"):
            deconv_layer.blobs[0].from_numpy(W.reshape(2, -1))

        inp = rng.randn(1, 2, 2, 2).astype(np.float32) * 0.5
        with ptrace("deconv projection forward"):
            out = net.forward({"data": inp})

        expected = deconv1x1_np(inp, W)
        np.testing.assert_allclose(out["deconv_out"], expected, rtol=1e-4, atol=1e-5)
        assert out["deconv_out"].shape == (1, 4, 2, 2)

    def test_deconv_1x1_with_bias(self, ptrace):
        """1x1 Deconv with bias term adds per-channel bias correctly."""
        proto = _make_deconv1x1_prototxt(batch=1, c_in=2, h=2, w=2, c_out=3, bias_term=True)
        with ptrace("Net(deconv 1x1 with bias)"):
            net = net_from_param(net_param_from_string(proto))

        rng = np.random.RandomState(456)
        W = rng.randn(2, 3, 1, 1).astype(np.float32) * 0.1
        b = rng.randn(3).astype(np.float32) * 0.01
        deconv_layer = net.layer_by_name("deconv")
        with ptrace("load W+b"):
            deconv_layer.blobs[0].from_numpy(W.reshape(2, -1))
            deconv_layer.blobs[1].from_numpy(b)

        inp = rng.randn(1, 2, 2, 2).astype(np.float32) * 0.3
        with ptrace("deconv + bias forward"):
            out = net.forward({"data": inp})

        expected = deconv1x1_np(inp, W, bias=b)
        np.testing.assert_allclose(out["deconv_out"], expected, rtol=1e-4, atol=1e-5)

    def test_deconv_output_shape_1x1(self, ptrace):
        """1x1 Deconv preserves spatial dimensions (H_out=H_in, W_out=W_in)."""
        for c_in, c_out, h, w in [(2, 4, 3, 5), (3, 2, 4, 4), (1, 8, 1, 1)]:
            proto = _make_deconv1x1_prototxt(batch=1, c_in=c_in, h=h, w=w, c_out=c_out)
            net = net_from_param(net_param_from_string(proto))
            inp = np.random.randn(1, c_in, h, w).astype(np.float32)
            out = net.forward({"data": inp})
            assert out["deconv_out"].shape == (1, c_out, h, w), \
                f"Expected (1,{c_out},{h},{w}), got {out['deconv_out'].shape}"

    def test_deconv_stride2_shape(self, ptrace):
        """Deconv with stride=2 upsamples spatial dims: H_out = stride*(H-1) + kernel."""
        prototxt = """name: "test_deconv_stride2"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 2 dim: 3 dim: 3 } }
}
layer {
  name: "deconv"
  type: "Deconvolution"
  bottom: "data"
  top: "deconv_out"
  convolution_param {
    num_output: 2
    kernel_size: 2
    stride: 2
    pad: 0
    bias_term: false
  }
}
"""
        with ptrace("Net(deconv stride=2, k=2)"):
            net = net_from_param(net_param_from_string(prototxt))

        inp = np.random.randn(1, 2, 3, 3).astype(np.float32) * 0.01
        with ptrace("deconv stride2 forward"):
            out = net.forward({"data": inp})

        # H_out = stride*(H-1) + kernel - 2*pad = 2*(3-1) + 2 - 0 = 6
        assert out["deconv_out"].shape == (1, 2, 6, 6), \
            f"Expected (1,2,6,6), got {out['deconv_out'].shape}"
        assert np.all(np.isfinite(out["deconv_out"]))


# ═══════════════════════════════════════════════════════════════════════
# Combination: Slice + Concat roundtrip
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSliceConcatRoundtrip:
    """Slice followed by Concat should reconstruct original tensor."""

    def test_slice_concat_roundtrip_3way(self, ptrace):
        """Slice N=3 -> Concat N=3 should return original data."""
        prototxt = """name: "slice_concat_rt"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 6 dim: 2 dim: 2 } }
}
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "s0"
  top: "s1"
  top: "s2"
  slice_param { axis: 1 }
}
layer {
  name: "concat"
  type: "Concat"
  bottom: "s0"
  bottom: "s1"
  bottom: "s2"
  top: "recon"
  concat_param { axis: 1 }
}
"""
        with ptrace("Net(slice->concat)"):
            net = net_from_param(net_param_from_string(prototxt))

        rng = np.random.RandomState(2024)
        inp = rng.randn(1, 6, 2, 2).astype(np.float32)
        with ptrace("slice->concat forward"):
            out = net.forward({"data": inp})

        np.testing.assert_array_equal(out["recon"], inp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
