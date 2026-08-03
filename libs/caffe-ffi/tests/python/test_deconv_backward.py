"""Deconvolution (transposed convolution) layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi dX/dW/db)
     - 1x1 kernel (equivalent to channel-wise matrix multiply)
  2. Numerical gradient check (central finite differences via _grad_check_utils)
     - 1x1 kernel: dX, dW, db
     - 2x2 kernel stride=2 (upsampling): dX, dW, db
  3. Configurations: no-bias, with-bias, stride=2 upsampling
  4. Zero dy -> zero gradients
  5. Shape/dtype/finite/determinism checks
  6. Forward output preserved after backward

Mathematical reference (1x1 deconv):
  Forward:   y = W^T @ x + b   (per spatial position; W shape (Ci, Co, 1, 1))
  Backward:
    dW = x_flat^T @ dy_flat    (accumulated over batch*spatial)
    db = sum(dy over N*Ho*Wo)
    dX = dy_flat @ W           (per spatial position)

For general kernels, Deconv backward uses transposed GEMM + col2im/im2col
(already implemented in C++); numerical gradient tests verify correctness
without needing an analytical numpy reference for the full im2col path.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import (
    assert_grad_close,
    numerical_grad_for_blob,
    numerical_grad_for_input,
)

EPS_NUMERICAL = 1e-3


# ---------------------------------------------------------------------------
# Numpy reference for 1x1 Deconv backward
# ---------------------------------------------------------------------------

def deconv1x1_backward_np(x, dy, weight, bias=None):
    """Numpy reference for 1x1 Deconvolution backward (stride=1, pad=0).

    Args:
        x: Input (N, Ci, H, W)
        dy: Upstream gradient (N, Co, H, W)  -- same spatial for 1x1 s1p0
        weight: (Ci, Co, 1, 1)
        bias: (Co,) or None

    Returns:
        (dX, dW, db) where db is None if bias is None
    """
    N, Ci, H, W = x.shape
    Co = weight.shape[1]
    W2d = weight.reshape(Ci, Co).astype(np.float64)
    x64 = x.astype(np.float64)
    dy64 = dy.astype(np.float64)

    x_flat = x64.transpose(0, 2, 3, 1).reshape(-1, Ci)  # (N*H*W, Ci)
    dy_flat = dy64.transpose(0, 2, 3, 1).reshape(-1, Co)  # (N*H*W, Co)

    dW_flat = x_flat.T @ dy_flat  # (Ci, Co)
    dW = dW_flat.reshape(Ci, Co, 1, 1).astype(np.float32)

    dx_flat = dy_flat @ W2d  # (N*H*W, Ci)
    dX = dx_flat.reshape(N, H, W, Ci).transpose(0, 3, 1, 2).astype(np.float32)

    db = None
    if bias is not None:
        db = dy_flat.sum(axis=0).astype(np.float32)

    return dX, dW, db


# ---------------------------------------------------------------------------
# Prototxt builders
# ---------------------------------------------------------------------------

def _make_deconv_prototxt(input_dims, num_output, kernel_size,
                          pad=0, stride=1, bias_term=True,
                          weight_filler="constant", weight_value=0.0,
                          bias_filler="constant", bias_value=0.0):
    """Create Input -> Deconvolution prototxt."""
    dims_str = " ".join(str(d) for d in input_dims)
    bias_str = "true" if bias_term else "false"
    return textwrap.dedent(f"""\
        name: "test_deconv_bw"
        input: "data"
        input_dim: {input_dims[0]}
        input_dim: {input_dims[1]}
        input_dim: {input_dims[2]}
        input_dim: {input_dims[3]}
        layer {{
          name: "deconv"
          type: "Deconvolution"
          bottom: "data"
          top: "deconv"
          convolution_param {{
            num_output: {num_output}
            kernel_size: {kernel_size}
            pad: {pad}
            stride: {stride}
            bias_term: {bias_str}
            weight_filler {{ type: "{weight_filler}" value: {weight_value} }}
            bias_filler {{ type: "{bias_filler}" value: {bias_value} }}
          }}
        }}
    """)


def _make_deconv_net(N, Ci, H, W, Co, Kh, Kw=None, pad=0, stride=1, bias=True):
    if Kw is None:
        Kw = Kh
    input_dims = (N, Ci, H, W)
    proto = _make_deconv_prototxt(input_dims, Co, kernel_size=Kh,
                                  pad=pad, stride=stride, bias_term=bias)
    return Net(proto)


def _set_deconv_weights(net, W, b=None):
    """Set Deconv layer weights. W shape for Deconv: (Ci, Co/g, Kh, Kw)."""
    deconv_layer = net.layer_by_name("deconv")
    # Net expects flat weights; reshape to 2D (Co, -1) like Caffe blob
    # Actually, deconv blobs[0] is (conv_out_channels_, conv_in_channels_/group, Kh, Kw)
    # which for Deconv is (Ci, Co, Kh, Kw)
    deconv_layer.blobs[0].from_numpy(W.astype(np.float32))
    if b is not None and len(deconv_layer.blobs) >= 2:
        deconv_layer.blobs[1].from_numpy(b.reshape(-1).astype(np.float32))


def _run_deconv_backward(net, x, dy, W, b=None):
    """Run forward then backward, return (y, dX, dW, db)."""
    _set_deconv_weights(net, W, b)
    out = net.forward({"data": x.astype(np.float32)})
    net.backward({"deconv": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    dW = net.layer_by_name("deconv").blobs[0].diff
    db = None
    if b is not None and len(net.layer_by_name("deconv").blobs) >= 2:
        db = net.layer_by_name("deconv").blobs[1].diff
    return out["deconv"], dX, dW, db


# ---------------------------------------------------------------------------
# Test Class 1: 1x1 Deconv backward (simplest case, analytical reference)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestDeconvBackward1x1:
    """1x1 Deconvolution backward tests (stride=1, pad=0, no im2col)."""

    def test_deconv1x1_known_values(self):
        """1x1 deconv identity W=I, dy=ones: dX=dy, dW=X^T@ones, db=H*W."""
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = np.eye(Ci, Co, dtype=np.float32).reshape(Ci, Co, 1, 1)
        b = np.zeros(Co, dtype=np.float32)
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]],
                       [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
        dy = np.ones((N, Co, H, W_dim), dtype=np.float32)

        y, dX, dW, db = _run_deconv_backward(net, x, dy, W, b)
        # dX = dy @ W = dy @ I = dy
        np.testing.assert_allclose(dX, dy, rtol=1e-5)
        # db = sum(dy) = 4 for each channel
        np.testing.assert_allclose(db, np.full(Co, 4.0, dtype=np.float32), rtol=1e-5)
        # dW[i,o] = sum_{n,h,w} x[n,i,h,w] * dy[n,o,h,w] = sum(x over spatial)
        expected_dW = np.array([[[[10.0]], [[10.0]]],
                                [[[26.0]], [[26.0]]]], dtype=np.float32)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5)

    def test_deconv1x1_analytical_dx_dw_db(self):
        """1x1 deconv: caffe-ffi gradients vs numpy reference."""
        rng = np.random.RandomState(222)
        N, Ci, H, W_dim, Co = 2, 3, 3, 3, 4
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = rng.randn(Ci, Co, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.2

        y, dX, dW, db = _run_deconv_backward(net, x, dy, W, b)
        exp_dX, exp_dW, exp_db = deconv1x1_backward_np(x, dy, W, b)
        np.testing.assert_allclose(dX, exp_dX, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dW, exp_dW, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(db, exp_db, rtol=1e-4, atol=1e-5)

    def test_deconv1x1_numerical_dx_dw_db(self):
        """1x1 deconv: numerical gradient check for dX, dW, db."""
        rng = np.random.RandomState(333)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 3
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = rng.randn(Ci, Co, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.1

        _set_deconv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"deconv": dy})
        analytic_dX = net.blob_by_name("data").diff
        analytic_dW = net.layer_by_name("deconv").blobs[0].diff
        analytic_db = net.layer_by_name("deconv").blobs[1].diff

        num_dX = numerical_grad_for_input(
            net, "data", x, "deconv", dy, h=EPS_NUMERICAL,
            name="deconv1x1_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, num_dX, name="dX(deconv 1x1)",
                          rtol=1e-2, atol=1e-3)

        num_dW = numerical_grad_for_blob(
            net, "deconv", 0, {"data": x}, "deconv", dy, h=EPS_NUMERICAL,
            name="deconv1x1_dW", verbose=True,
        )
        assert_grad_close(analytic_dW, num_dW, name="dW(deconv 1x1)",
                          rtol=1e-2, atol=1e-3)

        num_db = numerical_grad_for_blob(
            net, "deconv", 1, {"data": x}, "deconv", dy, h=EPS_NUMERICAL,
            name="deconv1x1_db", verbose=True,
        )
        assert_grad_close(analytic_db, num_db, name="db(deconv 1x1)",
                          rtol=1e-2, atol=1e-3)

    def test_deconv1x1_no_bias(self):
        """1x1 deconv no bias: gradients vs numpy reference."""
        rng = np.random.RandomState(444)
        N, Ci, H, W_dim, Co = 1, 3, 2, 2, 4
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=False)
        W = rng.randn(Ci, Co, 1, 1).astype(np.float32) * 0.3
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.2

        y, dX, dW, db = _run_deconv_backward(net, x, dy, W, None)
        exp_dX, exp_dW, _ = deconv1x1_backward_np(x, dy, W, None)
        np.testing.assert_allclose(dX, exp_dX, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dW, exp_dW, rtol=1e-4, atol=1e-5)
        assert db is None

    def test_deconv1x1_numerical_no_bias(self):
        """1x1 deconv no bias: numerical gradient check dX, dW."""
        rng = np.random.RandomState(555)
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 3
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=False)
        W = rng.randn(Ci, Co, 1, 1).astype(np.float32) * 0.4
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.1

        _set_deconv_weights(net, W)
        net.forward({"data": x})
        net.backward({"deconv": dy})
        a_dX = net.blob_by_name("data").diff
        a_dW = net.layer_by_name("deconv").blobs[0].diff

        n_dX = numerical_grad_for_input(net, "data", x, "deconv", dy, h=EPS_NUMERICAL,
                                         name="deconv1x1_nobias_dx")
        assert_grad_close(a_dX, n_dX, name="dX(deconv1x1 nobias)", rtol=1e-2, atol=1e-3)

        n_dW = numerical_grad_for_blob(net, "deconv", 0, {"data": x}, "deconv", dy,
                                        h=EPS_NUMERICAL, name="deconv1x1_nobias_dW")
        assert_grad_close(a_dW, n_dW, name="dW(deconv1x1 nobias)", rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 2: 2x2 Deconv stride=2 (upsampling)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestDeconvBackwardStride2:
    """2x2 Deconvolution stride=2 backward (upsampling, uses im2col/col2im)."""

    def test_deconv_2x2_s2_numerical_dx_dw_db(self):
        """2x2 deconv stride=2: numerical gradient check dX, dW, db (tiny input)."""
        rng = np.random.RandomState(666)
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2
        # Output size: s*(H-1)+k = 1*(2-1)+2 = 3? Wait stride=2: s*(H-1)+k = 2*1+2=4
        # compute_output_shape: output_h = stride*(height-1) + dilation*(k-1)+1 - 2*pad
        # = 2*(2-1) + 1*(2-1)+1 - 0 = 2+2=4, same for w -> 4x4
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=2, stride=2, bias=True)
        W = rng.randn(Ci, Co, 2, 2).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5

        _set_deconv_weights(net, W, b)
        out = net.forward({"data": x})
        assert out["deconv"].shape == (N, Co, 4, 4), f"Expected upsampled 4x4, got {out['deconv'].shape}"

        dy = rng.randn(N, Co, 4, 4).astype(np.float32) * 0.1
        net.backward({"deconv": dy})
        a_dX = net.blob_by_name("data").diff
        a_dW = net.layer_by_name("deconv").blobs[0].diff
        a_db = net.layer_by_name("deconv").blobs[1].diff

        n_dX = numerical_grad_for_input(net, "data", x, "deconv", dy, h=EPS_NUMERICAL,
                                         name="deconv2x2s2_dx", verbose=True)
        assert_grad_close(a_dX, n_dX, name="dX(deconv 2x2 s2)", rtol=1e-2, atol=2e-3)

        n_dW = numerical_grad_for_blob(net, "deconv", 0, {"data": x}, "deconv", dy,
                                        h=EPS_NUMERICAL, name="deconv2x2s2_dW", verbose=True)
        assert_grad_close(a_dW, n_dW, name="dW(deconv 2x2 s2)", rtol=1e-2, atol=2e-3)

        n_db = numerical_grad_for_blob(net, "deconv", 1, {"data": x}, "deconv", dy,
                                        h=EPS_NUMERICAL, name="deconv2x2s2_db", verbose=True)
        assert_grad_close(a_db, n_db, name="db(deconv 2x2 s2)", rtol=1e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# Test Class 3: Zero dy and edge cases
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestDeconvBackwardEdgeCases:
    """Edge case tests for Deconv backward."""

    def test_zero_dy_zero_gradients(self):
        """Zero dy should produce zero dX, dW, db."""
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = np.random.RandomState(0).randn(Ci, Co, 1, 1).astype(np.float32) * 0.3
        b = np.random.RandomState(1).randn(Co).astype(np.float32) * 0.1
        x = np.random.RandomState(2).randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        dy = np.zeros((N, Co, H, W_dim), dtype=np.float32)

        _, dX, dW, db = _run_deconv_backward(net, x, dy, W, b)
        np.testing.assert_array_equal(dX, np.zeros_like(dX))
        np.testing.assert_array_equal(dW, np.zeros_like(dW))
        np.testing.assert_array_equal(db, np.zeros_like(db))

    def test_deterministic(self):
        """Same input -> same gradients (determinism)."""
        rng = np.random.RandomState(77)
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2
        net1 = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        net2 = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = rng.randn(Ci, Co, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.2

        _, dX1, dW1, db1 = _run_deconv_backward(net1, x, dy, W, b)
        _, dX2, dW2, db2 = _run_deconv_backward(net2, x, dy, W, b)
        np.testing.assert_array_equal(dX1, dX2)
        np.testing.assert_array_equal(dW1, dW2)
        np.testing.assert_array_equal(db1, db2)

    def test_shapes_dtypes(self):
        """dX, dW, db have correct shapes and dtypes."""
        rng = np.random.RandomState(88)
        N, Ci, H, W_dim, Co = 2, 3, 4, 4, 4
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = rng.randn(Ci, Co, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32)
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32)

        _, dX, dW, db = _run_deconv_backward(net, x, dy, W, b)
        assert dX.shape == (N, Ci, H, W_dim)
        assert dW.shape == (Ci, Co, 1, 1)
        assert db.shape == (Co,)
        assert dX.dtype == np.float32
        assert dW.dtype == np.float32
        assert db.dtype == np.float32
        assert np.all(np.isfinite(dX))
        assert np.all(np.isfinite(dW))
        assert np.all(np.isfinite(db))

    def test_forward_preserved_after_backward(self):
        """Forward output data is unchanged after Backward."""
        rng = np.random.RandomState(99)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        net = _make_deconv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = rng.randn(Ci, Co, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32)
        _set_deconv_weights(net, W, b)
        out = net.forward({"data": x})
        y_before = out["deconv"].copy()
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.1
        net.backward({"deconv": dy})
        y_after = net.blob_by_name("deconv").data
        np.testing.assert_array_equal(y_before, y_after)
