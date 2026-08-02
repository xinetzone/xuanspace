"""Convolution layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX/dW/db)
  2. Numerical gradient check (central finite differences via _grad_check_utils)
  3. Known-value verification (1x1 identity conv, hand-computed)
  4. 1x1 conv backward (no padding/stride)
  5. 3x3 conv with padding and stride
  6. 3x3 conv with dilation=2 (atrous convolution)
  7. Stride=2 numerical gradient check
  8. Bias gradient correctness
  9. Group convolution backward (analytical + numerical)
  10. No-bias configuration
  11. Zero dy → zero gradients
  12. Shape/dtype/finite/determinism checks
  13. Forward output preserved after backward
  14. Detailed gradient diagnostic logging on mismatch

Mathematical reference:
  Forward: Y = conv2d(X, W) + b   (im2col + GEMM)
  Backward:
    dW = im2col(X)^T @ dy_flat  (accumulated over batch)
    dX = col2im(W^T @ dy_flat)
    db = sum(dy over N, Ho, Wo)
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._numpy_conv_reference import conv_forward, conv_backward
from ._grad_check_utils import (
    assert_grad_close,
    compare_gradients,
    numerical_grad_for_blob,
    numerical_grad_for_input,
)

EPS_NUMERICAL = 1e-3


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_conv_prototxt(input_dims, num_output, kernel_size,
                        pad=0, stride=1, dilation=1, groups=1,
                        bias_term=True,
                        weight_filler="constant", weight_value=0.0,
                        bias_filler="constant", bias_value=0.0):
    """Create Input -> Convolution prototxt."""
    dims_str = " ".join(str(d) for d in input_dims)
    bias_str = "true" if bias_term else "false"
    return textwrap.dedent(f"""\
        name: "test_conv_bw"
        input: "data"
        input_dim: {input_dims[0]}
        input_dim: {input_dims[1]}
        input_dim: {input_dims[2]}
        input_dim: {input_dims[3]}
        layer {{
          name: "conv"
          type: "Convolution"
          bottom: "data"
          top: "conv"
          convolution_param {{
            num_output: {num_output}
            kernel_size: {kernel_size}
            pad: {pad}
            stride: {stride}
            dilation: {dilation}
            group: {groups}
            bias_term: {bias_str}
            weight_filler {{ type: "{weight_filler}" value: {weight_value} }}
            bias_filler {{ type: "{bias_filler}" value: {bias_value} }}
          }}
        }}
    """)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _make_conv_net(N, Ci, H, W, Co, Kh, Kw=None, pad=0, stride=1, dilation=1,
                   groups=1, bias=True):
    if Kw is None:
        Kw = Kh
    input_dims = (N, Ci, H, W)
    proto = _make_conv_prototxt(input_dims, Co, kernel_size=Kh,
                                pad=pad, stride=stride, dilation=dilation,
                                groups=groups, bias_term=bias)
    return Net(proto)


def _set_conv_weights(net, W, b=None):
    """Set Conv layer weights (4D: Co, Ci/g, Kh, Kw) and optionally bias."""
    conv_layer = net.layer_by_name("conv")
    conv_layer.blobs[0].from_numpy(W.astype(np.float32))
    if b is not None and len(conv_layer.blobs) >= 2:
        conv_layer.blobs[1].from_numpy(b.reshape(-1).astype(np.float32))


def _run_conv_backward(net, x, dy, W, b=None, pad=0, stride=1, dilation=1, groups=1,
                       log_label=""):
    """Run forward then backward, return (dX, dW, db) with optional diagnostic logging."""
    _set_conv_weights(net, W, b)
    net.forward({"data": x.astype(np.float32)})
    net.backward({"conv": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    dW = net.layer_by_name("conv").blobs[0].diff
    db = None
    if b is not None and len(net.layer_by_name("conv").blobs) >= 2:
        db = net.layer_by_name("conv").blobs[1].diff
    return dX, dW, db


# ---------------------------------------------------------------------------
# Test Class 1: 1x1 Conv backward (simplest case, no im2col needed)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackward1x1:
    """1x1 convolution backward tests (Kh=Kw=1, pad=0, stride=1)."""

    def test_conv1x1_known_identity(self):
        """1x1 identity W=I, b=0, dy=ones → dW = dy^T @ X, dX=dy, db=sum(dy)."""
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        W = np.eye(Co, Ci, dtype=np.float32).reshape(Co, Ci, 1, 1)
        b = np.zeros(Co, dtype=np.float32)
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]],
                       [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
        dy = np.ones((N, Co, H, W_dim), dtype=np.float32)

        dX, dW, db = _run_conv_backward(net, x, dy, W, b)
        np.testing.assert_allclose(dX, dy, rtol=1e-5)
        np.testing.assert_allclose(db, np.full(Co, 4.0, dtype=np.float32), rtol=1e-5)
        expected_dW = np.array([[[[10.0]], [[26.0]]],
                                [[[10.0]], [[26.0]]]], dtype=np.float32)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5)

    def test_conv1x1_analytical_dx_dw_db(self):
        """1x1 conv: caffe-ffi gradients vs numpy reference (detailed logging)."""
        rng = np.random.RandomState(111)
        N, Ci, H, W_dim, Co = 2, 3, 3, 3, 4
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = rng.randn(Co, Ci, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, stride=1, pad=0)

        assert_grad_close(dX, dX_ref, name="dX (1x1 conv)", rtol=1e-4, atol=1e-5)
        assert_grad_close(dW, dW_ref, name="dW (1x1 conv)", rtol=1e-4, atol=1e-5)
        assert_grad_close(db, db_ref, name="db (1x1 conv)", rtol=1e-4, atol=1e-5)

    def test_conv1x1_no_bias(self):
        """1x1 conv without bias: caffe-ffi dX/dW vs numpy reference."""
        rng = np.random.RandomState(222)
        N, Ci, H, W_dim, Co = 2, 3, 2, 2, 5
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=False)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = rng.randn(Co, Ci, 1, 1).astype(np.float32) * 0.3
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b=None)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=None, stride=1, pad=0)

        assert db is None, "No bias blob expected when bias_term=false"
        assert_grad_close(dX, dX_ref, name="dX (1x1 no-bias)", rtol=1e-4, atol=1e-5)
        assert_grad_close(dW, dW_ref, name="dW (1x1 no-bias)", rtol=1e-4, atol=1e-5)

    def test_conv1x1_numerical_dx(self):
        """1x1 conv: numerical gradient check for dX (small tensor)."""
        rng = np.random.RandomState(1111)
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2  # 8 input elements
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = rng.randn(Co, Ci, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (1x1, num)",
        )
        assert_grad_close(dx_analytic, dx_numeric, name="dX (1x1)", rtol=1e-2, atol=1e-3)

    def test_conv1x1_numerical_dw_db(self):
        """1x1 conv: numerical gradient check for dW and db."""
        rng = np.random.RandomState(1112)
        N, Ci, H, W_dim, Co = 1, 2, 1, 2, 2  # W: 8 elements, b: 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = rng.randn(Co, Ci, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dw_analytic = net.layer_by_name("conv").blobs[0].diff
        db_analytic = net.layer_by_name("conv").blobs[1].diff

        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (1x1, num)",
        )
        db_numeric = numerical_grad_for_blob(
            net, "conv", 1, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="db (1x1, num)",
        )
        assert_grad_close(dw_analytic, dw_numeric, name="dW (1x1)", rtol=1e-3, atol=1e-4)
        assert_grad_close(db_analytic, db_numeric, name="db (1x1)", rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test Class 2: 3x3 Conv with padding/stride
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackward3x3:
    """3x3 convolution backward tests with padding/stride."""

    def test_conv3x3_pad1_analytical(self):
        """3x3 conv pad=1 stride=1: analytical dX/dW/db vs numpy reference."""
        rng = np.random.RandomState(333)
        N, Ci, H, W_dim, Co = 1, 2, 4, 4, 2
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, pad=pad)

        assert_grad_close(dX, dX_ref, name="dX (3x3 pad=1)", rtol=1e-3, atol=1e-4)
        assert_grad_close(dW, dW_ref, name="dW (3x3 pad=1)", rtol=1e-3, atol=1e-4)
        assert_grad_close(db, db_ref, name="db (3x3 pad=1)", rtol=1e-3, atol=1e-4)

    def test_conv3x3_stride2_analytical(self):
        """3x3 conv pad=1 stride=2: analytical dX/dW vs numpy reference."""
        rng = np.random.RandomState(444)
        N, Ci, H, W_dim, Co = 1, 1, 4, 4, 1
        Kh = Kw = 3
        pad = 1
        stride = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, stride=stride, bias=False)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        y = conv_forward(x, W, b=None, pad=pad, stride=stride)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, _ = _run_conv_backward(net, x, dy, W, b=None, pad=pad, stride=stride)
        dX_ref, dW_ref, _ = conv_backward(dy, x, W, b=None, pad=pad, stride=stride)

        assert_grad_close(dX, dX_ref, name="dX (3x3 stride=2)", rtol=5e-3, atol=5e-4)
        assert_grad_close(dW, dW_ref, name="dW (3x3 stride=2)", rtol=5e-3, atol=5e-4)

    def test_conv3x3_numerical_dx(self):
        """3x3 pad=1 stride=1: numerical gradient check for dX on tiny tensor."""
        rng = np.random.RandomState(555)
        N, Ci, H, W_dim, Co = 1, 1, 3, 3, 1  # 9 input elements
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = rng.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (3x3 pad=1, num)",
        )
        assert_grad_close(dx_analytic, dx_numeric, name="dX (3x3 pad=1)", rtol=1e-2, atol=1e-3)

    def test_conv3x3_stride2_numerical_dx(self):
        """3x3 pad=1 stride=2: numerical gradient for dX (tiny tensor)."""
        rng = np.random.RandomState(556)
        N, Ci, H, W_dim, Co = 1, 1, 4, 4, 1
        Kh = Kw = 3
        pad = 1
        stride = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, stride=stride, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = rng.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad, stride=stride)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (stride=2, num)",
        )
        assert_grad_close(dx_analytic, dx_numeric, name="dX (stride=2)", rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# Test Class 3: Dilated convolution (atrous)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackwardDilation:
    """Dilated (atrous) convolution backward tests."""

    def test_conv_dilation2_analytical(self):
        """3x3 dilation=2 pad=2 (same output size as 'same'): analytical dX/dW/db."""
        rng = np.random.RandomState(2000)
        N, Ci, H, W_dim, Co = 1, 1, 5, 5, 1
        Kh = Kw = 3
        pad = 2
        dilation = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, dilation=dilation, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad, dilation=dilation)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad, dilation=dilation)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, pad=pad, dilation=dilation)

        assert_grad_close(dX, dX_ref, name="dX (dilation=2)", rtol=1e-3, atol=1e-4)
        assert_grad_close(dW, dW_ref, name="dW (dilation=2)", rtol=1e-3, atol=1e-4)
        assert_grad_close(db, db_ref, name="db (dilation=2)", rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test Class 4: Group convolution
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackwardGroups:
    """Group convolution backward tests."""

    def test_conv_groups2_analytical(self):
        """groups=2, 1x1 per group: analytical dX/dW vs numpy reference."""
        rng = np.random.RandomState(666)
        N, Ci, H, W_dim, Co = 1, 4, 2, 2, 4
        groups = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, groups=groups, bias=False)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci // groups, 1, 1).astype(np.float32) * 0.2
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        dX, dW, _ = _run_conv_backward(net, x, dy, W, b=None, groups=groups)
        dX_ref, dW_ref, _ = conv_backward(dy, x, W, b=None, groups=groups)

        assert_grad_close(dX, dX_ref, name="dX (groups=2, 1x1)", rtol=1e-4, atol=1e-5)
        assert_grad_close(dW, dW_ref, name="dW (groups=2, 1x1)", rtol=1e-4, atol=1e-5)

    def test_conv_groups2_3x3_analytical(self):
        """groups=2, 3x3 pad=1: analytical dX/dW vs numpy reference."""
        rng = np.random.RandomState(777)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        groups = 2
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, groups=groups, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = rng.randn(Co, Ci // groups, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.2

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad, groups=groups)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, pad=pad, groups=groups)

        assert_grad_close(dX, dX_ref, name="dX (groups=2, 3x3)", rtol=5e-3, atol=5e-4)
        assert_grad_close(dW, dW_ref, name="dW (groups=2, 3x3)", rtol=5e-3, atol=5e-4)
        assert_grad_close(db, db_ref, name="db (groups=2, 3x3)", rtol=1e-3, atol=1e-4)

    def test_conv_groups2_numerical(self):
        """groups=2, 1x1: numerical gradient for dX/dW on tiny tensor."""
        rng = np.random.RandomState(667)
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2
        groups = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, groups=groups, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = rng.randn(Co, Ci // groups, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff
        dw_analytic = net.layer_by_name("conv").blobs[0].diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (groups=2, num)",
        )
        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (groups=2, num)",
        )
        assert_grad_close(dx_analytic, dx_numeric, name="dX (groups=2)", rtol=1e-2, atol=1e-3)
        assert_grad_close(dw_analytic, dw_numeric, name="dW (groups=2)", rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 5: Common invariants
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackwardInvariants:
    """Common invariants: zero dy, shape, determinism, forward preservation."""

    def test_conv_zero_dy_gives_zero_grads(self):
        """Zero upstream gradient → zero dX, dW, db."""
        N, Ci, H, W_dim, Co = 2, 2, 3, 3, 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        rng = np.random.RandomState(3000)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32)
        W = rng.randn(Co, Ci, 3, 3).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = np.zeros((N, Co, H, W_dim), dtype=np.float32)

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=1)
        np.testing.assert_array_equal(dX, np.zeros_like(dX))
        np.testing.assert_array_equal(dW, np.zeros_like(dW))
        np.testing.assert_array_equal(db, np.zeros_like(db))

    def test_conv_backward_shapes(self):
        """dX matches input shape, dW matches weight shape, db matches Co."""
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 4
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        rng = np.random.RandomState(3001)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32)
        W = rng.randn(Co, Ci, 3, 3).astype(np.float32) * 0.2
        b = np.zeros(Co, dtype=np.float32)
        y = conv_forward(x, W, b, pad=1)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=1)
        assert dX.shape == x.shape, f"dX shape {dX.shape} != x shape {x.shape}"
        assert dW.shape == W.shape, f"dW shape {dW.shape} != W shape {W.shape}"
        assert db.shape == (Co,), f"db shape {db.shape} != ({Co},)"
        assert dX.dtype == np.float32
        assert dW.dtype == np.float32
        assert db.dtype == np.float32
        assert np.all(np.isfinite(dX)), "dX contains NaN/Inf"
        assert np.all(np.isfinite(dW)), "dW contains NaN/Inf"
        assert np.all(np.isfinite(db)), "db contains NaN/Inf"

    def test_conv_backward_deterministic(self):
        """Same inputs → same dX/dW/db across repeated calls."""
        rng = np.random.RandomState(888)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci, 3, 3).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=1)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        results = []
        for _ in range(3):
            dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=1)
            results.append((dX.copy(), dW.copy(), db.copy()))
        for i in range(1, 3):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            np.testing.assert_array_equal(results[0][1], results[i][1])
            np.testing.assert_array_equal(results[0][2], results[i][2])

    def test_conv_backward_preserves_forward(self):
        """Backward does not corrupt forward output."""
        rng = np.random.RandomState(999)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci, 3, 3).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.2

        _set_conv_weights(net, W, b)
        y_before = net.forward({"data": x})["conv"].copy()
        net.backward({"conv": dy})
        y_after = net.blob_by_name("conv").data
        np.testing.assert_array_equal(y_before, y_after)

    def test_conv_batch_gradient_accumulation(self):
        """Verify gradients accumulate correctly over batch dimension (N>1).

        dW and db should be summed over N samples.
        Compare N=2 against running N=1 twice and summing manually.
        """
        rng = np.random.RandomState(4000)
        Ci, H, W_dim, Co = 1, 3, 3, 1
        Kh = Kw = 3
        pad = 1

        # Single sample networks
        net1 = _make_conv_net(1, Ci, H, W_dim, Co, Kh=Kh, pad=pad, bias=True)
        net2 = _make_conv_net(1, Ci, H, W_dim, Co, Kh=Kh, pad=pad, bias=True)
        # Batch network
        net_batch = _make_conv_net(2, Ci, H, W_dim, Co, Kh=Kh, pad=pad, bias=True)

        W = rng.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        x1 = rng.randn(1, Ci, H, W_dim).astype(np.float32) * 0.5
        x2 = rng.randn(1, Ci, H, W_dim).astype(np.float32) * 0.5
        dy1 = rng.randn(1, Co, H, W_dim).astype(np.float32) * 0.3
        dy2 = rng.randn(1, Co, H, W_dim).astype(np.float32) * 0.3

        # Run single samples
        _set_conv_weights(net1, W, b)
        net1.forward({"data": x1})
        net1.backward({"conv": dy1})
        dW1 = net1.layer_by_name("conv").blobs[0].diff.copy()
        db1 = net1.layer_by_name("conv").blobs[1].diff.copy()
        dX1 = net1.blob_by_name("data").diff.copy()

        _set_conv_weights(net2, W, b)
        net2.forward({"data": x2})
        net2.backward({"conv": dy2})
        dW2 = net2.layer_by_name("conv").blobs[0].diff.copy()
        db2 = net2.layer_by_name("conv").blobs[1].diff.copy()
        dX2 = net2.blob_by_name("data").diff.copy()

        # Run batch
        x_batch = np.concatenate([x1, x2], axis=0)
        dy_batch = np.concatenate([dy1, dy2], axis=0)
        _set_conv_weights(net_batch, W, b)
        net_batch.forward({"data": x_batch})
        net_batch.backward({"conv": dy_batch})
        dW_batch = net_batch.layer_by_name("conv").blobs[0].diff
        db_batch = net_batch.layer_by_name("conv").blobs[1].diff
        dX_batch = net_batch.blob_by_name("data").diff

        # dW and db should be SUM over batch
        dW_sum = dW1 + dW2
        db_sum = db1 + db2
        assert_grad_close(dW_batch, dW_sum, name="dW (batch accum)", rtol=1e-5, atol=1e-6,
                          verbose=False)
        assert_grad_close(db_batch, db_sum, name="db (batch accum)", rtol=1e-5, atol=1e-6,
                          verbose=False)
        # dX should be concatenation of per-sample dX
        dX_expected = np.concatenate([dX1, dX2], axis=0)
        assert_grad_close(dX_batch, dX_expected, name="dX (batch accum)", rtol=1e-5, atol=1e-6,
                          verbose=False)
