"""Convolution layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX/dW/db)
  2. Numerical gradient check (central finite differences)
  3. Known-value verification (1x1 identity conv, hand-computed)
  4. 1x1 conv backward (no padding/stride)
  5. 3x3 conv with padding and stride
  6. Bias gradient correctness
  7. Group convolution backward
  8. Zero dy → zero gradients
  9. Shape/dtype/finite/determinism checks
 10. Forward output preserved after backward

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


def _run_conv_backward(net, x, dy, W, b=None, pad=0, stride=1, dilation=1, groups=1):
    """Run forward then backward, return (dX, dW, db)."""
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
# Numerical gradient helpers
# ---------------------------------------------------------------------------

def _num_grad_x(net, x, W, b, dy, pad=0, stride=1, h=EPS_NUMERICAL):
    """Numerical gradient dX via central differences."""
    grad = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        xp = x.copy(); xp.flat[i] += h
        xm = x.copy(); xm.flat[i] -= h
        _set_conv_weights(net, W, b)
        yp = net.forward({"data": xp.astype(np.float32)})["conv"]
        _set_conv_weights(net, W, b)
        ym = net.forward({"data": xm.astype(np.float32)})["conv"]
        grad.flat[i] = float(np.sum(dy.astype(np.float64) * (yp.astype(np.float64) - ym.astype(np.float64)))) / (2 * h)
    return grad.astype(np.float32)


def _num_grad_W(net, x, W, b, dy, pad=0, stride=1, h=EPS_NUMERICAL):
    """Numerical gradient dW via central differences."""
    grad = np.zeros_like(W, dtype=np.float64)
    for i in range(W.size):
        Wp = W.copy(); Wp.flat[i] += h
        Wm = W.copy(); Wm.flat[i] -= h
        _set_conv_weights(net, Wp, b)
        yp = net.forward({"data": x.astype(np.float32)})["conv"]
        _set_conv_weights(net, Wm, b)
        ym = net.forward({"data": x.astype(np.float32)})["conv"]
        grad.flat[i] = float(np.sum(dy.astype(np.float64) * (yp.astype(np.float64) - ym.astype(np.float64)))) / (2 * h)
    return grad.astype(np.float32)


def _num_grad_b(net, x, W, b, dy, pad=0, stride=1, h=EPS_NUMERICAL):
    """Numerical gradient db via central differences."""
    grad = np.zeros_like(b, dtype=np.float64)
    for i in range(b.size):
        bp = b.copy(); bp.flat[i] += h
        bm = b.copy(); bm.flat[i] -= h
        _set_conv_weights(net, W, bp)
        yp = net.forward({"data": x.astype(np.float32)})["conv"]
        _set_conv_weights(net, W, bm)
        ym = net.forward({"data": x.astype(np.float32)})["conv"]
        grad.flat[i] = float(np.sum(dy.astype(np.float64) * (yp.astype(np.float64) - ym.astype(np.float64)))) / (2 * h)
    return grad.astype(np.float32)


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
        # dX = dy @ W = dy @ I = dy = ones
        np.testing.assert_allclose(dX, dy, rtol=1e-5)
        # db = sum(dy over spatial+batch) = N*H*W = 4 for each channel
        np.testing.assert_allclose(db, np.full(Co, 4.0, dtype=np.float32), rtol=1e-5)
        # dW = sum over n: dy_n^T @ col_n; for 1x1 col_n = X_n reshaped (Ci, H*W)
        # dy_n is (Co, H*W) all ones, X_n is (Ci, H*W) -> dW_g = dy_n @ X_n^T = (Co, Ci)
        # dW[co, ci] = sum over h,w of dy[co,h,w] * x[ci,h,w]
        # ch0 sum = 1+2+3+4 = 10; ch1 sum = 5+6+7+8 = 26
        expected_dW = np.array([[[[10.0]], [[26.0]]],
                                [[[10.0]], [[26.0]]]], dtype=np.float32)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5)

    def test_conv1x1_analytical_dx_dw_db(self):
        """1x1 conv: caffe-ffi gradients vs numpy reference."""
        np.random.seed(111)
        N, Ci, H, W_dim, Co = 2, 3, 3, 3, 4
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = np.random.randn(Co, Ci, 1, 1).astype(np.float32) * 0.3
        b = np.random.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, stride=1, pad=0)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, stride=1, pad=0)

        np.testing.assert_allclose(dX, dX_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dW, dW_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(db, db_ref, rtol=1e-4, atol=1e-5)

    def test_conv1x1_no_bias(self):
        """1x1 conv without bias: caffe-ffi dX/dW vs numpy reference."""
        np.random.seed(222)
        N, Ci, H, W_dim, Co = 2, 3, 2, 2, 5
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=False)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = np.random.randn(Co, Ci, 1, 1).astype(np.float32) * 0.3
        y = conv_forward(x, W, b=None, stride=1, pad=0)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b=None)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=None, stride=1, pad=0)

        assert db is None or db.size == 0, "No bias blob expected"
        np.testing.assert_allclose(dX, dX_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dW, dW_ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Test Class 2: 3x3 Conv with padding/stride
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackward3x3:
    """3x3 convolution backward tests with padding/stride."""

    def test_conv3x3_pad1_analytical(self):
        """3x3 conv pad=1 stride=1: analytical dX/dW/db vs numpy reference."""
        np.random.seed(333)
        N, Ci, H, W_dim, Co = 1, 2, 4, 4, 2
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = np.random.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        b = np.random.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, pad=pad)

        np.testing.assert_allclose(dX, dX_ref, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dW, dW_ref, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(db, db_ref, rtol=1e-3, atol=1e-4)

    def test_conv3x3_stride2_analytical(self):
        """3x3 conv pad=1 stride=2: analytical dX/dW vs numpy reference."""
        np.random.seed(444)
        N, Ci, H, W_dim, Co = 1, 1, 4, 4, 1
        Kh = Kw = 3
        pad = 1
        stride = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, stride=stride, bias=False)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = np.random.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        y = conv_forward(x, W, b=None, pad=pad, stride=stride)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b=None, pad=pad, stride=stride)
        dX_ref, dW_ref, _ = conv_backward(dy, x, W, b=None, pad=pad, stride=stride)

        np.testing.assert_allclose(dX, dX_ref, rtol=5e-3, atol=5e-4)
        np.testing.assert_allclose(dW, dW_ref, rtol=5e-3, atol=5e-4)

    def test_conv3x3_numerical_dx(self):
        """3x3 pad=1 stride=1: numerical gradient check for dX on tiny tensor."""
        np.random.seed(555)
        N, Ci, H, W_dim, Co = 1, 1, 3, 3, 1  # 9 input elements
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = np.random.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.2
        b = np.random.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.2

        dX_analytic, _, _ = _run_conv_backward(net, x, dy, W, b, pad=pad)
        dX_numeric = _num_grad_x(net, x, W, b, dy, pad=pad)
        np.testing.assert_allclose(dX_analytic, dX_numeric, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 3: Group convolution
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackwardGroups:
    """Group convolution backward tests."""

    def test_conv_groups2_analytical(self):
        """groups=2, 1x1 per group: analytical dX/dW vs numpy reference."""
        np.random.seed(666)
        N, Ci, H, W_dim, Co = 1, 4, 2, 2, 4
        groups = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, groups=groups, bias=False)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = np.random.randn(Co, Ci // groups, 1, 1).astype(np.float32) * 0.2
        y = conv_forward(x, W, b=None, groups=groups)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, _ = _run_conv_backward(net, x, dy, W, b=None, groups=groups)
        dX_ref, dW_ref, _ = conv_backward(dy, x, W, b=None, groups=groups)

        np.testing.assert_allclose(dX, dX_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dW, dW_ref, rtol=1e-4, atol=1e-5)

    def test_conv_groups2_3x3_analytical(self):
        """groups=2, 3x3 pad=1: analytical dX/dW vs numpy reference."""
        np.random.seed(777)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        groups = 2
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, groups=groups, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = np.random.randn(Co, Ci // groups, Kh, Kw).astype(np.float32) * 0.2
        b = np.random.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b=b, pad=pad, groups=groups)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.2

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad, groups=groups)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, pad=pad, groups=groups)

        np.testing.assert_allclose(dX, dX_ref, rtol=5e-3, atol=5e-4)
        np.testing.assert_allclose(dW, dW_ref, rtol=5e-3, atol=5e-4)
        np.testing.assert_allclose(db, db_ref, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test Class 4: Common invariants
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConvBackwardInvariants:
    """Common invariants: zero dy, shape, determinism, forward preservation."""

    def test_conv_zero_dy_gives_zero_grads(self):
        """Zero upstream gradient → zero dX, dW, db."""
        N, Ci, H, W_dim, Co = 2, 2, 3, 3, 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32)
        W = np.random.randn(Co, Ci, 3, 3).astype(np.float32) * 0.3
        b = np.random.randn(Co).astype(np.float32) * 0.1
        dy = np.zeros((N, Co, H, W_dim), dtype=np.float32)

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=1)
        np.testing.assert_array_equal(dX, np.zeros_like(dX))
        np.testing.assert_array_equal(dW, np.zeros_like(dW))
        np.testing.assert_array_equal(db, np.zeros_like(db))

    def test_conv_backward_shapes(self):
        """dX matches input shape, dW matches weight shape, db matches Co."""
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 4
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32)
        W = np.random.randn(Co, Ci, 3, 3).astype(np.float32) * 0.2
        b = np.zeros(Co, dtype=np.float32)
        y = conv_forward(x, W, b, pad=1)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.2

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=1)
        assert dX.shape == x.shape
        assert dW.shape == W.shape
        assert db.shape == (Co,)
        assert dX.dtype == np.float32
        assert dW.dtype == np.float32
        assert db.dtype == np.float32
        assert np.all(np.isfinite(dX))
        assert np.all(np.isfinite(dW))
        assert np.all(np.isfinite(db))

    def test_conv_backward_deterministic(self):
        """Same inputs → same dX/dW/db across calls."""
        np.random.seed(888)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = np.random.randn(Co, Ci, 3, 3).astype(np.float32) * 0.2
        b = np.random.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=1)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.2

        dX1, dW1, db1 = _run_conv_backward(net, x, dy, W, b, pad=1)
        dX2, dW2, db2 = _run_conv_backward(net, x, dy, W, b, pad=1)
        dX3, dW3, db3 = _run_conv_backward(net, x, dy, W, b, pad=1)
        np.testing.assert_array_equal(dX1, dX2)
        np.testing.assert_array_equal(dX1, dX3)
        np.testing.assert_array_equal(dW1, dW2)
        np.testing.assert_array_equal(dW1, dW3)
        np.testing.assert_array_equal(db1, db2)
        np.testing.assert_array_equal(db1, db3)

    def test_conv_backward_preserves_forward(self):
        """Backward does not corrupt forward output."""
        np.random.seed(999)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=3, pad=1, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = np.random.randn(Co, Ci, 3, 3).astype(np.float32) * 0.2
        b = np.random.randn(Co).astype(np.float32) * 0.1
        dy = np.random.randn(N, Co, H, W_dim).astype(np.float32) * 0.2

        _set_conv_weights(net, W, b)
        y_before = net.forward({"data": x})["conv"].copy()
        net.backward({"conv": dy})
        y_after = net.blob_by_name("conv").data
        np.testing.assert_array_equal(y_before, y_after)

    def test_conv_numerical_dw_db(self):
        """1x1 conv: numerical gradient check for dW and db."""
        np.random.seed(1010)
        N, Ci, H, W_dim, Co = 1, 2, 1, 2, 2  # very small for numerical speed
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, bias=True)
        x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = np.random.randn(Co, Ci, 1, 1).astype(np.float32) * 0.2
        b = np.random.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b)
        dy = np.random.randn(*y.shape).astype(np.float32) * 0.2

        _, dW_a, db_a = _run_conv_backward(net, x, dy, W, b)
        dW_n = _num_grad_W(net, x, W, b, dy)
        db_n = _num_grad_b(net, x, W, b, dy)
        np.testing.assert_allclose(dW_a, dW_n, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(db_a, db_n, rtol=1e-3, atol=1e-4)
