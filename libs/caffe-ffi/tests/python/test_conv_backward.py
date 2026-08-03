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
  10. Depthwise convolution backward (groups=C, 3x3+pad+stride, analytical + numerical)
  11. No-bias configuration
  12. Zero dy → zero gradients
  13. Shape/dtype/finite/determinism checks
  14. Forward output preserved after backward
  15. Detailed gradient diagnostic logging on mismatch (cos_sim/norm_ratio/3x3 neighborhood/per-channel)

Mathematical reference:
  Forward: Y = conv2d(X, W) + b   (im2col + GEMM)
  Backward:
    dW = im2col(X)^T @ dy_flat  (accumulated over batch)
    dX = col2im(W^T @ dy_flat)
    db = sum(dy over N, Ho, Wo)
  Depthwise (groups=C): each channel has its own 1-channel filter, no cross-talk.
"""
from __future__ import annotations

import logging
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
                   groups=1, bias=True, weight_filler="constant", weight_value=0.0,
                   bias_filler="constant", bias_value=0.0):
    if Kw is None:
        Kw = Kh
    input_dims = (N, Ci, H, W)
    proto = _make_conv_prototxt(input_dims, Co, kernel_size=Kh,
                                pad=pad, stride=stride, dilation=dilation,
                                groups=groups, bias_term=bias,
                                weight_filler=weight_filler, weight_value=weight_value,
                                bias_filler=bias_filler, bias_value=bias_value)
    return Net(proto)


def _set_conv_weights(net, W, b=None):
    """Set Conv layer weights (4D: Co, Ci/g, Kh, Kw) and optionally bias."""
    conv_layer = net.layer_by_name("conv")
    conv_layer.blobs[0].from_numpy(W.astype(np.float32))
    if b is not None and len(conv_layer.blobs) >= 2:
        conv_layer.blobs[1].from_numpy(b.reshape(-1).astype(np.float32))


_bw_logger = logging.getLogger("caffe_ffi.test.conv_bw")
if not _bw_logger.handlers:
    _bh = logging.StreamHandler()
    _bh.setFormatter(logging.Formatter(
        "%(asctime)s [CONV-BW] %(message)s", datefmt="%H:%M:%S",
    ))
    _bw_logger.addHandler(_bh)
    _bw_logger.propagate = False
_bw_logger.setLevel(logging.INFO)


def _run_conv_backward(net, x, dy, W, b=None, pad=0, stride=1, dilation=1, groups=1,
                       log_label=""):
    """Run forward then backward, return (dX, dW, db) with detailed diagnostic logging."""
    N, Ci, H, W_dim = x.shape
    Co, Ci_per_g, Kh, Kw = W.shape
    Co_per_g = Co // groups if groups > 0 else Co
    Ho = (H + 2 * pad - dilation * (Kh - 1) - 1) // stride + 1
    Wo = (W_dim + 2 * pad - dilation * (Kw - 1) - 1) // stride + 1

    _bw_logger.info(
        "%s_conv_bw: X=%s W=%s b=%s dy=%s  groups=%d pad=%d stride=%d dilation=%d  "
        "expected Ho=%d Wo=%d Ci/g=%d Co/g=%d",
        log_label, x.shape, W.shape,
        b.shape if b is not None else None, dy.shape,
        groups, pad, stride, dilation,
        Ho, Wo, Ci_per_g, Co_per_g,
    )

    _set_conv_weights(net, W, b)

    # Forward pass with output logging
    fwd_out = net.forward({"data": x.astype(np.float32)})
    y = fwd_out["conv"]
    _bw_logger.info(
        "%s_forward: output shape=%s  range=[%.3g, %.3g]  |y|=%.3g",
        log_label, y.shape, float(y.min()), float(y.max()),
        float(np.linalg.norm(y)),
    )

    # Check for NaN/Inf in forward output
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        _bw_logger.warning(
            "%s_forward: ⚠ NaN/Inf detected in forward output!", log_label,
        )

    net.backward({"conv": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    dW = net.layer_by_name("conv").blobs[0].diff
    db = None
    if b is not None and len(net.layer_by_name("conv").blobs) >= 2:
        db = net.layer_by_name("conv").blobs[1].diff

    # Log gradient quality diagnostics
    _bw_logger.info(
        "%s_backward: dX=%s range=[%.3g, %.3g] |dX|=%.3g  "
        "dW=%s range=[%.3g, %.3g] |dW|=%.3g",
        log_label, dX.shape, float(dX.min()), float(dX.max()), float(np.linalg.norm(dX)),
        dW.shape, float(dW.min()), float(dW.max()), float(np.linalg.norm(dW)),
    )
    if db is not None:
        _bw_logger.info(
            "%s_backward: db=%s range=[%.3g, %.3g] |db|=%.3g",
            log_label, db.shape, float(db.min()), float(db.max()), float(np.linalg.norm(db)),
        )

    # Finite value checks
    for name, arr in [("dX", dX), ("dW", dW), ("db", db)]:
        if arr is not None and (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
            _bw_logger.warning(
                "%s_backward: ⚠ NaN/Inf detected in %s!", log_label, name,
            )

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

_group_logger = logging.getLogger("caffe_ffi.test.group_conv")
if not _group_logger.handlers:
    _gh = logging.StreamHandler()
    _gh.setFormatter(logging.Formatter("%(asctime)s [GROUP-CONV] %(message)s", datefmt="%H:%M:%S"))
    _group_logger.addHandler(_gh)
    _group_logger.propagate = False
_group_logger.setLevel(logging.INFO)


def _log_group_diagnostics(dX, dW, db, dX_ref, dW_ref, db_ref, groups, name=""):
    """Per-group gradient diagnostic logging for GroupConv backward verification.

    Splits dX/dW/db by group and logs per-group error metrics to help isolate
    which group(s) have gradient mismatches.
    """
    N, C, H, W_dim = dX.shape
    Co, Ci_per_g, Kh, Kw = dW.shape
    Co_per_g = Co // groups
    Ci_per_g_check = C // groups

    _group_logger.info("=== GroupConv diagnostic [%s]: groups=%d  dX=(%d,%d,%d,%d) dW=(%d,%d,%d,%d) ===",
                       name, groups, N, C, H, W_dim, Co, Ci_per_g, Kh, Kw)

    for g in range(groups):
        # dX per group: channels [g*Ci/g : (g+1)*Ci/g]
        cx_sl = slice(g * Ci_per_g_check, (g + 1) * Ci_per_g_check)
        dX_g = dX[:, cx_sl]
        dX_g_ref = dX_ref[:, cx_sl]
        dx_err = np.abs(dX_g - dX_g_ref)
        dx_max = float(dx_err.max()) if dx_err.size > 0 else 0.0

        # dW per group: output channels [g*Co/g : (g+1)*Co/g]
        cw_sl = slice(g * Co_per_g, (g + 1) * Co_per_g)
        dW_g = dW[cw_sl]
        dW_g_ref = dW_ref[cw_sl]
        dw_err = np.abs(dW_g - dW_g_ref)
        dw_max = float(dw_err.max()) if dw_err.size > 0 else 0.0

        # db per group
        db_err_str = ""
        if db is not None and db_ref is not None:
            db_g = db[cw_sl]
            db_g_ref = db_ref[cw_sl]
            db_max = float(np.abs(db_g - db_g_ref).max())
            db_err_str = f"  db_max_err={db_max:.2e}"

        _group_logger.info(
            "  group %d: dX_max_err=%.2e (ch %d-%d)  dW_max_err=%.2e (och %d-%d)%s",
            g, dx_max, g * Ci_per_g_check, (g + 1) * Ci_per_g_check - 1,
            dw_max, g * Co_per_g, (g + 1) * Co_per_g - 1, db_err_str,
        )


@require_cpp_extension
class TestConvBackwardGroups:
    """Group convolution backward tests.

    GroupConv splits input channels into G independent groups:
      - Input: X (N, Ci, H, W) split into G groups of size Ci/G
      - Weights: W (Co, Ci/G, Kh, Kw) split into G groups of size Co/G
      - Each group operates independently: Y_g = conv(X_g, W_g) + b_g
      - Backward gradients accumulate per-group with no cross-talk
    """

    def test_conv_groups2_known_identity(self):
        """groups=2, 1x1, W=1.0 constant, b=0, dy=ones → hand-computed dX/dW/db.

        Setup:
          N=1, Ci=2, H=2, W=2, Co=2, groups=2, Kh=Kw=1
          X = [[[[1,2],[3,4]]], [[[5,6],[7,8]]]]
          W = ones(2,1,1,1) → group 0: W[0,0]=1 on ch0; group 1: W[1,0]=1 on ch1
          b = [0, 0]
          dy = ones(1,2,2,2)

        Expected:
          Y_g = X_g (1x1 conv with W=1, b=0)
          dX_g = dy_g = ones (W^T @ dy = 1 @ ones)
          dW_g = sum_n X_g^T @ dy_g = sum of X_g values
            group 0 (ch0: [1,2,3,4]): dW[0,0,0,0] = 1+2+3+4 = 10
            group 1 (ch1: [5,6,7,8]): dW[1,0,0,0] = 5+6+7+8 = 26
          db_g = sum over Ho*Wo of dy_g = 4 for each group
        """
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2
        groups = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, groups=groups, bias=True,
                             weight_value=1.0, bias_value=0.0)
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]],
                       [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
        W = np.ones((Co, Ci // groups, 1, 1), dtype=np.float32)
        b = np.zeros(Co, dtype=np.float32)
        dy = np.ones((N, Co, H, W_dim), dtype=np.float32)

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, groups=groups)

        _group_logger.info("test_conv_groups2_known_identity: dX=\n%s", dX[0])
        _group_logger.info("test_conv_groups2_known_identity: dW=%s db=%s",
                           dW.flatten(), db)

        # dX should be all ones (each group's dX = dy = ones)
        np.testing.assert_allclose(dX, dy, rtol=1e-5)
        # dW per group: ch0 sum=10, ch1 sum=26
        np.testing.assert_allclose(dW[0, 0, 0, 0], 10.0, rtol=1e-5)
        np.testing.assert_allclose(dW[1, 0, 0, 0], 26.0, rtol=1e-5)
        # db = sum of dy over H*W = 4 per output channel
        np.testing.assert_allclose(db, np.full(Co, 4.0, dtype=np.float32), rtol=1e-5)

    def test_conv_groups2_analytical(self):
        """groups=2, 1x1 per group: analytical dX/dW vs numpy reference (per-group log)."""
        rng = np.random.RandomState(666)
        N, Ci, H, W_dim, Co = 1, 4, 2, 2, 4
        groups = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, groups=groups, bias=False)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci // groups, 1, 1).astype(np.float32) * 0.2
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        dX, dW, _ = _run_conv_backward(net, x, dy, W, b=None, groups=groups)
        dX_ref, dW_ref, _ = conv_backward(dy, x, W, b=None, groups=groups)

        _log_group_diagnostics(dX, dW, None, dX_ref, dW_ref, None, groups,
                               name="groups=2 1x1 analytical")

        assert_grad_close(dX, dX_ref, name="dX (groups=2, 1x1)", rtol=1e-4, atol=1e-5)
        assert_grad_close(dW, dW_ref, name="dW (groups=2, 1x1)", rtol=1e-4, atol=1e-5)

    def test_conv_groups2_3x3_analytical(self):
        """groups=2, 3x3 pad=1: analytical dX/dW/db vs numpy reference (per-group log)."""
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

        _log_group_diagnostics(dX, dW, db, dX_ref, dW_ref, db_ref, groups,
                               name="groups=2 3x3 pad=1 analytical")

        assert_grad_close(dX, dX_ref, name="dX (groups=2, 3x3)", rtol=5e-3, atol=5e-4)
        assert_grad_close(dW, dW_ref, name="dW (groups=2, 3x3)", rtol=5e-3, atol=5e-4)
        assert_grad_close(db, db_ref, name="db (groups=2, 3x3)", rtol=1e-3, atol=1e-4)

    def test_conv_groups2_numerical(self):
        """groups=2, 1x1: numerical gradient for dX/dW/db on tiny tensor."""
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
        db_analytic = net.layer_by_name("conv").blobs[1].diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (groups=2, num)",
        )
        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (groups=2, num)",
        )
        db_numeric = numerical_grad_for_blob(
            net, "conv", 1, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="db (groups=2, num)",
        )

        assert_grad_close(dx_analytic, dx_numeric, name="dX (groups=2)", rtol=1e-2, atol=1e-3)
        assert_grad_close(dw_analytic, dw_numeric, name="dW (groups=2)", rtol=1e-2, atol=1e-3)
        assert_grad_close(db_analytic, db_numeric, name="db (groups=2)", rtol=1e-2, atol=1e-3)

    def test_conv_groups2_3x3_numerical_dw_db(self):
        """groups=2, 3x3 pad=1: numerical gradient for dW and db (small tensor)."""
        rng = np.random.RandomState(668)
        N, Ci, H, W_dim, Co = 1, 2, 3, 3, 2
        groups = 2
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, groups=groups, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = rng.randn(Co, Ci // groups, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad, groups=groups)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dw_analytic = net.layer_by_name("conv").blobs[0].diff
        db_analytic = net.layer_by_name("conv").blobs[1].diff

        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (groups=2, 3x3, num)",
        )
        db_numeric = numerical_grad_for_blob(
            net, "conv", 1, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="db (groups=2, 3x3, num)",
        )

        _group_logger.info("test_conv_groups2_3x3_numerical_dw_db: W shape=%s, b shape=%s",
                           dw_analytic.shape, db_analytic.shape)

        assert_grad_close(dw_analytic, dw_numeric, name="dW (groups=2, 3x3)",
                          rtol=2e-2, atol=2e-3)
        assert_grad_close(db_analytic, db_numeric, name="db (groups=2, 3x3)",
                          rtol=1e-2, atol=1e-3)

    def test_conv_groups4_analytical(self):
        """groups=4 (depthwise-like), 1x1: analytical dX/dW/db vs numpy reference."""
        rng = np.random.RandomState(669)
        N, Ci, H, W_dim, Co = 1, 4, 2, 2, 4
        groups = 4
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, groups=groups, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci // groups, 1, 1).astype(np.float32) * 0.3
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, groups=groups)
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, groups=groups)

        _log_group_diagnostics(dX, dW, db, dX_ref, dW_ref, db_ref, groups,
                               name="groups=4 1x1 analytical")

        assert_grad_close(dX, dX_ref, name="dX (groups=4)", rtol=1e-4, atol=1e-5)
        assert_grad_close(dW, dW_ref, name="dW (groups=4)", rtol=1e-4, atol=1e-5)
        assert_grad_close(db, db_ref, name="db (groups=4)", rtol=1e-4, atol=1e-5)

    def test_conv_groups4_numerical(self):
        """groups=4 (depthwise), 1x1: numerical gradient for dX/dW/db (tiny tensor)."""
        rng = np.random.RandomState(670)
        N, Ci, H, W_dim, Co = 1, 4, 2, 2, 4
        groups = 4
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
        db_analytic = net.layer_by_name("conv").blobs[1].diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (groups=4, num)",
        )
        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (groups=4, num)",
        )
        db_numeric = numerical_grad_for_blob(
            net, "conv", 1, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="db (groups=4, num)",
        )

        assert_grad_close(dx_analytic, dx_numeric, name="dX (groups=4)", rtol=1e-2, atol=1e-3)
        assert_grad_close(dw_analytic, dw_numeric, name="dW (groups=4)", rtol=1e-2, atol=1e-3)
        assert_grad_close(db_analytic, db_numeric, name="db (groups=4)", rtol=1e-2, atol=1e-3)

    def test_conv_groups_stride2_numerical(self):
        """groups=2, 3x3 pad=1 stride=2: numerical gradient for dX (small tensor)."""
        rng = np.random.RandomState(671)
        N, Ci, H, W_dim, Co = 1, 2, 4, 4, 2
        groups = 2
        Kh = Kw = 3
        pad = 1
        stride = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, stride=stride,
                             groups=groups, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.2
        W = rng.randn(Co, Ci // groups, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad, stride=stride, groups=groups)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_ref, dw_ref, db_ref = conv_backward(dy, x, W, b=b, pad=pad, stride=stride,
                                                groups=groups)

        _group_logger.info(
            "test_conv_groups_stride2_numerical: input=%s output=%s",
            x.shape, y.shape,
        )

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (groups=2, s=2, num)",
        )

        assert_grad_close(dx_analytic, dx_numeric, name="dX (groups=2, stride=2)",
                          rtol=2e-2, atol=2e-3)
        assert_grad_close(dx_analytic, dx_ref, name="dX (groups=2, stride=2, ref)",
                          rtol=5e-3, atol=5e-4)

    def test_conv_groups_no_bias_numerical(self):
        """groups=2, 1x1 no bias: numerical gradient for dX/dW."""
        rng = np.random.RandomState(672)
        N, Ci, H, W_dim, Co = 1, 2, 2, 2, 2
        groups = 2
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=1, groups=groups, bias=False)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
        W = rng.randn(Co, Ci // groups, 1, 1).astype(np.float32) * 0.3
        dy = rng.randn(N, Co, H, W_dim).astype(np.float32) * 0.3

        _set_conv_weights(net, W, None)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff
        dw_analytic = net.layer_by_name("conv").blobs[0].diff

        assert len(net.layer_by_name("conv").blobs) == 1, "No bias blob expected"

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (groups=2, nobias, num)",
        )
        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (groups=2, nobias, num)",
        )

        assert_grad_close(dx_analytic, dx_numeric, name="dX (groups=2, no-bias)",
                          rtol=1e-2, atol=1e-3)
        assert_grad_close(dw_analytic, dw_numeric, name="dW (groups=2, no-bias)",
                          rtol=1e-2, atol=1e-3)

    def test_conv_groups_zero_dy(self):
        """groups=2: zero dy → zero dX, dW, db for all groups."""
        rng = np.random.RandomState(673)
        N, Ci, H, W_dim, Co = 1, 4, 3, 3, 4
        groups = 2
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, Ci, H, W_dim, Co, Kh=Kh, pad=pad, groups=groups, bias=True)
        x = rng.randn(N, Ci, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(Co, Ci // groups, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(Co).astype(np.float32) * 0.1
        dy = np.zeros((N, Co, H, W_dim), dtype=np.float32)

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad, groups=groups)
        np.testing.assert_array_equal(dX, np.zeros_like(dX))
        np.testing.assert_array_equal(dW, np.zeros_like(dW))
        np.testing.assert_array_equal(db, np.zeros_like(db))


# ---------------------------------------------------------------------------
# Test Class 5: Common invariants
# ---------------------------------------------------------------------------

_dw_logger = logging.getLogger("caffe_ffi.test.depthwise_conv")
if not _dw_logger.handlers:
    _dwh = logging.StreamHandler()
    _dwh.setFormatter(logging.Formatter("%(asctime)s [DEPTHWISE-CONV] %(message)s", datefmt="%H:%M:%S"))
    _dw_logger.addHandler(_dwh)
    _dw_logger.propagate = False
_dw_logger.setLevel(logging.INFO)


def _log_depthwise_diagnostics(dX, dW, db, dX_ref, dW_ref, db_ref, C, name=""):
    """Per-channel gradient diagnostic for Depthwise Conv (groups=C, Ci/g=Co/g=1).

    In depthwise convolution each channel is fully independent, so per-channel
    error breakdown immediately isolates which channel(s) have bugs.
    """
    Co, Ci_per_g, Kh, Kw = dW.shape
    _dw_logger.info(
        "=== Depthwise diagnostic [%s]: C=%d dX=%s dW=%s  Ci/g=%d Co/g=%d Kh=%d Kw=%d ===",
        name, C, dX.shape, dW.shape, Ci_per_g, Co // C, Kh, Kw,
    )
    for c in range(C):
        dx_ch = dX[:, c]
        dx_ch_ref = dX_ref[:, c]
        dx_err = float(np.abs(dx_ch - dx_ch_ref).max())
        dw_ch = dW[c, 0]  # (Kh, Kw) since Ci/g=1
        dw_ch_ref = dW_ref[c, 0]
        dw_err = float(np.abs(dw_ch - dw_ch_ref).max())
        db_str = ""
        if db is not None and db_ref is not None:
            db_err = float(abs(db[c] - db_ref[c]))
            db_str = f"  db_err={db_err:.2e}"
        _dw_logger.info(
            "  ch %d: dX_max_err=%.2e  dW_max_err=%.2e  dW_range=[%.3g,%.3g]%s",
            c, dx_err, dw_err,
            float(dw_ch.min()), float(dw_ch.max()),
            db_str,
        )


@require_cpp_extension
class TestConvBackwardDepthwise:
    """Depthwise convolution backward tests (groups=Ci=Co, each channel independent).

    Depthwise conv is a special case of GroupConv where groups=C and each group
    has exactly 1 input and 1 output channel.  Common in MobileNet-family
    architectures.  These tests use 3x3 kernels with padding/stride (the
    practically relevant configuration) and include both analytical (numpy ref)
    and numerical (central-difference) gradient verification.
    """

    def test_depthwise_3x3_pad1_analytical(self):
        """Depthwise 3x3 pad=1 stride=1: analytical dX/dW/db vs numpy reference (per-channel log)."""
        rng = np.random.RandomState(800)
        N, C, H, W_dim = 1, 4, 3, 3
        groups = C
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, C, H, W_dim, C, Kh=Kh, pad=pad, groups=groups, bias=True)
        x = rng.randn(N, C, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(C, 1, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(C).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad, groups=groups)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.3

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad, groups=groups,
                                         log_label="depthwise_3x3_pad1")
        dX_ref, dW_ref, db_ref = conv_backward(dy, x, W, b=b, pad=pad, groups=groups)

        _log_depthwise_diagnostics(dX, dW, db, dX_ref, dW_ref, db_ref, C,
                                   name="depthwise 3x3 pad=1 analytical")

        assert_grad_close(dX, dX_ref, name="dX (depthwise 3x3 pad=1)", rtol=1e-3, atol=1e-4)
        assert_grad_close(dW, dW_ref, name="dW (depthwise 3x3 pad=1)", rtol=1e-3, atol=1e-4)
        assert_grad_close(db, db_ref, name="db (depthwise 3x3 pad=1)", rtol=1e-3, atol=1e-4)

    def test_depthwise_3x3_pad1_stride2_numerical(self):
        """Depthwise 3x3 pad=1 stride=2: numerical gradient for dX/dW/db (small tensor)."""
        rng = np.random.RandomState(801)
        N, C, H, W_dim = 1, 3, 4, 4
        groups = C
        Kh = Kw = 3
        pad = 1
        stride = 2
        net = _make_conv_net(N, C, H, W_dim, C, Kh=Kh, pad=pad, stride=stride,
                             groups=groups, bias=True)
        x = rng.randn(N, C, H, W_dim).astype(np.float32) * 0.2
        W = rng.randn(C, 1, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(C).astype(np.float32) * 0.1
        y = conv_forward(x, W, b, pad=pad, stride=stride, groups=groups)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        _set_conv_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff
        dw_analytic = net.layer_by_name("conv").blobs[0].diff
        db_analytic = net.layer_by_name("conv").blobs[1].diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (depthwise s=2, num)",
        )
        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (depthwise s=2, num)",
        )
        db_numeric = numerical_grad_for_blob(
            net, "conv", 1, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="db (depthwise s=2, num)",
        )

        _dw_logger.info(
            "test_depthwise_3x3_s2: input=%s output=%s W=%s (W elements=%d dX elements=%d)",
            x.shape, y.shape, W.shape, W.size, x.size,
        )

        assert_grad_close(dx_analytic, dx_numeric, name="dX (depthwise s=2)",
                          rtol=2e-2, atol=2e-3)
        assert_grad_close(dw_analytic, dw_numeric, name="dW (depthwise s=2)",
                          rtol=2e-2, atol=2e-3)
        assert_grad_close(db_analytic, db_numeric, name="db (depthwise s=2)",
                          rtol=1e-2, atol=1e-3)

    def test_depthwise_3x3_numerical_dx_dw(self):
        """Depthwise 3x3 pad=1 stride=1: numerical gradient for dX and dW on tiny tensor."""
        rng = np.random.RandomState(802)
        N, C, H, W_dim = 1, 2, 3, 3
        groups = C
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, C, H, W_dim, C, Kh=Kh, pad=pad, groups=groups, bias=False)
        x = rng.randn(N, C, H, W_dim).astype(np.float32) * 0.2
        W = rng.randn(C, 1, Kh, Kw).astype(np.float32) * 0.2
        y = conv_forward(x, W, b=None, pad=pad, groups=groups)
        dy = rng.randn(*y.shape).astype(np.float32) * 0.2

        _set_conv_weights(net, W, None)
        net.forward({"data": x})
        net.backward({"conv": dy})
        dx_analytic = net.blob_by_name("data").diff
        dw_analytic = net.layer_by_name("conv").blobs[0].diff

        dx_numeric = numerical_grad_for_input(
            net, "data", x, "conv", dy, h=EPS_NUMERICAL, name="dX (depthwise 3x3, num)",
        )
        dw_numeric = numerical_grad_for_blob(
            net, "conv", 0, {"data": x}, "conv", dy,
            h=EPS_NUMERICAL, name="dW (depthwise 3x3, num)",
        )

        assert_grad_close(dx_analytic, dx_numeric, name="dX (depthwise 3x3)",
                          rtol=2e-2, atol=2e-3)
        assert_grad_close(dw_analytic, dw_numeric, name="dW (depthwise 3x3)",
                          rtol=2e-2, atol=2e-3)

    def test_depthwise_known_identity(self):
        """Depthwise 3x3 pad=1 with W=identity-center (1 at center, 0 elsewhere), b=0.

        With W[c,0,1,1]=1 for each channel and 0 elsewhere, forward Y_c = X_c
        (centered 3x3 kernel extracts center element which equals input when pad=1
        keeps output size same).
        Backward: dX = dy (W^T @ dy with identity kernel = dy),
                  dW = X^T @ dy (cross-correlation between x and dy per channel),
                  db = sum(dy over spatial dims).
        """
        N, C, H, W_dim = 1, 2, 3, 3
        groups = C
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, C, H, W_dim, C, Kh=Kh, pad=pad, groups=groups, bias=True,
                             weight_value=0.0, bias_value=0.0)
        # Identity kernel: 1 at center position (Kh//2, Kw//2) = (1,1)
        W = np.zeros((C, 1, Kh, Kw), dtype=np.float32)
        W[:, 0, 1, 1] = 1.0
        b = np.zeros(C, dtype=np.float32)
        x = np.array([[[[1,2,3],[4,5,6],[7,8,9]],
                       [[9,8,7],[6,5,4],[3,2,1]]]], dtype=np.float32)
        dy = np.ones((N, C, H, W_dim), dtype=np.float32)

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad, groups=groups,
                                         log_label="depthwise_identity")

        # dX should equal dy (identity kernel backward = dy)
        np.testing.assert_allclose(dX, dy, rtol=1e-5)
        # db = sum over H*W = 9 per channel
        np.testing.assert_allclose(db, np.full(C, 9.0, dtype=np.float32), rtol=1e-5)
        # dW per channel: correlation with dy=ones → dW[c,0,i,j] = sum of X[c] at position (i,j)
        # over all spatial locations where kernel (i,j) overlaps input
        # For pad=1 stride=1 Kh=3, the 3x3 dW sums all x values
        for c in range(C):
            expected_dw_c = np.zeros((Kh, Kw), dtype=np.float32)
            xc = x[0, c]  # (H, W_dim) = (3,3)
            for kh in range(Kh):
                for kw in range(Kw):
                    s = 0.0
                    for oh in range(H):
                        for ow in range(W_dim):
                            ih = oh + kh - pad
                            iw = ow + kw - pad
                            if 0 <= ih < H and 0 <= iw < W_dim:
                                s += float(xc[ih, iw])
                    expected_dw_c[kh, kw] = s
            np.testing.assert_allclose(dW[c, 0], expected_dw_c, rtol=1e-5)

    def test_depthwise_zero_dy(self):
        """Depthwise 3x3 pad=1: zero dy → zero dX, dW, db for all channels."""
        rng = np.random.RandomState(803)
        N, C, H, W_dim = 1, 3, 3, 3
        groups = C
        Kh = Kw = 3
        pad = 1
        net = _make_conv_net(N, C, H, W_dim, C, Kh=Kh, pad=pad, groups=groups, bias=True)
        x = rng.randn(N, C, H, W_dim).astype(np.float32) * 0.3
        W = rng.randn(C, 1, Kh, Kw).astype(np.float32) * 0.2
        b = rng.randn(C).astype(np.float32) * 0.1
        dy = np.zeros((N, C, H, W_dim), dtype=np.float32)

        dX, dW, db = _run_conv_backward(net, x, dy, W, b, pad=pad, groups=groups)
        np.testing.assert_array_equal(dX, np.zeros_like(dX))
        np.testing.assert_array_equal(dW, np.zeros_like(dW))
        np.testing.assert_array_equal(db, np.zeros_like(db))


# ---------------------------------------------------------------------------
# Test Class 6: Common invariants
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
