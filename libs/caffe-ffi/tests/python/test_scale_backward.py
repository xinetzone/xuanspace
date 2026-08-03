"""Scale layer Backward gradient tests.

Scale computes y = x * alpha + beta (broadcast along specified axis).
Gradients:
  dX[n,d,i]  = dy[n,d,i] * alpha[d]
  d_alpha[d] = sum over n,i of dy[n,d,i] * x[n,d,i]
  d_beta[d]  = sum over n,i of dy[n,d,i]

Covers:
  1. Known-value hand verification (scale only, scale+bias)
  2. Analytical gradient (numpy reference vs caffe-ffi)
  3. Numerical gradient check (central finite differences for dX)
  4. Numerical gradient for d_scale and d_bias
  5. Multiple scale factors and bias values
  6. Multiple shapes (2D NxD, 4D NxCxHxW)
  7. Zero dy -> zero gradients
  8. Shape/finite/determinism checks
  9. Forward output preserved after backward
  10. bias_term=false: only d_scale and dX
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

EPS = 1e-3
RTOL = 1e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Numpy reference for Scale forward/backward
# ---------------------------------------------------------------------------

def scale_forward_np(x, alpha, beta=None, axis=1, num_axes=1):
    """Numpy reference: y = x * alpha + beta with broadcasting along axis."""
    x64 = x.astype(np.float64)
    ndim = x64.ndim
    out_shape = [1] * ndim
    for i in range(num_axes):
        out_shape[axis + i] = x64.shape[axis + i]
    a64 = alpha.reshape(out_shape).astype(np.float64)
    y = x64 * a64
    if beta is not None:
        b64 = beta.reshape(out_shape).astype(np.float64)
        y = y + b64
    return y.astype(np.float32)


def scale_backward_np(x, dy, alpha, beta=None, axis=1, num_axes=1,
                      need_dx=True, need_dalpha=True, need_dbeta=True):
    """Numpy reference for Scale backward."""
    x64 = x.astype(np.float64)
    dy64 = dy.astype(np.float64)
    ndim = x64.ndim
    out_shape = [1] * ndim
    for i in range(num_axes):
        out_shape[axis + i] = x64.shape[axis + i]
    a64 = alpha.reshape(out_shape).astype(np.float64)

    dX = None
    if need_dx:
        dX = (dy64 * a64).astype(np.float32)

    # Determine reduction axes (all except the scale axes)
    scale_axes = tuple(range(axis, axis + num_axes))
    reduce_axes = tuple(i for i in range(ndim) if i not in scale_axes)

    dalpha = None
    if need_dalpha:
        dalpha = (dy64 * x64).sum(axis=reduce_axes).astype(np.float32)

    dbeta = None
    if need_dbeta and beta is not None:
        dbeta = dy64.sum(axis=reduce_axes).astype(np.float32)

    return dX, dalpha, dbeta


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_scale_prototxt(input_dims, axis=1, num_axes=1, bias_term=False,
                         filler_value=1.0, bias_filler=0.0):
    """Create Input -> Scale prototxt."""
    dims_lines = "\n".join(f"input_dim: {d}" for d in input_dims)
    bias_str = "true" if bias_term else "false"
    return textwrap.dedent(f"""\
        name: "test_scale_bw"
        input: "data"
        {dims_lines}
        layer {{
          name: "scale"
          type: "Scale"
          bottom: "data"
          top: "scale_out"
          scale_param {{
            axis: {axis}
            num_axes: {num_axes}
            bias_term: {bias_str}
            filler {{ type: "constant" value: {filler_value} }}
            bias_filler {{ type: "constant" value: {bias_filler} }}
          }}
        }}
    """)


def _make_scale_net(input_dims, axis=1, num_axes=1, bias_term=False,
                    filler_value=1.0, bias_filler=0.0):
    proto = _make_scale_prototxt(input_dims, axis, num_axes, bias_term,
                                  filler_value, bias_filler)
    return Net(proto)


# ---------------------------------------------------------------------------
# Helper: set scale blobs and run backward
# ---------------------------------------------------------------------------

def _set_scale_blobs(net, alpha, beta=None):
    """Set scale layer blobs: blobs[0]=alpha, blobs[1]=beta (if bias)."""
    scale_layer = net.layer_by_name("scale")
    scale_layer.blobs[0].from_numpy(alpha.astype(np.float32).ravel())
    if beta is not None and len(scale_layer.blobs) > 1:
        scale_layer.blobs[1].from_numpy(beta.astype(np.float32).ravel())


def _run_scale_backward(net, x, dy, alpha, beta=None):
    """Run forward then backward, return (dX, d_alpha, d_beta)."""
    _set_scale_blobs(net, alpha, beta)
    net.forward({"data": x.astype(np.float32)})
    net.backward({"scale_out": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    scale_layer = net.layer_by_name("scale")
    d_alpha = scale_layer.blobs[0].diff
    d_beta = scale_layer.blobs[1].diff if (beta is not None and len(scale_layer.blobs) > 1) else None
    return dX, d_alpha, d_beta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestScaleBackwardKnownValues:
    """L1: Hand-computed known-value verification."""

    def test_forward_identity_scale(self):
        """Forward with alpha=1, no bias: y == x."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, num_axes=1, bias_term=False, filler_value=1.0)
        rng = np.random.RandomState(42)
        x = rng.randn(N, D).astype(np.float32)
        out = net.forward({"data": x})["scale_out"]
        np.testing.assert_array_equal(out, x)

    def test_forward_scale_only(self):
        """Forward with known alpha: y = x * alpha (channel-wise)."""
        N, C, H, W = 1, 2, 1, 1
        net = _make_scale_net((N, C, H, W), axis=1, bias_term=False)
        alpha = np.array([2.0, 0.5], dtype=np.float32)
        _set_scale_blobs(net, alpha)
        x = np.array([[[[1.0]], [[2.0]]]], dtype=np.float32)
        out = net.forward({"data": x})["scale_out"]
        expected = np.array([[[[2.0]], [[1.0]]]], dtype=np.float32)  # 1*2=2, 2*0.5=1
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_forward_scale_plus_bias(self):
        """Forward with alpha=2, beta=3: y = 2*x + 3."""
        N, C, H, W = 1, 1, 1, 1
        net = _make_scale_net((N, C, H, W), axis=1, bias_term=True, filler_value=2.0, bias_filler=3.0)
        # Reset blobs to known values
        _set_scale_blobs(net, np.array([2.0]), np.array([3.0]))
        x = np.array([[[[4.0]]]], dtype=np.float32)
        out = net.forward({"data": x})["scale_out"]
        expected = np.array([[[[11.0]]]], dtype=np.float32)  # 2*4+3=11
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_backward_dx_known_values(self):
        """Backward dX known: dX = dy * alpha, alpha=[2,0.5]."""
        N, C, H, W = 1, 2, 1, 1
        net = _make_scale_net((N, C, H, W), axis=1, bias_term=False)
        alpha = np.array([2.0, 0.5], dtype=np.float32)
        x = np.array([[[[1.0]], [[2.0]]]], dtype=np.float32)
        dy = np.array([[[[1.0]], [[1.0]]]], dtype=np.float32)
        dX, _, _ = _run_scale_backward(net, x, dy, alpha)
        expected_dx = np.array([[[[2.0]], [[0.5]]]], dtype=np.float32)  # 1*2=2, 1*0.5=0.5
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-6)

    def test_backward_dscale_known_values(self):
        """Backward d_scale known: d_alpha = sum(dy*x) over broadcast dims."""
        N, D = 2, 3
        net = _make_scale_net((N, D), axis=1, bias_term=False)
        alpha = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        x = np.array([[1.0, 2.0, 4.0],
                       [3.0, 0.0, -2.0]], dtype=np.float32)
        dy = np.ones_like(x)
        _, d_alpha, _ = _run_scale_backward(net, x, dy, alpha)
        # d_alpha[d] = sum over n of dy[n,d]*x[n,d] = sum x[:,d]
        expected = x.sum(axis=0)
        np.testing.assert_allclose(d_alpha, expected, rtol=1e-5)

    def test_backward_dbias_known_values(self):
        """Backward d_bias known: d_beta = sum(dy) over broadcast dims."""
        N, D = 2, 3
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.0, bias_filler=0.0)
        alpha = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        beta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        x = np.zeros((N, D), dtype=np.float32)
        dy = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], dtype=np.float32)
        _, _, d_beta = _run_scale_backward(net, x, dy, alpha, beta)
        expected = dy.sum(axis=0)
        np.testing.assert_allclose(d_beta, expected, rtol=1e-5)


@require_cpp_extension
class TestScaleBackwardAnalytical:
    """L2: Analytical gradient comparison with numpy reference."""

    @pytest.mark.parametrize("N,D", [(2, 4), (4, 8), (1, 16)])
    def test_dx_vs_numpy(self, N, D):
        """dX = dy * alpha matches numpy reference."""
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.5, bias_filler=0.3)
        rng = np.random.RandomState(N * 100 + D)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = np.array([1.5] * D, dtype=np.float32)
        beta = np.array([0.3] * D, dtype=np.float32)
        dX, _, _ = _run_scale_backward(net, x, dy, alpha, beta)
        dX_ref, _, _ = scale_backward_np(x, dy, alpha, beta, axis=1)
        np.testing.assert_allclose(dX, dX_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("N,D", [(2, 4), (4, 8)])
    def test_dscale_vs_numpy(self, N, D):
        """d_alpha matches numpy reference."""
        net = _make_scale_net((N, D), axis=1, bias_term=False)
        rng = np.random.RandomState(N * 100 + D + 1)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = rng.uniform(0.5, 2.0, D).astype(np.float32)
        _, d_alpha, _ = _run_scale_backward(net, x, dy, alpha)
        _, d_alpha_ref, _ = scale_backward_np(x, dy, alpha, axis=1, need_dx=False, need_dbeta=False)
        np.testing.assert_allclose(d_alpha, d_alpha_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("N,D", [(2, 4), (4, 8)])
    def test_dbias_vs_numpy(self, N, D):
        """d_beta matches numpy reference."""
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.0, bias_filler=0.0)
        rng = np.random.RandomState(N * 100 + D + 2)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = rng.uniform(0.5, 2.0, D).astype(np.float32)
        beta = rng.randn(D).astype(np.float32)
        _, _, d_beta = _run_scale_backward(net, x, dy, alpha, beta)
        _, _, d_beta_ref = scale_backward_np(x, dy, alpha, beta, axis=1, need_dx=False, need_dalpha=False)
        np.testing.assert_allclose(d_beta, d_beta_ref, rtol=RTOL, atol=ATOL)

    def test_4d_spatial_shape(self):
        """4D (N,C,H,W) shape with alpha per channel."""
        N, C, H, W = 2, 3, 4, 4
        net = _make_scale_net((N, C, H, W), axis=1, bias_term=True, filler_value=1.0, bias_filler=0.0)
        rng = np.random.RandomState(123)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        alpha = rng.uniform(0.5, 2.0, C).astype(np.float32)
        beta = rng.randn(C).astype(np.float32)
        dX, d_alpha, d_beta = _run_scale_backward(net, x, dy, alpha, beta)
        dX_ref, d_alpha_ref, d_beta_ref = scale_backward_np(x, dy, alpha, beta, axis=1)
        np.testing.assert_allclose(dX, dX_ref, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(d_alpha, d_alpha_ref, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(d_beta, d_beta_ref, rtol=RTOL, atol=ATOL)

    def test_alpha_equals_zero(self):
        """alpha=0: dX=0 (dy*0), d_alpha=sum(dy*x) [independent of alpha value], d_beta=sum(dy)."""
        N, D = 2, 3
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=0.0, bias_filler=0.0)
        rng = np.random.RandomState(99)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = np.zeros(D, dtype=np.float32)
        beta = np.zeros(D, dtype=np.float32)
        dX, d_alpha, d_beta = _run_scale_backward(net, x, dy, alpha, beta)
        # dX = dy * alpha = 0 when alpha=0
        np.testing.assert_allclose(dX, np.zeros_like(x), atol=1e-6)
        # d_alpha = sum(dy * x) over n (does NOT depend on alpha value!)
        expected_dalpha = (dy.astype(np.float64) * x.astype(np.float64)).sum(axis=0).astype(np.float32)
        np.testing.assert_allclose(d_alpha, expected_dalpha, rtol=1e-5)
        # d_beta = sum(dy) over n
        np.testing.assert_allclose(d_beta, dy.sum(axis=0), rtol=1e-5)


@require_cpp_extension
class TestScaleBackwardNumerical:
    """L3: Numerical gradient via central finite differences."""

    def test_numerical_grad_dx(self):
        """Numerical gradient for dX matches analytical."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.3, bias_filler=0.5)
        rng = np.random.RandomState(42)
        x = rng.randn(N, D).astype(np.float32) * 0.5
        alpha = np.array([1.3] * D, dtype=np.float32)
        beta = np.array([0.5] * D, dtype=np.float32)
        dy = rng.randn(N, D).astype(np.float32)

        _set_scale_blobs(net, alpha, beta)
        net.forward({"data": x})
        net.backward({"scale_out": dy})
        analytic_dx = net.blob_by_name("data").diff.copy()

        num_dx = numerical_grad_for_input(
            net, "data", x, "scale_out", dy, h=EPS, name="scale_dX",
        )
        assert_grad_close(analytic_dx, num_dx, name="scale_dX", rtol=RTOL, atol=ATOL*10)

    def test_numerical_grad_dscale(self):
        """Numerical gradient for d_scale matches analytical (perturb alpha blob)."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, bias_term=False)
        rng = np.random.RandomState(43)
        x = rng.randn(N, D).astype(np.float32) * 0.5
        alpha = rng.uniform(0.5, 2.0, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)

        _set_scale_blobs(net, alpha)
        net.forward({"data": x})
        net.backward({"scale_out": dy})
        analytic_ds = net.layer_by_name("scale").blobs[0].diff.copy()

        num_ds = numerical_grad_for_blob(
            net, "scale", 0, {"data": x}, "scale_out", dy, h=EPS, name="scale_dAlpha",
        )
        assert_grad_close(analytic_ds, num_ds, name="scale_dAlpha", rtol=RTOL, atol=ATOL*10)

    def test_numerical_grad_dbias(self):
        """Numerical gradient for d_bias matches analytical (perturb beta blob)."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.0, bias_filler=0.0)
        rng = np.random.RandomState(44)
        x = rng.randn(N, D).astype(np.float32) * 0.5
        alpha = rng.uniform(0.5, 2.0, D).astype(np.float32)
        beta = rng.randn(D).astype(np.float32) * 0.1
        dy = rng.randn(N, D).astype(np.float32)

        _set_scale_blobs(net, alpha, beta)
        net.forward({"data": x})
        net.backward({"scale_out": dy})
        analytic_db = net.layer_by_name("scale").blobs[1].diff.copy()

        num_db = numerical_grad_for_blob(
            net, "scale", 1, {"data": x}, "scale_out", dy, h=EPS, name="scale_dBeta",
        )
        assert_grad_close(analytic_db, num_db, name="scale_dBeta", rtol=RTOL, atol=ATOL*10)


@require_cpp_extension
class TestScaleBackwardProperties:
    """Property-based tests: zero gradients, shapes, determinism, forward preserved."""

    def test_zero_dy_gives_zero_gradients(self):
        """Zero dy produces zero dX, d_scale, d_bias."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.5, bias_filler=0.3)
        rng = np.random.RandomState(50)
        x = rng.randn(N, D).astype(np.float32)
        dy = np.zeros_like(x)
        alpha = np.array([1.5] * D, dtype=np.float32)
        beta = np.array([0.3] * D, dtype=np.float32)
        dX, d_alpha, d_beta = _run_scale_backward(net, x, dy, alpha, beta)
        np.testing.assert_allclose(dX, np.zeros_like(x), atol=1e-7)
        np.testing.assert_allclose(d_alpha, np.zeros(D), atol=1e-7)
        np.testing.assert_allclose(d_beta, np.zeros(D), atol=1e-7)

    def test_gradient_shapes(self):
        """Gradient shapes match blob/param shapes."""
        N, C, H, W = 2, 3, 4, 5
        net = _make_scale_net((N, C, H, W), axis=1, bias_term=True, filler_value=1.0, bias_filler=0.0)
        rng = np.random.RandomState(51)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        alpha = rng.uniform(0.5, 2.0, C).astype(np.float32)
        beta = rng.randn(C).astype(np.float32)
        dX, d_alpha, d_beta = _run_scale_backward(net, x, dy, alpha, beta)
        assert dX.shape == (N, C, H, W)
        assert d_alpha.shape == (C,)
        assert d_beta.shape == (C,)

    def test_determinism(self):
        """Running backward twice with same inputs gives same gradients."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.0, bias_filler=0.0)
        rng = np.random.RandomState(52)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = rng.uniform(0.5, 2.0, D).astype(np.float32)
        beta = rng.randn(D).astype(np.float32)
        dX1, ds1, db1 = _run_scale_backward(net, x, dy, alpha, beta)
        dX2, ds2, db2 = _run_scale_backward(net, x, dy, alpha, beta)
        np.testing.assert_array_equal(dX1, dX2)
        np.testing.assert_array_equal(ds1, ds2)
        np.testing.assert_array_equal(db1, db2)

    def test_forward_preserved_after_backward(self):
        """Forward output is unchanged after backward."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.5, bias_filler=0.3)
        rng = np.random.RandomState(53)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = np.array([1.5] * D, dtype=np.float32)
        beta = np.array([0.3] * D, dtype=np.float32)
        _set_scale_blobs(net, alpha, beta)
        out1 = net.forward({"data": x})["scale_out"].copy()
        net.backward({"scale_out": dy})
        out2 = net.blob_by_name("scale_out").data.copy()
        np.testing.assert_array_equal(out1, out2)

    def test_bias_term_false_no_bias_blob(self):
        """When bias_term=false, only 1 blob (scale), no bias diff."""
        N, D = 2, 4
        net = _make_scale_net((N, D), axis=1, bias_term=False)
        assert len(net.layer_by_name("scale").blobs) == 1

    def test_no_bias_dalpha_only(self):
        """Without bias: dX and d_scale correct, no bias blob."""
        N, D = 3, 5
        net = _make_scale_net((N, D), axis=1, bias_term=False)
        rng = np.random.RandomState(54)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = rng.uniform(0.5, 2.0, D).astype(np.float32)
        dX, d_alpha, d_beta = _run_scale_backward(net, x, dy, alpha)
        assert d_beta is None
        dX_ref, d_alpha_ref, _ = scale_backward_np(x, dy, alpha, axis=1, need_dbeta=False)
        np.testing.assert_allclose(dX, dX_ref, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(d_alpha, d_alpha_ref, rtol=RTOL, atol=ATOL)

    def test_finite_values(self):
        """All gradients are finite (no NaN/Inf)."""
        N, D = 4, 8
        net = _make_scale_net((N, D), axis=1, bias_term=True, filler_value=1.0, bias_filler=0.0)
        rng = np.random.RandomState(55)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        alpha = rng.uniform(0.1, 3.0, D).astype(np.float32)
        beta = rng.randn(D).astype(np.float32) * 0.5
        dX, d_alpha, d_beta = _run_scale_backward(net, x, dy, alpha, beta)
        assert np.all(np.isfinite(dX))
        assert np.all(np.isfinite(d_alpha))
        assert np.all(np.isfinite(d_beta))
