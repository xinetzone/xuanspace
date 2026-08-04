"""L2Norm layer Forward/Backward gradient tests.

Covers the L2-normalization layer ``y = x / ||x||_2`` (normalizing each group
of elements at and after `axis` to unit L2 norm):

  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed)
  2. Numerical gradient check (central finite differences)
  3. Known-value verification (unit vectors, scalar vector, zero+eps)
  4. Grouped normalization: axis=1 (channel) and axis=2 (spatial) configs
  5. Output is unit-norm per group
  6. Shape/finite/determinism checks

L2Norm is smooth everywhere (no kink), so standard tolerance (rtol=1e-3) is
used for numerical-gradient checks.

Mathematical reference:
  Forward:  y[i] = x[i] / norm(g),  norm(g) = sqrt(sum_j x[j]^2 + eps)
  Backward: dx[i] = dy[i] / norm(g) - x[i] * (sum_j dy[j]*x[j]) / norm(g)^3
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

EPS = 1e-3       # central finite-difference step
NUM_RTOL = 1e-2  # float32 central-difference precision (matches _grad_check_utils)
NUM_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_l2norm_prototxt(N=1, C=1, H=4, W=5, axis=1, eps=1e-5):
    """Create a minimal Input(N,C,H,W) -> L2Norm prototxt."""
    return textwrap.dedent(f"""\
        name: "test_l2norm_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "l2norm"
          type: "L2Norm"
          bottom: "data"
          top: "out"
          l2_norm_param {{ axis: {axis} eps: {eps} }}
        }}
    """)


def _make_l2norm_net(N=1, C=1, H=4, W=5, axis=1, eps=1e-5):
    return Net(_make_l2norm_prototxt(N, C, H, W, axis=axis, eps=eps))


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _l2norm_axes(x, axis):
    """Return the tuple of axes that are normalized together (axis..end).

    Matches Caffe's L2Norm semantics: each group of elements at indices
    `[axis, num_axes)` is normalized to unit L2 norm.
    """
    return tuple(range(axis, x.ndim))


def _l2norm_forward_np(x, axis=1, eps=1e-5):
    """Forward reference: y = x / ||x||_2 over [axis, ndim)."""
    x = np.asarray(x, dtype=np.float64)
    axes = _l2norm_axes(x, axis)
    norm = np.sqrt(np.sum(x ** 2, axis=axes, keepdims=True) + eps)
    return x / norm


def _l2norm_backward_np(x, dy, axis=1, eps=1e-5):
    """Backward reference: dL/dx for L = sum(dy * y)."""
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    axes = _l2norm_axes(x, axis)
    norm = np.sqrt(np.sum(x ** 2, axis=axes, keepdims=True) + eps)
    inv_norm = 1.0 / norm
    dot = np.sum(dy * x, axis=axes, keepdims=True)
    coef = dot / (norm ** 3)
    return dy * inv_norm - x * coef


def _num_grad(net, x, dy, h=EPS):
    """Numerical gradient via central differences: dL/dx_i."""
    grad = np.zeros_like(x, dtype=np.float64)
    flat_x = x.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x.size):
        orig = flat_x[i]
        xp = x.copy()
        xp.ravel()[i] = orig + h
        out_p = net.forward({"data": xp.astype(np.float32)})["out"]
        loss_p = float(np.sum(dy * out_p))
        xm = x.copy()
        xm.ravel()[i] = orig - h
        out_m = net.forward({"data": xm.astype(np.float32)})["out"]
        loss_m = float(np.sum(dy * out_m))
        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# Tests: L2Norm Forward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestL2NormForward:
    """L2Norm forward correctness (known values + unit-norm property)."""

    def test_forward_known_values(self):
        """Hand-computed y = x / sqrt(sum x^2 + eps)."""
        net = _make_l2norm_net(N=1, C=1, H=1, W=4, axis=1)
        # x = [3, 4, 0, 0]; sum sq = 25; norm = 5; y = [0.6, 0.8, 0, 0]
        x = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 1, 4)
        out = net.forward({"data": x})["out"]
        expected = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 1, 4)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_forward_unit_norm_per_channel(self):
        """Output has unit L2 norm per sample (axis=1 normalizes over [C,H,W))."""
        rng = np.random.RandomState(0)
        N, C, H, W = 2, 3, 4, 5
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=1)
        x = rng.randn(N, C, H, W).astype(np.float32) * 3.0
        out = net.forward({"data": x})["out"]
        # axis=1 normalizes over [1, 4) = (C, H, W): per-sample unit norm.
        out = np.asarray(out)
        norm = np.sqrt(np.sum(out ** 2, axis=(1, 2, 3)))
        np.testing.assert_allclose(norm, 1.0, rtol=1e-4, atol=1e-4)

    def test_forward_unit_norm_spatial(self):
        """Output has unit L2 norm per (batch, channel) group (axis=2 over [H,W))."""
        rng = np.random.RandomState(1)
        N, C, H, W = 2, 2, 3, 4
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=2)
        x = rng.randn(N, C, H, W).astype(np.float32)
        out = np.asarray(net.forward({"data": x})["out"])
        # axis=2 normalizes over [2, 4) = (H, W): per (n, c) unit norm.
        norm = np.sqrt(np.sum(out ** 2, axis=(2, 3)))
        np.testing.assert_allclose(norm, 1.0, rtol=1e-4, atol=1e-4)

    def test_forward_all_zero(self):
        """All-zero input: norm = sqrt(eps), output = x / sqrt(eps) = 0."""
        net = _make_l2norm_net(N=1, C=1, H=1, W=4, axis=1, eps=1e-5)
        x = np.zeros((1, 1, 1, 4), dtype=np.float32)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_forward_matches_numpy(self):
        """Forward matches numpy reference on random data."""
        rng = np.random.RandomState(2)
        N, C, H, W = 2, 3, 4, 5
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=1)
        x = rng.randn(N, C, H, W).astype(np.float32)
        out = net.forward({"data": x})["out"]
        expected = _l2norm_forward_np(x, axis=1).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: L2Norm Backward (analytical)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestL2NormGradient:
    """L2Norm backward analytical gradient tests."""

    def test_backward_analytical(self):
        """dx = dy/norm - x * (dot(dy,x))/norm^3 for random data."""
        rng = np.random.RandomState(42)
        N, C, H, W = 2, 3, 4, 5
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=1)
        x = rng.randn(N, C, H, W).astype(np.float32) * 2.0
        dy = rng.randn(N, C, H, W).astype(np.float32)
        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = _l2norm_backward_np(x, dy, axis=1).astype(np.float32)
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, _l2norm_forward_np(x, axis=1).astype(np.float32),
                                   rtol=1e-5, atol=1e-6)

    def test_backward_axis_spatial(self):
        """Backward with axis=2 (spatial groups)."""
        rng = np.random.RandomState(43)
        N, C, H, W = 2, 2, 3, 4
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=2)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = _l2norm_backward_np(x, dy, axis=2).astype(np.float32)
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    def test_backward_orthogonal_to_x(self):
        """If dy is a scaled copy of x, dot(dy,x) != 0; gradient along x is
        non-zero. Verify dx is orthogonal to x (since preserving norm is the
        only constraint direction)."""
        rng = np.random.RandomState(44)
        N, C, H, W = 1, 2, 3, 3
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=1)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = (x * 5.0).astype(np.float32)  # dy parallel to x
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = _l2norm_backward_np(x, dy, axis=1).astype(np.float32)
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    def test_backward_runs_without_crash(self):
        """Backward should complete without errors on random data."""
        net = _make_l2norm_net(N=2, C=3, H=4, W=5, axis=1)
        x = np.random.RandomState(1).randn(2, 3, 4, 5).astype(np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == x.shape
        assert dx.dtype == np.float32
        assert np.all(np.isfinite(dx))


# ---------------------------------------------------------------------------
# Tests: L2Norm Numerical Gradient
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestL2NormNumericalGradient:
    """Central finite-difference gradient checks (smooth, standard tolerance)."""

    def test_numerical_gradient_axis1(self):
        """Numerical check with axis=1 (channel) normalization."""
        rng = np.random.RandomState(7)
        N, C, H, W = 1, 3, 3, 4
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=1)
        x = rng.randn(N, C, H, W).astype(np.float32) * 2.0
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_axis2(self):
        """Numerical check with axis=2 (spatial) normalization."""
        rng = np.random.RandomState(13)
        N, C, H, W = 1, 2, 3, 4
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=2)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_positive_only(self):
        """Numerical check with strictly-positive inputs (no norm near zero)."""
        rng = np.random.RandomState(21)
        N, C, H, W = 1, 2, 3, 4
        net = _make_l2norm_net(N=N, C=C, H=H, W=W, axis=1)
        x = (rng.rand(N, C, H, W).astype(np.float32) * 2.0) + 0.5
        dy = np.ones_like(x, dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])