"""Softplus layer Forward/Backward gradient tests.

Covers the Softplus activation ``y = log(1 + exp(x))`` with derivative
``dy/dx = 1 / (1 + exp(-x))`` (the logistic sigmoid):

  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed)
  2. Numerical gradient check (central finite differences)
  3. Boundary conditions:
       - x -> -inf  -> y -> 0,  gradient -> 0 (saturating bottom)
       - x -> +inf  -> y -> x, gradient -> 1 (identity)
       - x = 0      -> y = log(2), gradient = 0.5 (no kink)
       - x > 0 region -> the numerically stable branch x + log1p(exp(-x))
       - x < 0 region -> the naive log1p(exp(x)) branch
  4. Non-uniform upstream gradient dy scaling

Softplus is C∞ (infinitely smooth, no kink anywhere), so it is a "Type 0"
function: central differences have full O(h^2) truncation error everywhere
(note |x| is never used), and the standard tight tolerance is applicable.
"""

import textwrap
import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
EPS = 1e-3       # central finite-difference step
NUM_RTOL = 1e-3  # Type-0 (C∞ smooth) tolerance
NUM_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_softplus_prototxt(N=1, C=1, H=4, W=5):
    """Create a minimal Input(N,C,H,W) -> Softplus prototxt."""
    return textwrap.dedent(f"""\
        name: "test_softplus_bw"
        input: "data"
        input_dim: {N}
        input_dim: {C}
        input_dim: {H}
        input_dim: {W}
        layer {{
          name: "softplus"
          type: "Softplus"
          bottom: "data"
          top: "out"
        }}
    """)


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _softplus_ref(x):
    """Forward y = log(1+exp(x)) and backward factor dy/dx = sigmoid(x)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.log1p(np.exp(x))
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return y, sigmoid


def _num_grad(net, x, dy, h=EPS):
    """Compute numerical gradient via central differences: dL/dx_i.

    L = sum(dy * out) so dy is the upstream gradient.
    """
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
# Tests: Softplus Forward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftplusForward:
    """Softplus forward correctness (hand-computed known values)."""

    def _make_net(self):
        return Net(_make_softplus_prototxt(N=1, C=1, H=1, W=6))

    def test_forward_known_values(self):
        """Hand-computed y = log(1+exp(x)) for a set of representative inputs."""
        net = self._make_net()
        # x:                    0,      1,     -1,      2,     -2,    0.5
        # y:                   ln2, ln(1+e), ln(1+1/e), ln(1+e^2), ln(1+1/e^2), ln(1+sqrt(e))
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float32).reshape(1, 1, 1, 6)
        expected = np.array(
            [np.log(2.0), np.log(1.0 + np.e), np.log(1.0 + np.exp(-1.0)),
             np.log(1.0 + np.exp(2.0)), np.log(1.0 + np.exp(-2.0)),
             np.log(1.0 + np.exp(0.5))],
            dtype=np.float32).reshape(1, 1, 1, 6)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_forward_range(self):
        """Output is positive and monotone increasing for all inputs."""
        net = self._make_net()
        x = np.random.RandomState(1).randn(1, 1, 4, 5).astype(np.float32) * 10.0
        out = net.forward({"data": x})["out"]
        assert np.all(out > 0.0), f"Softplus output must be positive, got min={out.min()}"
        # monotone: sorting x ascending must sort out ascending
        flat_x = x.ravel()
        flat_out = out.ravel()
        order = np.argsort(flat_x)
        assert np.all(np.diff(flat_out[order]) >= 0.0), "Softplus output must be monotone non-decreasing"

    def test_forward_large_positive_no_overflow(self):
        """Large positive x must not overflow: y ~ x (numerically stable branch)."""
        net = self._make_net()
        x = np.array([100.0, 500.0, 1000.0, 1e4], dtype=np.float32).reshape(1, 1, 1, 4)
        out = net.forward({"data": x})["out"]
        assert np.all(np.isfinite(out)), f"Softplus overflowed for large positive x: {out}"
        # y ~ x for large x
        np.testing.assert_allclose(out, x, rtol=1e-4, atol=1e-4)

    def test_forward_large_negative_no_underflow(self):
        """Large negative x must not overflow: y -> 0 (well-behaved)."""
        net = self._make_net()
        x = np.array([-100.0, -500.0, -1000.0, -1e4], dtype=np.float32).reshape(1, 1, 1, 4)
        out = net.forward({"data": x})["out"]
        assert np.all(np.isfinite(out)), f"Softplus overflowed for large negative x: {out}"
        assert np.all(out >= 0.0), f"Softplus output must be >= 0, got {out}"
        # y ~ exp(x) -> ~0 for large negative x
        assert np.all(out < 1e-3), f"Softplus output should be ~0 for large negative x, got {out}"


# ---------------------------------------------------------------------------
# Tests: Softplus Backward (analytical)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftplusGradient:
    """Softplus backward analytical gradient tests."""

    def _make_net(self):
        return Net(_make_softplus_prototxt())

    def test_backward_analytical(self):
        """dx = dy * sigmoid(x) for random x (mixed signs)."""
        net = self._make_net()
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        y_ref, sigmoid = _softplus_ref(x)
        expected = dy * sigmoid
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, y_ref, rtol=1e-5, atol=1e-6)

    def test_backward_zero_input(self):
        """x=0 -> sigmoid(0)=0.5 -> dx = 0.5*dy (C∞, no kink)."""
        net = self._make_net()
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(1, 1, 2, 3)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, 0.5 * dy, rtol=1e-6, atol=1e-6)

    def test_backward_positive_region(self):
        """x > 0: dx = dy * 1/(1+exp(-x))."""
        net = self._make_net()
        x = np.array([1.0, 2.0, 3.0, 0.5, 10.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 1, 1, 5)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = dy / (1.0 + np.exp(-x))
        np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)

    def test_backward_negative_region(self):
        """x < 0: dx = dy * exp(x)/(1+exp(x))."""
        net = self._make_net()
        x = np.array([-1.0, -2.0, -3.0, -0.5, -10.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 1, 1, 5)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = dy * (np.exp(x) / (1.0 + np.exp(x)))
        np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)

    def test_backward_saturation_positive(self):
        """x -> +inf => gradient -> 1 (identity regime)."""
        net = self._make_net()
        x = np.array([10.0, 20.0, 50.0, 100.0, 1000.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # sigmoid(10) ~ 0.99995, sigmoid(1000) ~ 1. Gradient should be ~1.
        assert np.all(np.abs(dx) > 0.999), f"Softplus positive saturation should be ~1, got {dx}"

    def test_backward_saturation_negative(self):
        """x -> -inf => gradient -> 0 (vanishing)."""
        net = self._make_net()
        x = np.array([-10.0, -20.0, -50.0, -100.0, -1000.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # sigmoid(-10) ~ 4.5e-5, sigmoid(-1000) ~ 0. Gradient should be ~0.
        assert np.all(np.abs(dx) < 0.001), f"Softplus negative saturation should be ~0, got {dx}"

    def test_backward_nonuniform_dy(self):
        """Non-uniform upstream gradient dy is scaled correctly."""
        net = self._make_net()
        rng = np.random.RandomState(99)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32) * 5.0
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        _, sigmoid = _softplus_ref(x)
        expected = dy * sigmoid
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    def test_backward_runs_without_crash(self):
        """Backward should complete without errors on random data."""
        net = self._make_net()
        x = np.random.RandomState(1).randn(1, 1, 4, 5).astype(np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == x.shape
        assert dx.dtype == np.float32
        assert np.all(np.isfinite(dx))


# ---------------------------------------------------------------------------
# Tests: Softplus Numerical Gradient
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftplusNumericalGradient:
    """Central finite-difference gradient checks (Type-0, C∞ tolerance)."""

    def _make_net(self):
        return Net(_make_softplus_prototxt(N=1, C=1, H=3, W=4))

    def test_numerical_gradient_mixed_signs(self):
        """Numerical check with mixed-sign inputs (no kink, tight tolerance)."""
        net = self._make_net()
        rng = np.random.RandomState(7)
        x = rng.randn(1, 1, 3, 4).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_positive_only(self):
        """Numerical check in the strictly-positive region (branch 1)."""
        net = self._make_net()
        rng = np.random.RandomState(13)
        x = (rng.rand(1, 1, 3, 4).astype(np.float32) * 2.0) + 0.5  # in (0.5, 2.5)
        dy = np.ones_like(x, dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_negative_only(self):
        """Numerical check in the strictly-negative region (branch 2)."""
        net = self._make_net()
        rng = np.random.RandomState(21)
        x = -(rng.rand(1, 1, 3, 4).astype(np.float32) * 2.0) - 0.5  # in (-2.5, -0.5)
        dy = np.ones_like(x, dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])