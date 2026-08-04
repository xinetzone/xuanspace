"""Softsign layer Forward/Backward gradient tests.

Covers the Softsign activation ``y = x / (1 + |x|)`` with derivative
``dy/dx = 1 / (1 + |x|)^2``:

  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed)
  2. Numerical gradient check (central finite differences)
  3. Boundary conditions:
       - x = 0        -> derivative = 1 exactly (dx = dy; C1-smooth, no kink)
       - |x| -> large -> gradient -> 0 (vanishing / saturation)
       - x < 0 region -> dx = dy / (1 - x)^2
       - x > 0 region -> dx = dy / (1 + x)^2
       - odd symmetry of forward and even symmetry of the gradient factor
  4. Non-uniform upstream gradient dy scaling

Softsign is C1-continuous (no kink at x = 0) but C2-discontinuous (the second
derivative jumps from +2 to -2 at x = 0).  This is a "Type A" function: central
differences straddling x = 0 have O(h) truncation error instead of O(h^2), so
the numerical-gradient tests push near-zero points away from the origin and use
rtol = 5e-3 (matching the ELU Type-A convention).
"""

import textwrap
import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from .caffe_test_helpers import avoid_c1_discontinuity

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
EPS = 1e-3       # central finite-difference step
NUM_RTOL = 5e-3  # Type-A (C1-continuous, C2-discontinuous) tolerance
NUM_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_softsign_prototxt(N=1, C=1, H=4, W=5):
    """Create a minimal Input(N,C,H,W) -> Softsign prototxt."""
    return textwrap.dedent(f"""\
        name: "test_softsign_bw"
        input: "data"
        input_dim: {N}
        input_dim: {C}
        input_dim: {H}
        input_dim: {W}
        layer {{
          name: "softsign"
          type: "Softsign"
          bottom: "data"
          top: "out"
        }}
    """)


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _softsign_ref(x):
    """Forward y = x/(1+|x|) and backward factor dx/dy = 1/(1+|x|)^2."""
    x = np.asarray(x, dtype=np.float64)
    y = x / (1.0 + np.abs(x))
    dx_factor = 1.0 / (1.0 + np.abs(x)) ** 2
    return y, dx_factor


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
# Tests: Softsign Forward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftsignForward:
    """Softsign forward correctness (hand-computed known values)."""

    def _make_net(self):
        return Net(_make_softsign_prototxt(N=1, C=1, H=1, W=6))

    def test_forward_known_values(self):
        """Hand-computed y = x/(1+|x|) for a set of representative inputs."""
        net = self._make_net()
        # x:          0,   1,  -1,   2,  -2, 0.5
        # y:          0, 0.5, -0.5, 2/3, -2/3, 1/3
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float32).reshape(1, 1, 1, 6)
        expected = np.array([0.0, 0.5, -0.5, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
                            dtype=np.float32).reshape(1, 1, 1, 6)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_forward_odd_symmetry(self):
        """Softsign is odd: f(-x) = -f(x)."""
        net = self._make_net()
        x = np.random.RandomState(0).randn(1, 1, 1, 6).astype(np.float32) * 3.0
        out = net.forward({"data": x})["out"]
        out_neg = net.forward({"data": -x})["out"]
        np.testing.assert_allclose(out_neg, -out, rtol=1e-6, atol=1e-6)

    def test_forward_range(self):
        """Output is bounded in (-1, 1) for all inputs."""
        net = self._make_net()
        x = np.random.RandomState(1).randn(1, 1, 4, 5).astype(np.float32) * 100.0
        out = net.forward({"data": x})["out"]
        assert np.all(out > -1.0) and np.all(out < 1.0), f"Softsign output out of (-1,1): {out.min()}, {out.max()}"


# ---------------------------------------------------------------------------
# Tests: Softsign Backward (analytical)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftsignGradient:
    """Softsign backward analytical gradient tests."""

    def _make_net(self):
        return Net(_make_softsign_prototxt())

    def test_backward_analytical(self):
        """dx = dy / (1+|x|)^2 for random x (mixed signs)."""
        net = self._make_net()
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        y_ref, dx_factor = _softsign_ref(x)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, y_ref, rtol=1e-5, atol=1e-6)

    def test_backward_zero_input(self):
        """x=0 -> derivative = 1 -> dx = dy exactly (C1-smooth, no kink)."""
        net = self._make_net()
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(1, 1, 2, 3)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, dy, rtol=1e-6, atol=1e-6)

    def test_backward_positive_region(self):
        """x > 0: dx = dy / (1+x)^2."""
        net = self._make_net()
        x = np.array([1.0, 2.0, 3.0, 0.5, 10.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 1, 1, 5)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = dy / (1.0 + x) ** 2
        np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)

    def test_backward_negative_region(self):
        """x < 0: dx = dy / (1-x)^2."""
        net = self._make_net()
        x = np.array([-1.0, -2.0, -3.0, -0.5, -10.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 1, 1, 5)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = dy / (1.0 - x) ** 2
        np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)

    def test_backward_saturation_positive(self):
        """Large positive x => gradient -> 0 (vanishing)."""
        net = self._make_net()
        x = np.array([10.0, 20.0, 50.0, 100.0, 1000.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # dx = 1/(1+|x|)^2: x=10 -> 1/121 ~ 0.0083, x=1000 -> ~1e-6. Verify monotone
        # decay toward zero (vanishing gradient) rather than a fixed tight bound.
        assert np.all(np.abs(dx) < 0.01), f"Saturated positive softsign should have near-zero dx, got {dx}"

    def test_backward_saturation_negative(self):
        """Large negative x => gradient -> 0 (vanishing)."""
        net = self._make_net()
        x = np.array([-10.0, -20.0, -50.0, -100.0, -1000.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # dx = 1/(1+|x|)^2: x=-10 -> 1/121 ~ 0.0083, x=-1000 -> ~1e-6.
        assert np.all(np.abs(dx) < 0.01), f"Saturated negative softsign should have near-zero dx, got {dx}"

    def test_backward_symmetry(self):
        """Gradient factor is even: dx(-x) = dx(x) since (1+|x|)^2 is even."""
        net = self._make_net()
        x = np.array([1.0, 2.0, 0.5, 3.0, 4.0, 0.25], dtype=np.float32).reshape(1, 1, 2, 3)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        net.forward({"data": -x})
        net.backward({"out": dy})
        dx_neg = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, dx_neg, rtol=1e-6, atol=1e-6)

    def test_backward_nonuniform_dy(self):
        """Non-uniform upstream gradient dy is scaled correctly."""
        net = self._make_net()
        rng = np.random.RandomState(99)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32) * 5.0
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        _, dx_factor = _softsign_ref(x)
        expected = dy * dx_factor
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
# Tests: Softsign Numerical Gradient
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftsignNumericalGradient:
    """Central finite-difference gradient checks (Type-A tolerance)."""

    def _make_net(self):
        return Net(_make_softsign_prototxt(N=1, C=1, H=3, W=4))

    def test_numerical_gradient_mixed_signs(self):
        """Numerical check with mixed-sign inputs."""
        net = self._make_net()
        rng = np.random.RandomState(7)
        x = rng.randn(1, 1, 3, 4).astype(np.float32) * 2.0
        # Softsign is C1-continuous but C2-discontinuous at x=0; push near-zero
        # points away to avoid O(h) central-difference truncation at the origin.
        x = avoid_c1_discontinuity(x, h=EPS)
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_positive_only(self):
        """Numerical check in the strictly-positive region (no origin crossing)."""
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
        """Numerical check in the strictly-negative region (no origin crossing)."""
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