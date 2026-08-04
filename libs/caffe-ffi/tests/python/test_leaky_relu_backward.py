"""LeakyReLU layer Forward/Backward gradient tests.

Covers the LeakyReLU activation ``y = max(x,0) + negative_slope * min(x,0)``
with derivative ``dy/dx = (x > 0 ? 1 : negative_slope)``:

  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed)
  2. Numerical gradient check (central finite differences)
  3. Boundary conditions:
       - x > 0  -> y = x,  gradient = 1
       - x < 0  -> y = negative_slope * x, gradient = negative_slope
       - x = 0  -> y = 0, forward continuous; gradient = negative_slope (C++ branch)
       - non-uniform upstream gradient dy scaling
  4. C¹ kink protection: for negative_slope != 1 the function is C¹-discontinuous
     at x = 0, so numerical-gradient tests must call ``avoid_c1_discontinuity``
     to push near-zero points away from the kink.
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
NUM_RTOL = 5e-3  # Type-B (C¹ discontinuous) tolerance; float32 forward limits precision
NUM_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_leaky_relu_prototxt(N=1, C=1, H=4, W=5, negative_slope=0.01):
    """Create a minimal Input(N,C,H,W) -> LeakyReLU prototxt."""
    return textwrap.dedent(f"""\
        name: "test_leaky_relu_bw"
        input: "data"
        input_dim: {N}
        input_dim: {C}
        input_dim: {H}
        input_dim: {W}
        layer {{
          name: "leaky_relu"
          type: "LeakyReLU"
          bottom: "data"
          top: "out"
          leaky_relu_param {{
            negative_slope: {negative_slope}
          }}
        }}
    """)


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _leaky_relu_ref(x, negative_slope=0.01):
    """Forward y = max(x,0)+slope*min(x,0) and backward factor dx/dy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.where(x > 0, x, negative_slope * x)
    dx_factor = np.where(x > 0, 1.0, negative_slope)
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
# Tests: LeakyReLU Forward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestLeakyReLUForward:
    """LeakyReLU forward correctness (hand-computed known values)."""

    def _make_net(self, negative_slope=0.1):
        return Net(_make_leaky_relu_prototxt(N=1, C=1, H=1, W=6, negative_slope=negative_slope))

    def test_forward_known_values(self):
        """Hand-computed y = max(x,0)+slope*min(x,0) with slope=0.1."""
        net = self._make_net(negative_slope=0.1)
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 10.0], dtype=np.float32).reshape(1, 1, 1, 6)
        expected = np.array([0.0, 1.0, -0.1, 2.0, -0.2, 10.0], dtype=np.float32).reshape(1, 1, 1, 6)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_forward_positive_identity(self):
        """Positive x -> y = x (identity)."""
        net = self._make_net(negative_slope=0.01)
        x = np.array([1.0, 2.5, 100.0, 0.5], dtype=np.float32).reshape(1, 1, 1, 4)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-6)

    def test_forward_negative_scaled(self):
        """Negative x -> y = negative_slope * x (scaled)."""
        net = self._make_net(negative_slope=0.2)
        x = np.array([-1.0, -2.0, -5.0, -0.5], dtype=np.float32).reshape(1, 1, 1, 4)
        out = net.forward({"data": x})["out"]
        expected = x * 0.2
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_forward_zero(self):
        """x=0 -> y=0 (continuous through kink)."""
        net = self._make_net(negative_slope=0.1)
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        out = net.forward({"data": x})["out"]
        np.testing.assert_array_equal(out, np.zeros_like(out))


# ---------------------------------------------------------------------------
# Tests: LeakyReLU Backward (analytical)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestLeakyReLUGradient:
    """LeakyReLU backward analytical gradient tests."""

    def _make_net(self, negative_slope=0.1):
        return Net(_make_leaky_relu_prototxt(negative_slope=negative_slope))

    def test_backward_analytical(self):
        """dx = dy * (x>0 ? 1 : slope) for random x (mixed signs)."""
        net = self._make_net(negative_slope=0.1)
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = dy * np.where(x > 0, 1.0, 0.1)
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    def test_backward_zero_input(self):
        """x=0 -> gradient = negative_slope (C++ branch uses x > 0 ? 1 : slope)."""
        net = self._make_net(negative_slope=0.1)
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        dy = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3) + 1.0
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, 0.1 * dy, rtol=1e-6, atol=1e-6)

    def test_backward_positive_region(self):
        """x > 0: dx = dy (identity)."""
        net = self._make_net(negative_slope=0.1)
        x = np.array([1.0, 2.0, 3.0, 0.5, 10.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 1, 1, 5)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, dy, rtol=1e-6, atol=1e-6)

    def test_backward_negative_region(self):
        """x < 0: dx = dy * negative_slope."""
        net = self._make_net(negative_slope=0.2)
        x = np.array([-1.0, -2.0, -3.0, -0.5, -10.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 1, 1, 5)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, 0.2 * dy, rtol=1e-6, atol=1e-6)

    def test_backward_nonuniform_dy(self):
        """Non-uniform upstream gradient dy is scaled correctly."""
        net = self._make_net(negative_slope=0.05)
        rng = np.random.RandomState(99)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32) * 5.0
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = dy * np.where(x > 0, 1.0, 0.05)
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    def test_backward_default_slope(self):
        """Uses proto default negative_slope=0.01 when param omitted."""
        net = Net(_make_leaky_relu_prototxt())  # no slope specified -> default 0.01
        x = np.array([-1.0, -2.0, 3.0], dtype=np.float32).reshape(1, 1, 1, 3)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected = np.where(x > 0, 1.0, 0.01)
        np.testing.assert_allclose(dx, expected, rtol=1e-6, atol=1e-6)

    def test_backward_runs_without_crash(self):
        """Backward should complete without errors on random data."""
        net = self._make_net(negative_slope=0.1)
        x = np.random.RandomState(1).randn(1, 1, 4, 5).astype(np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == x.shape
        assert dx.dtype == np.float32
        assert np.all(np.isfinite(dx))


# ---------------------------------------------------------------------------
# Tests: LeakyReLU Numerical Gradient
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestLeakyReLINumericalGradient:
    """Central finite-difference gradient checks (Type-B, C¹ kink protection)."""

    def _make_net(self, negative_slope=0.1):
        return Net(_make_leaky_relu_prototxt(N=1, C=1, H=3, W=4, negative_slope=negative_slope))

    def test_numerical_gradient_mixed_signs(self):
        """Numerical check with mixed signs; avoids C¹ kink at x=0."""
        net = self._make_net(negative_slope=0.1)
        rng = np.random.RandomState(7)
        x = rng.randn(1, 1, 3, 4).astype(np.float32) * 2.0
        x = avoid_c1_discontinuity(x, h=EPS)  # push near-zero points away from kink
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_positive_only(self):
        """Numerical check in strictly-positive region (no kink; full precision)."""
        net = self._make_net(negative_slope=0.1)
        rng = np.random.RandomState(13)
        x = (rng.rand(1, 1, 3, 4).astype(np.float32) * 2.0) + 0.5  # (0.5, 2.5)
        dy = np.ones_like(x, dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_negative_only(self):
        """Numerical check in strictly-negative region far from kink (full precision)."""
        net = self._make_net(negative_slope=0.1)
        rng = np.random.RandomState(21)
        x = -(rng.rand(1, 1, 3, 4).astype(np.float32) * 2.0) - 0.5  # (-2.5, -0.5)
        dy = np.ones_like(x, dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])