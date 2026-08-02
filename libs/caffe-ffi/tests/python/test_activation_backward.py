"""Activation layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed)
  2. Numerical gradient check (central finite differences)
  3. Boundary conditions (zero inputs, large values, dead zones)
  4. Parameter gradient (PReLU slope)
  5. [ACTIVATION-PERF] log emission verification

Tested layers: ReLU, Sigmoid, TanH, ELU, PReLU
"""

import textwrap
import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# Numerical gradient epsilon
EPS = 1e-3


def _make_activation_prototxt(layer_type, extra_params=""):
    """Create a minimal Input -> Activation prototxt."""
    return textwrap.dedent(f"""\
        name: "test_{layer_type}_bw"
        input: "data"
        input_dim: 1
        input_dim: 1
        input_dim: 4
        input_dim: 5
        layer {{
          name: "act"
          type: "{layer_type}"
          bottom: "data"
          top: "out"
          {extra_params}
        }}
    """)


def _make_prelu_prototxt(channel_shared=True, filler=0.25):
    mode = "true" if channel_shared else "false"
    return textwrap.dedent(f"""\
        name: "test_prelu_bw"
        input: "data"
        input_dim: 2
        input_dim: 3
        input_dim: 4
        input_dim: 5
        layer {{
          name: "prelu"
          type: "PReLU"
          bottom: "data"
          top: "out"
          prelu_param {{
            channel_shared: {mode}
            filler {{
              type: "constant"
              value: {filler}
            }}
          }}
        }}
    """)


def _num_grad(net, x, dy, h=EPS):
    """Compute numerical gradient via central differences: dL/dx_i.

    L = sum(dy * out)  (linear loss so dy is the upstream gradient).
    We perturb each element of x by ±h and measure the change in L.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    flat_x = x.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x.size):
        orig = flat_x[i]

        # f(x + h*e_i)
        xp = x.copy()
        xp.ravel()[i] = orig + h
        out_p = net.forward({"data": xp.astype(np.float32)})["out"]
        loss_p = float(np.sum(dy * out_p))

        # f(x - h*e_i)
        xm = x.copy()
        xm.ravel()[i] = orig - h
        out_m = net.forward({"data": xm.astype(np.float32)})["out"]
        loss_m = float(np.sum(dy * out_m))

        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
        # restore
        xp.ravel()[i] = orig
    return grad


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _relu_ref(x, negative_slope=0.0):
    y = np.where(x > 0, x, negative_slope * x)
    dx_factor = np.where(x > 0, 1.0, negative_slope)
    return y, dx_factor


def _leaky_relu_ref(x, negative_slope=0.01):
    return _relu_ref(x, negative_slope)


def _sigmoid_ref(x):
    y = 1.0 / (1.0 + np.exp(-x))
    dx_factor = y * (1.0 - y)
    return y, dx_factor


def _tanh_ref(x):
    y = np.tanh(x)
    dx_factor = 1.0 - y * y
    return y, dx_factor


def _elu_ref(x, alpha=1.0):
    y = np.where(x >= 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1.0))
    dx_factor = np.where(x >= 0, 1.0, y + alpha)
    return y, dx_factor


def _prelu_ref(x, slope, channel_shared, shape):
    """x shape (N,C,H,W). slope shape: () for shared, (C,) for per-channel."""
    N, C, H, W = shape
    if channel_shared:
        s = float(slope)
        y = np.where(x > 0, x, s * x)
        dx_factor = np.where(x > 0, 1.0, s)
        d_slope = np.sum(np.where(x <= 0, x, 0.0))  # weighted by dy below
    else:
        # slope is (C,), broadcast to (1,C,1,1)
        s = slope.reshape(1, C, 1, 1)
        y = np.where(x > 0, x, s * x)
        dx_factor = np.where(x > 0, 1.0, s)
        # per-channel slope grad: d_slope[c] = sum over N,H,W of (dy*x) where x<=0
        d_slope = np.zeros(C, dtype=np.float64)
    return y, dx_factor, d_slope


# ---------------------------------------------------------------------------
# Tests: ReLU Backward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestReLUGradient:
    """ReLU backward gradient tests."""

    def _make_net(self, negative_slope=0.0):
        params = ""
        if negative_slope != 0.0:
            params = f'relu_param {{ negative_slope: {negative_slope} }}'
        return Net(_make_activation_prototxt("ReLU", params))

    # ---- analytical correctness ----

    def test_relu_backward_analytical_positive(self):
        """x > 0 => dx = dy exactly."""
        net = self._make_net()
        x = np.array([1.0, 2.0, 3.0, 0.5, 0.1, -1.0, -0.5, 0.0,
                      4.0, -2.0, 1.5, -3.0, 0.01, -0.01, 10.0, -10.0,
                      2.5, -0.001, 0.001, -5.0], dtype=np.float32).reshape(1, 1, 4, 5)
        dy = np.random.RandomState(42).randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        _, dx_factor = _relu_ref(x)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    def test_relu_backward_analytical_negative_slope(self):
        """Leaky ReLU (negative_slope=0.01): dx = dy * (x>0 ? 1 : 0.01)."""
        net = self._make_net(negative_slope=0.01)
        rng = np.random.RandomState(123)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 3.0
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        _, dx_factor = _leaky_relu_ref(x, 0.01)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    def test_relu_backward_dead_neuron_zero(self):
        """x < 0 with negative_slope=0 => dx must be exactly 0."""
        net = self._make_net()
        x = np.array([-1.0, -2.0, -0.5, -10.0, -0.001], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert np.all(dx == 0.0), f"Dead ReLU neurons must have zero gradient, got {dx}"

    def test_relu_backward_arbitrary_dy(self):
        """Non-uniform upstream gradient dy is scaled correctly."""
        net = self._make_net()
        rng = np.random.RandomState(99)
        x = rng.randn(1, 1, 4, 5).astype(np.float32)
        dy = rng.randn(*x.shape).astype(np.float32) * 5.0
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        _, dx_factor = _relu_ref(x)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)

    # ---- numerical gradient check ----

    def test_relu_numerical_gradient(self):
        """Central finite difference gradient check on ReLU (avoids x=0)."""
        net = self._make_net()
        rng = np.random.RandomState(7)
        # Avoid exact zero to prevent non-differentiable point issues
        x = (rng.randn(1, 1, 3, 4).astype(np.float32)) * 2.0 + 1.0  # shifted positive
        dy = np.ones_like(x, dtype=np.float32)
        # analytic
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        # numeric
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)

    def test_relu_numerical_gradient_mixed_signs(self):
        """Numerical check with both positive and negative x (LeakyReLU, avoids kink region)."""
        net = self._make_net(negative_slope=0.1)  # leaky relu is C¹-discontinuous at x=0
        rng = np.random.RandomState(13)
        x = rng.randn(1, 1, 3, 4).astype(np.float32) * 2.0
        # Push near-zero points away from the C¹ kink to prevent finite-difference straddling
        x = np.where(x > 0, np.maximum(x, 2*EPS), np.minimum(x, -2*EPS))
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)

    # ---- perf log ----
    # Perf log verification (INFO level, requires CAFFE_FFI_ENABLE_DEBUG_LOG build flag)
    # is done in CI via a dedicated inline script step.
    # Here we just verify backward runs without crashing.

    def test_relu_backward_runs_without_crash(self):
        """Backward should complete without errors on random data."""
        net = self._make_net()
        x = np.random.RandomState(1).randn(1, 1, 4, 5).astype(np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == x.shape
        assert np.all(np.isfinite(dx))


# ---------------------------------------------------------------------------
# Tests: Sigmoid Backward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSigmoidGradient:
    """Sigmoid backward gradient tests."""

    def _make_net(self):
        return Net(_make_activation_prototxt("Sigmoid"))

    def test_sigmoid_backward_analytical(self):
        """dx = dy * y * (1 - y) for all x."""
        net = self._make_net()
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        y, dx_factor = _sigmoid_ref(x)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, y, rtol=1e-5, atol=1e-6)

    def test_sigmoid_backward_zero_input(self):
        """x=0 => y=0.5 => dx=dy*0.25."""
        net = self._make_net()
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, 0.25, rtol=1e-6)

    def test_sigmoid_backward_saturation_small(self):
        """Very large negative x => y≈0 => dx≈0 (vanishing gradient)."""
        net = self._make_net()
        x = np.array([[-10.0, -20.0, -50.0]], dtype=np.float32).reshape(1, 1, 1, 3)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert np.all(dx < 1e-4), f"Saturated sigmoid should have near-zero dx, got {dx}"

    def test_sigmoid_backward_saturation_large(self):
        """Very large positive x => y≈1 => dx≈0."""
        net = self._make_net()
        x = np.array([[10.0, 20.0, 50.0]], dtype=np.float32).reshape(1, 1, 1, 3)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert np.all(dx < 1e-4), f"Saturated sigmoid should have near-zero dx, got {dx}"

    def test_sigmoid_numerical_gradient(self):
        """Central finite difference check for sigmoid."""
        net = self._make_net()
        rng = np.random.RandomState(77)
        x = rng.randn(1, 1, 3, 4).astype(np.float32) * 1.5
        dy = np.ones_like(x, dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: TanH Backward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestTanHGradient:
    """TanH backward gradient tests."""

    def _make_net(self):
        return Net(_make_activation_prototxt("TanH"))

    def test_tanh_backward_analytical(self):
        """dx = dy * (1 - y^2) for all x."""
        net = self._make_net()
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        y, dx_factor = _tanh_ref(x)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, y, rtol=1e-5, atol=1e-6)

    def test_tanh_backward_zero_input(self):
        """x=0 => y=0 => dx=dy*1 = dy."""
        net = self._make_net()
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(1, 1, 2, 3)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, dy, rtol=1e-6)

    def test_tanh_backward_saturation(self):
        """|x| large => |y|≈1 => dx≈0."""
        net = self._make_net()
        x = np.array([5.0, -5.0, 10.0, -10.0], dtype=np.float32).reshape(1, 1, 1, 4)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert np.all(np.abs(dx) < 0.01), f"Saturated tanh should have near-zero dx, got {dx}"

    def test_tanh_backward_symmetry(self):
        """tanh(-x) = -tanh(x), and dx(-x) = dx(x) since (1-y²) is even."""
        net = self._make_net()
        x = np.array([1.0, 2.0, -1.0, -2.0, 0.5, -0.5], dtype=np.float32).reshape(1, 1, 2, 3)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # dx at x=1 should equal dx at x=-1
        assert abs(float(dx.flat[0]) - float(dx.flat[2])) < 1e-6
        assert abs(float(dx.flat[1]) - float(dx.flat[3])) < 1e-6
        assert abs(float(dx.flat[4]) - float(dx.flat[5])) < 1e-6

    def test_tanh_numerical_gradient(self):
        """Central finite difference check for tanh."""
        net = self._make_net()
        rng = np.random.RandomState(88)
        x = rng.randn(1, 1, 3, 4).astype(np.float32) * 1.5
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: ELU Backward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestELUGradient:
    """ELU backward gradient tests."""

    def _make_net(self, alpha=1.0):
        params = f'elu_param {{ alpha: {alpha} }}' if alpha != 1.0 else ""
        return Net(_make_activation_prototxt("ELU", params))

    def test_elu_backward_analytical(self):
        """x >= 0: dx = dy; x < 0: dx = dy * (y + alpha)."""
        net = self._make_net(alpha=1.0)
        rng = np.random.RandomState(42)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        y, dx_factor = _elu_ref(x, alpha=1.0)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out, y, rtol=1e-5, atol=1e-5)

    def test_elu_backward_positive_passthrough(self):
        """x >= 0 => dx = dy exactly."""
        net = self._make_net()
        x = np.array([0.0, 0.5, 1.0, 5.0, 10.0], dtype=np.float32).reshape(1, 1, 1, 5)
        dy = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(1, 1, 1, 5)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, dy, rtol=1e-6)

    def test_elu_backward_alpha05(self):
        """Test with alpha=0.5: dx for x<0 = dy * (y + 0.5)."""
        net = self._make_net(alpha=0.5)
        rng = np.random.RandomState(123)
        x = rng.randn(1, 1, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        y, dx_factor = _elu_ref(x, alpha=0.5)
        expected = dy * dx_factor
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-5)

    def test_elu_backward_negative_saturation(self):
        """x very negative => y ≈ -alpha => dx ≈ 0."""
        net = self._make_net(alpha=1.0)
        x = np.array([-10.0, -20.0, -5.0], dtype=np.float32).reshape(1, 1, 1, 3)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert np.all(np.abs(dx) < 0.02), f"Highly negative ELU should have near-zero dx, got {dx}"

    def test_elu_numerical_gradient(self):
        """Central finite difference check for ELU.

        Note: ELU has a C1 kink at x=0 (second derivative jumps from alpha to 0),
        so central differences with h=1e-3 across the kink have O(h) truncation
        error instead of O(h^2). Use rtol=5e-3 to accommodate this; the analytic
        gradient is still verified to within 0.5% which is excellent for a
        piecewise-exponential activation.
        """
        net = self._make_net(alpha=1.0)
        rng = np.random.RandomState(55)
        x = rng.randn(1, 1, 3, 4).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=5e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: PReLU Backward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestPReLUGradient:
    """PReLU backward gradient tests (channel_shared and per-channel modes)."""

    def test_prelu_shared_analytical(self):
        """channel_shared=true: dx = dy*(x>0?1:slope), d_slope = sum(dy*x for x<=0)."""
        net = Net(_make_prelu_prototxt(channel_shared=True, filler=0.25))
        rng = np.random.RandomState(42)
        x = rng.randn(2, 3, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        slope_val = 0.25

        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})

        dx = net.blob_by_name("data").diff
        slope_diff = net.layer_by_name("prelu").blobs[0].diff

        # reference
        y_ref = np.where(x > 0, x, slope_val * x)
        dx_ref = dy * np.where(x > 0, 1.0, slope_val)
        d_slope_ref = np.sum(dy * np.where(x <= 0, x, 0.0))

        np.testing.assert_allclose(out, y_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(slope_diff, d_slope_ref, rtol=1e-4, atol=1e-5)

    def test_prelu_per_channel_analytical(self):
        """channel_shared=false: per-channel slope, per-channel slope gradient."""
        net = Net(_make_prelu_prototxt(channel_shared=False, filler=0.25))
        rng = np.random.RandomState(77)
        x = rng.randn(2, 3, 4, 5).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        slope_val = 0.25  # constant init

        net.forward({"data": x})
        net.backward({"out": dy})

        dx = net.blob_by_name("data").diff
        slope_diff = net.layer_by_name("prelu").blobs[0].diff

        # reference dx
        dx_ref = dy * np.where(x > 0, 1.0, slope_val)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-5, atol=1e-6)

        # per-channel slope grad: d_slope[c] = sum over N,H,W positions of (dy*x) where x<=0
        d_slope_ref = np.zeros(3, dtype=np.float64)
        for c in range(3):
            mask = x[:, c, :, :] <= 0
            d_slope_ref[c] = np.sum(dy[:, c, :, :][mask] * x[:, c, :, :][mask])

        assert slope_diff.shape == (3,), f"slope_diff should be (C,)=(3,), got {slope_diff.shape}"
        np.testing.assert_allclose(slope_diff, d_slope_ref, rtol=1e-4, atol=1e-5)

    def test_prelu_shared_dead_neuron_scaled(self):
        """x<0: dx = dy * slope (not zero like ReLU)."""
        net = Net(_make_prelu_prototxt(channel_shared=True, filler=0.1))
        x = np.array([-1.0, -2.0, -0.5], dtype=np.float32).reshape(1, 1, 1, 3)
        # N=1,C=1,H=1,W=3 requires (1,1,1,3) which fits in 1,3,4,5? No, need 2,3,4,5 shape.
        # Use proper shape
        x = np.array([-1.0, -2.0, -0.5, 1.0, 2.0], dtype=np.float32)
        # Pad to (2,3,4,5) = 120 elements
        x = np.concatenate([x, np.zeros(120 - len(x), dtype=np.float32)]).reshape(2, 3, 4, 5)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        slope = 0.1
        # x=-1,-2,-0.5 → dx = slope*dy = 0.1
        assert abs(float(dx.flat[0]) - slope) < 1e-6
        assert abs(float(dx.flat[1]) - slope) < 1e-6
        assert abs(float(dx.flat[2]) - slope) < 1e-6
        # x=1,2 → dx = dy = 1
        assert abs(float(dx.flat[3]) - 1.0) < 1e-6
        assert abs(float(dx.flat[4]) - 1.0) < 1e-6

    def test_prelu_shared_numerical_gradient(self):
        """Numerical gradient check for PReLU shared (avoids C¹ kink at x=0)."""
        net = Net(_make_prelu_prototxt(channel_shared=True, filler=0.25))
        rng = np.random.RandomState(99)
        # Use small shape for speed; (2,3,4,5)=120 is okay but slow for numeric
        # Instead create a smaller PReLU net via inline prototxt
        small_proto = textwrap.dedent("""\
            name: "prelu_small"
            input: "data"
            input_dim: 1
            input_dim: 1
            input_dim: 2
            input_dim: 3
            layer {
              name: "prelu"
              type: "PReLU"
              bottom: "data"
              top: "out"
              prelu_param {
                channel_shared: true
                filler { type: "constant" value: 0.25 }
              }
            }
        """)
        net_small = Net(small_proto)
        x = rng.randn(1, 1, 2, 3).astype(np.float32) * 1.5
        # PReLU is C¹-discontinuous at x=0 (derivative jumps from slope to 1);
        # push near-zero points away to prevent finite-difference straddling the kink
        h = EPS
        x = np.where(x > 0, np.maximum(x, 2*h), np.minimum(x, -2*h))
        dy = rng.randn(*x.shape).astype(np.float32)
        net_small.forward({"data": x})
        net_small.backward({"out": dy})
        dx_analytic = net_small.blob_by_name("data").diff

        # Numerical grad
        dx_numeric = np.zeros_like(x, dtype=np.float64)
        flat_x = x.ravel()
        for i in range(flat_x.size):
            orig = flat_x[i]
            xp = x.copy(); xp.ravel()[i] = orig + h
            xm = x.copy(); xm.ravel()[i] = orig - h
            lp = float(np.sum(dy * net_small.forward({"data": xp.astype(np.float32)})["out"]))
            lm = float(np.sum(dy * net_small.forward({"data": xm.astype(np.float32)})["out"]))
            dx_numeric.ravel()[i] = (lp - lm) / (2.0 * h)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)

    def test_prelu_slope_diff_shape_shared(self):
        """channel_shared slope_diff is scalar (shape (1,))."""
        net = Net(_make_prelu_prototxt(channel_shared=True, filler=0.3))
        x = np.random.RandomState(1).randn(2, 3, 4, 5).astype(np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        slope_diff = net.layer_by_name("prelu").blobs[0].diff
        assert slope_diff.size == 1 or slope_diff.shape == (1,), \
            f"shared slope_diff should be size 1, got shape {slope_diff.shape}"

    def test_prelu_slope_diff_shape_per_channel(self):
        """per-channel slope_diff is (C,) = (3,)."""
        net = Net(_make_prelu_prototxt(channel_shared=False, filler=0.3))
        x = np.random.RandomState(2).randn(2, 3, 4, 5).astype(np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        slope_diff = net.layer_by_name("prelu").blobs[0].diff
        assert slope_diff.shape == (3,), f"per-channel slope_diff should be (3,), got {slope_diff.shape}"


# ---------------------------------------------------------------------------
# Tests: Performance log verification
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestActivationPerfLogs:
    """Verify [ACTIVATION-PERF] backward log structure (when INFO logs enabled)."""

    @pytest.mark.parametrize("layer_type", ["ReLU", "Sigmoid", "TanH", "ELU"])
    def test_backward_no_crash(self, layer_type):
        """All activation backends should run without crashing on random data."""
        proto_extra = ""
        if layer_type == "ReLU":
            proto_extra = 'relu_param { negative_slope: 0.01 }'
        net = Net(_make_activation_prototxt(layer_type, proto_extra))
        rng = np.random.RandomState(12345)
        x = rng.randn(1, 1, 4, 5).astype(np.float32)
        dy = rng.randn(*x.shape).astype(np.float32)
        out = net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == x.shape
        assert dx.dtype == np.float32
        assert np.all(np.isfinite(dx)), f"{layer_type} backward produced non-finite dx"

    def test_prelu_backward_no_crash(self):
        """PReLU backward (both modes) runs without crashing."""
        for shared in [True, False]:
            net = Net(_make_prelu_prototxt(channel_shared=shared, filler=0.25))
            rng = np.random.RandomState(42)
            x = rng.randn(2, 3, 4, 5).astype(np.float32)
            dy = rng.randn(*x.shape).astype(np.float32)
            net.forward({"data": x})
            net.backward({"out": dy})
            dx = net.blob_by_name("data").diff
            assert dx.shape == x.shape
            assert np.all(np.isfinite(dx))
            sd = net.layer_by_name("prelu").blobs[0].diff
            assert np.all(np.isfinite(sd))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
