"""P1 activation-layer Forward/Backward tests for the 7 P1 activation ops.

Covers Threshold, Power, BNLL, Clip, Exp, Log, Swish:
  1. Forward numerical correctness vs numpy reference
  2. Operator-specific branches (threshold, clip min/max, power/exp/log base,
     swish beta)
  3. Numerical gradient (central finite differences) with C^1 kinks pushed
     away via ``avoid_c1_discontinuity``
  4. Registration / instantiation (the layer can be created by the Net)

Notes on differentiability:
  - Threshold is NOT differentiable (Backward is a no-op); validated as such.
  - Clip has C^1 kinks at x=min and x=max; gradient test avoids them.
  - BNLL (softplus) is C^inf but the layer is piecewise implemented, so the
    x=0 point is pushed away per the p1 convention.
  - Power uses an integer power (2) for the gradient test to avoid the
    non-integer-power singular surface at t=0.
  - Log requires its argument (scale*x + shift) to stay positive.
"""

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from .caffe_test_helpers import make_net, avoid_c1_discontinuity

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
EPS = 1e-3                 # central finite-difference step
FWD_RTOL = 1e-4            # forward numpy-reference tolerance
FWD_ATOL = 1e-4
NUM_RTOL = 1e-3            # numerical-gradient tolerance
NUM_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Prototxt builder + shared helpers
# ---------------------------------------------------------------------------

def _make_proto(layer_type, params="", shape=(2, 3)):
    """Build an Input(N,C,...) -> <layer_type> prototxt."""
    dims = " ".join(f"dim: {d}" for d in shape)
    return textwrap.dedent(f"""\
        name: "test_{layer_type.lower()}"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ {dims} }} }}
        }}
        layer {{
          name: "act"
          type: "{layer_type}"
          bottom: "data"
          top: "out"
          {params}
        }}
    """)


def _num_grad(net, out_name, x, dy, h=EPS):
    """Central finite-difference gradient dL/dx_i, L = sum(dy * out)."""
    grad = np.zeros_like(x, dtype=np.float64)
    flat_x = x.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x.size):
        orig = flat_x[i]
        xp = x.copy()
        xp.ravel()[i] = orig + h
        out_p = net.forward({"data": xp.astype(np.float32)})[out_name]
        loss_p = float(np.sum(dy * out_p))
        xm = x.copy()
        xm.ravel()[i] = orig - h
        out_m = net.forward({"data": xm.astype(np.float32)})[out_name]
        loss_m = float(np.sum(dy * out_m))
        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad


def _analytic_grad(net, out_name, x, dy):
    """Run forward + backward and return the input gradient dX."""
    net.forward({"data": x})
    net.backward({out_name: dy})
    return net.blob_by_name("data").diff


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestThreshold:
    """Threshold: y = (x > threshold) ? 1 : 0. Not differentiable."""

    def _make_net(self, threshold=0.0):
        return make_net(_make_proto("Threshold", f"threshold_param {{ threshold: {threshold} }}"))

    def test_forward_threshold_zero(self):
        """threshold=0 (default): y = (x > 0) ? 1 : 0."""
        net = self._make_net(0.0)
        x = np.array([-1.0, 0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float32).reshape(2, 3)
        out = net.forward({"data": x})["out"]
        expected = (x > 0.0).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_threshold_half(self):
        """threshold=0.5: y = (x > 0.5) ? 1 : 0."""
        net = self._make_net(0.5)
        x = np.array([-1.0, 0.0, 0.5, 0.6, 1.0, -0.5], dtype=np.float32).reshape(2, 3)
        out = net.forward({"data": x})["out"]
        expected = (x > 0.5).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_binary_output(self):
        """Output must be exactly 0/1 binary."""
        net = self._make_net(0.0)
        x = np.random.RandomState(1).randn(2, 3).astype(np.float32) * 3.0
        out = net.forward({"data": x})["out"]
        assert np.all((out == 0.0) | (out == 1.0)), f"Threshold output must be binary, got {out}"

    def test_backward_is_noop(self):
        """Threshold is not differentiable: Backward is a no-op (dx stays 0)."""
        net = self._make_net(0.5)
        x = np.random.RandomState(2).randn(2, 3).astype(np.float32) * 2.0
        dy = np.random.RandomState(3).randn(*x.shape).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert np.all(dx == 0.0), f"Threshold backward should be a no-op, got {dx}"

    def test_registration(self):
        """Threshold layer can be created by the Net and exposes the expected blobs."""
        net = self._make_net(0.5)
        assert net.has_layer("act")
        assert "act" in net.layer_names()
        assert net.has_blob("out")
        assert "out" in net.blob_names()


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestPower:
    """Power: y = (scale*x + shift) ** power."""

    def _make_net(self, power=1.0, scale=1.0, shift=0.0):
        params = (f"power_param {{ power: {power} scale: {scale} shift: {shift} }}")
        return make_net(_make_proto("Power", params))

    def test_forward_default_identity(self):
        """power=1, scale=1, shift=0 => y = x."""
        net = self._make_net(1.0, 1.0, 0.0)
        x = np.random.RandomState(1).randn(2, 3).astype(np.float32) * 2.0
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, x, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_square(self):
        """power=2, scale=1, shift=0 => y = x^2."""
        net = self._make_net(2.0, 1.0, 0.0)
        x = np.random.RandomState(2).randn(2, 3).astype(np.float32) * 2.0
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, x ** 2, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_scale_shift(self):
        """power=3, scale=2, shift=1 => y = (2x + 1)^3."""
        net = self._make_net(3.0, 2.0, 1.0)
        x = np.random.RandomState(3).randn(2, 3).astype(np.float32)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, (2.0 * x + 1.0) ** 3, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_backward_analytical_square(self):
        """power=2, scale=1.5, shift=0.5 => dx = dy * 2*1.5*(1.5x+0.5)."""
        net = self._make_net(2.0, 1.5, 0.5)
        rng = np.random.RandomState(7)
        x = (rng.rand(2, 3).astype(np.float32) * 1.5) + 0.3  # base positive
        dy = rng.randn(*x.shape).astype(np.float32)
        dx = _analytic_grad(net, "out", x, dy)
        t = 1.5 * x + 0.5
        expected = dy * (2.0 * 1.5 * t ** 1.0)
        np.testing.assert_allclose(dx, expected, rtol=1e-4, atol=1e-5)

    def test_numerical_gradient(self):
        """power=2 (smooth, no kink): dx == central difference."""
        net = self._make_net(2.0, 1.5, 0.5)
        rng = np.random.RandomState(13)
        x = (rng.rand(2, 3).astype(np.float32) * 1.5) + 0.3   # base positive, no power-kink
        dy = rng.randn(*x.shape).astype(np.float32)
        dx_analytic = _analytic_grad(net, "out", x, dy)
        dx_numeric = _num_grad(net, "out", x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_registration(self):
        """Power layer can be created by the Net."""
        net = self._make_net(2.0, 1.0, 0.0)
        assert net.has_layer("act")
        assert net.has_blob("out")


# ---------------------------------------------------------------------------
# BNLL (softplus)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestBNLL:
    """BNLL: y = log(1 + exp(x)) (numerically-stable softplus)."""

    def _make_net(self):
        return make_net(_make_proto("BNLL"))

    def test_forward_reference(self):
        """y = log1p(exp(x)) for mixed-sign inputs."""
        net = self._make_net()
        rng = np.random.RandomState(1)
        x = rng.randn(2, 3).astype(np.float32) * 2.0
        out = net.forward({"data": x})["out"]
        expected = np.log1p(np.exp(x.astype(np.float64))).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_known_values(self):
        """y(0)=ln2, y(1)=ln(1+e), y(-1)=ln(1+1/e)."""
        net = self._make_net()
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32).reshape(1, 5)
        expected = np.array(
            [np.log(2.0), np.log(1.0 + np.e), np.log(1.0 + np.exp(-1.0)),
             np.log(1.0 + np.exp(2.0)), np.log(1.0 + np.exp(-2.0))],
            dtype=np.float32).reshape(1, 5)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_forward_large_positive_no_overflow(self):
        """Large positive x => y ~ x (stable branch)."""
        net = self._make_net()
        x = np.array([100.0, 500.0, 1000.0], dtype=np.float32).reshape(1, 3)
        out = net.forward({"data": x})["out"]
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, x, rtol=1e-4, atol=1e-4)

    def test_backward_analytical(self):
        """dx = dy * sigmoid(x)."""
        net = self._make_net()
        rng = np.random.RandomState(21)
        x = rng.randn(2, 3).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        dx = _analytic_grad(net, "out", x, dy)
        expected = dy / (1.0 + np.exp(-x))  # sigmoid(x) = d/dx log(1+e^x)
        np.testing.assert_allclose(dx, expected, rtol=1e-4, atol=1e-5)

    def test_numerical_gradient(self):
        """BNLL is C^inf; push x=0 away per convention, then central-diff check."""
        net = self._make_net()
        rng = np.random.RandomState(55)
        x = rng.randn(2, 3).astype(np.float32) * 2.0
        x = avoid_c1_discontinuity(x, h=EPS, kink_points=0.0)
        dy = rng.randn(*x.shape).astype(np.float32)
        dx_analytic = _analytic_grad(net, "out", x, dy)
        dx_numeric = _num_grad(net, "out", x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_registration(self):
        """BNLL layer can be created by the Net."""
        net = self._make_net()
        assert net.has_layer("act")
        assert net.has_blob("out")


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestClip:
    """Clip: y = clamp(x, min, max)."""

    def _make_net(self, min_val=-1.0, max_val=1.0):
        params = f"clip_param {{ min: {min_val} max: {max_val} }}"
        return make_net(_make_proto("Clip", params))

    def test_forward_symmetric(self):
        """min=-1, max=1: y = clamp(x, -1, 1)."""
        net = self._make_net(-1.0, 1.0)
        x = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5], dtype=np.float32).reshape(1, 7)
        out = net.forward({"data": x})["out"]
        expected = np.clip(x, -1.0, 1.0)
        np.testing.assert_allclose(out, expected, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_asymmetric_boundary(self):
        """min=0.2, max=0.8: interior values pass through, outside clamped."""
        net = self._make_net(0.2, 0.8)
        x = np.array([0.0, 0.2, 0.5, 0.8, 1.0, -0.5], dtype=np.float32).reshape(2, 3)
        out = net.forward({"data": x})["out"]
        expected = np.clip(x, 0.2, 0.8)
        np.testing.assert_allclose(out, expected, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_backward_analytical_interior(self):
        """Inside (min, max): dx = dy; outside: dx = 0."""
        net = self._make_net(-1.0, 1.0)
        rng = np.random.RandomState(7)
        x = (rng.rand(2, 3).astype(np.float32) - 0.5) * 1.0  # in (-0.5, 0.5), interior
        dy = rng.randn(*x.shape).astype(np.float32)
        dx = _analytic_grad(net, "out", x, dy)
        np.testing.assert_allclose(dx, dy, rtol=1e-5, atol=1e-6)

    def test_backward_saturated_zero(self):
        """x outside [min, max] => dx = 0."""
        net = self._make_net(-1.0, 1.0)
        x = np.array([-3.0, -2.0, 2.0, 3.0, -5.0, 5.0], dtype=np.float32).reshape(2, 3)
        dy = np.ones_like(x)
        # push exactly-at-boundary values off the kink isn't needed here (all outside)
        dx = _analytic_grad(net, "out", x, dy)
        assert np.all(dx == 0.0), f"Saturated Clip should give zero dx, got {dx}"

    def test_numerical_gradient(self):
        """Interior region, kinks at min/max pushed away."""
        net = self._make_net(-1.0, 1.0)
        rng = np.random.RandomState(13)
        x = (rng.rand(2, 3).astype(np.float32) - 0.5) * 1.0   # in (-0.5, 0.5)
        x = avoid_c1_discontinuity(x, h=EPS, kink_points=(-1.0, 1.0))
        dy = rng.randn(*x.shape).astype(np.float32)
        dx_analytic = _analytic_grad(net, "out", x, dy)
        dx_numeric = _num_grad(net, "out", x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_registration(self):
        """Clip layer can be created by the Net."""
        net = self._make_net(-1.0, 1.0)
        assert net.has_layer("act")
        assert net.has_blob("out")


# ---------------------------------------------------------------------------
# Exp
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestExp:
    """Exp: y = base^(scale*x + shift); base=-1 means natural e."""

    def _make_net(self, base=-1.0, scale=1.0, shift=0.0):
        params = f"exp_param {{ base: {base} scale: {scale} shift: {shift} }}"
        return make_net(_make_proto("Exp", params))

    def test_forward_default_exp(self):
        """base=-1, scale=1, shift=0 => y = exp(x)."""
        net = self._make_net(-1.0, 1.0, 0.0)
        x = np.random.RandomState(1).randn(2, 3).astype(np.float32)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, np.exp(x), rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_base_three(self):
        """base=3, scale=1, shift=0 => y = 3^x."""
        net = self._make_net(3.0, 1.0, 0.0)
        x = np.random.RandomState(2).randn(2, 3).astype(np.float32)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, 3.0 ** x, rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_scale_shift(self):
        """base=-1, scale=2, shift=1 => y = e * exp(2x)."""
        net = self._make_net(-1.0, 2.0, 1.0)
        x = np.random.RandomState(3).randn(2, 3).astype(np.float32)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, np.e * np.exp(2.0 * x), rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_backward_analytical(self):
        """base=-1, scale=1.5, shift=0 => dx = dy * exp(1.5x) * 1.5."""
        net = self._make_net(-1.0, 1.5, 0.0)
        rng = np.random.RandomState(7)
        x = rng.randn(2, 3).astype(np.float32)
        dy = rng.randn(*x.shape).astype(np.float32)
        dx = _analytic_grad(net, "out", x, dy)
        expected = dy * np.exp(1.5 * x) * 1.5
        np.testing.assert_allclose(dx, expected, rtol=1e-4, atol=1e-5)

    def test_numerical_gradient(self):
        """Exp is smooth; central-difference check."""
        net = self._make_net(-1.0, 1.5, 0.0)
        rng = np.random.RandomState(13)
        x = rng.randn(2, 3).astype(np.float32)
        dy = rng.randn(*x.shape).astype(np.float32)
        dx_analytic = _analytic_grad(net, "out", x, dy)
        dx_numeric = _num_grad(net, "out", x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_registration(self):
        """Exp layer can be created by the Net."""
        net = self._make_net(-1.0, 1.0, 0.0)
        assert net.has_layer("act")
        assert net.has_blob("out")


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestLog:
    """Log: y = log_base(scale*x + shift); base=-1 means natural e."""

    def _make_net(self, base=-1.0, scale=1.0, shift=0.0):
        params = f"log_param {{ base: {base} scale: {scale} shift: {shift} }}"
        return make_net(_make_proto("Log", params))

    def _positive_x(self, rng, shape=(2, 3), lo=0.5, hi=2.5):
        return (rng.rand(*shape).astype(np.float32) * (hi - lo)) + lo

    def test_forward_default_ln(self):
        """base=-1, scale=1, shift=0 => y = log(x)."""
        net = self._make_net(-1.0, 1.0, 0.0)
        x = self._positive_x(np.random.RandomState(1))
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, np.log(x), rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_base_two(self):
        """base=2, scale=1, shift=0 => y = log2(x)."""
        net = self._make_net(2.0, 1.0, 0.0)
        x = self._positive_x(np.random.RandomState(2))
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, np.log(x) / np.log(2.0), rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_scale_shift(self):
        """base=-1, scale=2, shift=1 => y = log(2x + 1)."""
        net = self._make_net(-1.0, 2.0, 1.0)
        x = self._positive_x(np.random.RandomState(3), lo=0.2, hi=1.5)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, np.log(2.0 * x + 1.0), rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_backward_analytical(self):
        """base=-1, scale=1.5, shift=0.5 => dx = dy * 1.5 / (1.5x + 0.5)."""
        net = self._make_net(-1.0, 1.5, 0.5)
        rng = np.random.RandomState(7)
        x = self._positive_x(rng, lo=0.5, hi=2.0)
        dy = rng.randn(*x.shape).astype(np.float32)
        dx = _analytic_grad(net, "out", x, dy)
        t = 1.5 * x + 0.5
        expected = dy * (1.5 / t)
        np.testing.assert_allclose(dx, expected, rtol=1e-4, atol=1e-5)

    def test_numerical_gradient(self):
        """Log argument kept positive; x=0 singularity avoided."""
        net = self._make_net(-1.0, 1.5, 0.5)
        rng = np.random.RandomState(13)
        x = self._positive_x(rng, lo=0.5, hi=2.0)
        x = avoid_c1_discontinuity(x, h=EPS, kink_points=0.0)
        dy = rng.randn(*x.shape).astype(np.float32)
        dx_analytic = _analytic_grad(net, "out", x, dy)
        dx_numeric = _num_grad(net, "out", x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_registration(self):
        """Log layer can be created by the Net."""
        net = self._make_net(-1.0, 1.0, 0.0)
        assert net.has_layer("act")
        assert net.has_blob("out")


# ---------------------------------------------------------------------------
# Swish
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSwish:
    """Swish: y = x * sigmoid(beta * x)."""

    def _make_net(self, beta=1.0):
        return make_net(_make_proto("Swish", f"swish_param {{ beta: {beta} }}"))

    def _ref(self, x, beta):
        s = 1.0 / (1.0 + np.exp(-beta * x))
        return x * s

    def test_forward_default_beta(self):
        """beta=1: y = x * sigmoid(x) == x/(1+exp(-x))."""
        net = self._make_net(1.0)
        x = np.random.RandomState(1).randn(2, 3).astype(np.float32) * 2.0
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, self._ref(x, 1.0), rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_beta_two(self):
        """beta=2: y = x * sigmoid(2x)."""
        net = self._make_net(2.0)
        x = np.random.RandomState(2).randn(2, 3).astype(np.float32) * 2.0
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, self._ref(x, 2.0), rtol=FWD_RTOL, atol=FWD_ATOL)

    def test_forward_zero_input(self):
        """y(0) = 0 for any beta."""
        net = self._make_net(2.0)
        x = np.zeros((2, 3), dtype=np.float32)
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_backward_analytical(self):
        """beta=1.5: dx = dy * (beta*y + sigmoid*(1 - beta*y))."""
        net = self._make_net(1.5)
        rng = np.random.RandomState(7)
        x = rng.randn(2, 3).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        dx = _analytic_grad(net, "out", x, dy)
        beta = 1.5
        s = 1.0 / (1.0 + np.exp(-beta * x))
        y = x * s
        expected = dy * (beta * y + s * (1.0 - beta * y))
        np.testing.assert_allclose(dx, expected, rtol=1e-4, atol=1e-5)

    def test_numerical_gradient(self):
        """Swish is smooth; central-difference check."""
        net = self._make_net(1.5)
        rng = np.random.RandomState(13)
        x = rng.randn(2, 3).astype(np.float32) * 2.0
        dy = rng.randn(*x.shape).astype(np.float32)
        dx_analytic = _analytic_grad(net, "out", x, dy)
        dx_numeric = _num_grad(net, "out", x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_registration(self):
        """Swish layer can be created by the Net."""
        net = self._make_net(1.0)
        assert net.has_layer("act")
        assert net.has_blob("out")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])