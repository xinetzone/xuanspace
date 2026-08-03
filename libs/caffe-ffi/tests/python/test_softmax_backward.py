"""Softmax layer Backward gradient tests.

Softmax: y_i = exp(x_i) / sum_j(exp(x_j))
Jacobian: J_{ij} = dy_i/dx_j = y_i * (delta_{ij} - y_j)
Backward (Jacobian-vector product):
    dx_i = y_i * (dy_i - sum_j(dy_j * y_j)) = y_i * (dy_i - dot)
    where dot = dy . y (per outer x inner position)

Covers:
  1. Known-value hand verification (uniform, one-hot, simple 2-class/3-class)
  2. Analytical gradient (numpy reference vs caffe-ffi)
  3. Numerical gradient check (central finite differences)
  4. Multi-axis support (axis=1 for NCHW, axis=-1 for common use)
  5. Zero dy -> zero gradients (for uniform dy? No: only zero dy gives zero dx)
  6. Shape/finite/determinism checks
  7. Probability conservation: sum(y)=1, gradient properties
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import (
    numerical_gradient,
)

EPS = 1e-3
RTOL = 1e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Numpy reference for Softmax forward/backward
# ---------------------------------------------------------------------------

def _softmax_forward_np(x, axis=1):
    """Numpy reference: numerically stable softmax along given axis."""
    x64 = x.astype(np.float64)
    x_max = np.max(x64, axis=axis, keepdims=True)
    exp_x = np.exp(x64 - x_max)
    return (exp_x / np.sum(exp_x, axis=axis, keepdims=True)).astype(np.float32)


def _softmax_backward_np(y, dy, axis=1):
    """Numpy reference: softmax backward (Jacobian-vector product).

    Args:
        y: softmax output (probabilities), same shape as x.
        dy: upstream gradient, same shape as y.
        axis: softmax axis.

    Returns:
        dx: gradient w.r.t. input, same shape.
    """
    y64 = y.astype(np.float64)
    dy64 = dy.astype(np.float64)
    # dot = sum_j(dy_j * y_j) along softmax axis, per position
    dot = np.sum(dy64 * y64, axis=axis, keepdims=True)
    # dx_i = y_i * (dy_i - dot)
    dx = y64 * (dy64 - dot)
    return dx.astype(np.float32)


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_softmax_prototxt(shape, axis=1):
    """Create Input -> Softmax prototxt.

    Args:
        shape: tuple/list of input dimensions (e.g. (2, 3)).
        axis: softmax axis (default 1, i.e. channel axis in NCHW).
    """
    dims_lines = "\n".join(f"          dim: {d}" for d in shape)
    return textwrap.dedent(f"""\
        name: "test_softmax_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{
{dims_lines}
          }} }}
        }}
        layer {{
          name: "sm"
          type: "Softmax"
          bottom: "data"
          top: "out"
          softmax_param {{ axis: {axis} }}
        }}
    """)


def _make_softmax_net(shape, axis=1):
    proto = _make_softmax_prototxt(shape, axis=axis)
    return Net(proto)


# ---------------------------------------------------------------------------
# Helper: run forward+backward
# ---------------------------------------------------------------------------

def _run_softmax_backward(net, x, dy):
    """Run forward then backward, return dx and output y."""
    out = net.forward({"data": x.astype(np.float32)})
    net.backward({"out": dy.astype(np.float32)})
    dx = net.blob_by_name("data").diff
    return dx, out["out"]


def _numerical_grad(net, x, dy, h=EPS):
    """Compute numerical gradient w.r.t. input via central finite differences."""
    current = x.astype(np.float32).copy()

    def _forward():
        out = net.forward({"data": current})
        return out["out"]

    def _get():
        return current.copy()

    def _set(arr):
        nonlocal current
        np.copyto(current, arr)

    return numerical_gradient(_forward, _get, _set, dy, h=h,
                              name="input:data", verbose=False)


# ---------------------------------------------------------------------------
# Tests: Known values (L1)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftmaxBackwardKnownValues:
    """Hand-computed known value tests."""

    def test_uniform_input_gives_uniform_output(self):
        """Uniform input (all equal) -> uniform output (1/C each),
        uniform dy -> zero gradient (symmetric)."""
        C = 3
        net = _make_softmax_net((1, C), axis=1)
        x = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        dy = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        # Uniform input -> uniform probabilities
        np.testing.assert_allclose(y, np.array([[1/C, 1/C, 1/C]]), rtol=1e-5)
        # When dy is uniform: dot = sum(y_j * dy_j) = dy_0 * sum(y_j) = dy_0 = 1
        # dx_i = y_i * (1 - 1) = 0
        np.testing.assert_allclose(dx, np.zeros_like(x), atol=1e-6)

    def test_two_class_confident(self):
        """2-class: x=[large, 0] -> y≈[1, 0], dy=[0, 1] -> dx≈[0, 0]
        (gradient of near-certain wrong class is near zero)."""
        net = _make_softmax_net((1, 2), axis=1)
        x = np.array([[10.0, 0.0]], dtype=np.float32)
        dy = np.array([[0.0, 1.0]], dtype=np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        # y should be approximately [1, 0]
        assert y[0, 0] > 0.9999
        assert y[0, 1] < 0.0001
        # For near-one-hot, dy on the near-zero class:
        # dot ≈ y1*dy1 ≈ 0 (since y1≈0)
        # dx0 = y0*(dy0 - dot) ≈ 1*(0 - 0) ≈ 0
        # dx1 = y1*(dy1 - dot) ≈ 0*(1 - 0) ≈ 0
        np.testing.assert_allclose(dx, np.zeros_like(x), atol=1e-3)

    def test_three_class_known_values(self):
        """3-class: x=[0, ln2, ln3] -> y = [1/6, 2/6, 3/6] = [1/6, 1/3, 1/2].
        dy = [0, 0, 1] (gradient only on class 2).
        dot = 0*(1/6) + 0*(1/3) + 1*(1/2) = 1/2
        dx0 = (1/6)*(0 - 1/2) = -1/12
        dx1 = (1/3)*(0 - 1/2) = -1/6
        dx2 = (1/2)*(1 - 1/2) = 1/4
        """
        net = _make_softmax_net((1, 3), axis=1)
        x = np.array([[0.0, np.log(2.0), np.log(3.0)]], dtype=np.float32)
        dy = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        expected_y = np.array([[1/6, 2/6, 3/6]], dtype=np.float32)
        np.testing.assert_allclose(y, expected_y, rtol=1e-5)
        expected_dx = np.array([[-1/12, -1/6, 1/4]], dtype=np.float32)
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-5)

    def test_one_hot_dy_on_correct_class(self):
        """Uniform y=[1/3,1/3,1/3], dy=[1,0,0]:
        dot = 1*(1/3) + 0 + 0 = 1/3
        dx0 = (1/3)*(1 - 1/3) = 2/9
        dx1 = (1/3)*(0 - 1/3) = -1/9
        dx2 = (1/3)*(0 - 1/3) = -1/9
        """
        net = _make_softmax_net((1, 3), axis=1)
        x = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        dy = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        np.testing.assert_allclose(y, np.array([[1/3, 1/3, 1/3]]), rtol=1e-5)
        expected_dx = np.array([[2/9, -1/9, -1/9]], dtype=np.float32)
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Numpy analytical gradient comparison (L2)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftmaxBackwardNumpy:
    """Compare caffe-ffi backward with numpy reference."""

    @pytest.mark.parametrize("shape,axis", [
        ((2, 3), 1),       # 2D, axis=1 (channel)
        ((1, 4), 1),       # 2D, single batch
        ((3, 5), 1),       # 2D, multi-batch
        ((2, 3, 4), 1),    # 3D NCL, axis=1
        ((1, 3, 2, 2), 1),  # 4D NCHW, axis=1
        ((2, 4, 3), 2),    # 3D, axis=2
    ])
    def test_softmax_vs_numpy(self, shape, axis):
        """Random data: caffe-ffi backward matches numpy reference."""
        np.random.seed(42 + shape[0] * 100 + axis)
        net = _make_softmax_net(shape, axis=axis)
        x = np.random.randn(*shape).astype(np.float32) * 2.0
        dy = np.random.randn(*shape).astype(np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        y_np = _softmax_forward_np(x, axis=axis)
        dx_np = _softmax_backward_np(y_np, dy, axis=axis)
        np.testing.assert_allclose(y, y_np, rtol=1e-5)
        np.testing.assert_allclose(dx, dx_np, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Numerical gradient (L3)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftmaxBackwardNumerical:
    """Central finite difference numerical gradient check."""

    @pytest.mark.parametrize("shape,axis", [
        ((1, 3), 1),
        ((2, 4), 1),
        ((2, 3, 2), 1),
        ((1, 2, 2, 2), 1),
    ])
    def test_numerical_grad(self, shape, axis):
        """Numerical gradient matches analytical gradient."""
        np.random.seed(123 + shape[0] * 50 + axis)
        net = _make_softmax_net(shape, axis=axis)
        x = np.random.randn(*shape).astype(np.float32) * 1.5
        dy = np.random.randn(*shape).astype(np.float32) * 0.5
        dx_analytic, y = _run_softmax_backward(net, x, dy)
        dx_num = _numerical_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_num, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Properties (L4)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftmaxBackwardProperties:
    """Property-based tests."""

    def test_zero_dy_gives_zero_gradients(self):
        """dy=0 -> dx=0."""
        net = _make_softmax_net((2, 3), axis=1)
        x = np.random.randn(2, 3).astype(np.float32)
        dy = np.zeros((2, 3), dtype=np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        np.testing.assert_allclose(dx, np.zeros_like(x), atol=1e-7)

    def test_gradient_shapes(self):
        """dx shape matches input shape."""
        net = _make_softmax_net((2, 3, 4), axis=1)
        x = np.random.randn(2, 3, 4).astype(np.float32)
        dy = np.random.randn(2, 3, 4).astype(np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        assert dx.shape == x.shape
        assert y.shape == x.shape

    def test_determinism(self):
        """Same input -> same gradients (deterministic)."""
        net = _make_softmax_net((2, 4), axis=1)
        x = np.random.randn(2, 4).astype(np.float32)
        dy = np.random.randn(2, 4).astype(np.float32)
        dx1, _ = _run_softmax_backward(net, x, dy)
        dx2, _ = _run_softmax_backward(net, x, dy)
        np.testing.assert_array_equal(dx1, dx2)

    def test_forward_preserved_after_backward(self):
        """Backward does not change Forward output probabilities."""
        net = _make_softmax_net((2, 3), axis=1)
        x = np.random.randn(2, 3).astype(np.float32)
        dy = np.random.randn(2, 3).astype(np.float32)
        out1 = net.forward({"data": x})
        y1 = out1["out"].copy()
        net.backward({"out": dy})
        y2 = net.blob_by_name("out").data
        np.testing.assert_array_equal(y1, y2)

    def test_finite_values(self):
        """Gradients are finite (non-NaN, non-Inf) for normal inputs."""
        net = _make_softmax_net((3, 5), axis=1)
        x = np.random.randn(3, 5).astype(np.float32) * 3.0
        dy = np.random.randn(3, 5).astype(np.float32)
        dx, y = _run_softmax_backward(net, x, dy)
        assert np.all(np.isfinite(dx))
        assert np.all(np.isfinite(y))

    def test_probability_sums_to_one(self):
        """Softmax output sums to 1 along axis."""
        net = _make_softmax_net((2, 4, 3), axis=1)
        x = np.random.randn(2, 4, 3).astype(np.float32) * 2.0
        dy = np.random.randn(2, 4, 3).astype(np.float32)
        _, y = _run_softmax_backward(net, x, dy)
        sums = np.sum(y, axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5)

    def test_gradient_sums_to_zero_per_position(self):
        """For softmax backward: sum_j(dx_j) = 0 per position.
        This holds because sum_j(dx_j) = sum_j(y_j*(dy_j - dot))
        = sum_j(y_j*dy_j) - dot*sum_j(y_j) = dot - dot*1 = 0.
        This is a fundamental property of the softmax Jacobian.
        """
        net = _make_softmax_net((2, 5), axis=1)
        x = np.random.randn(2, 5).astype(np.float32)
        dy = np.random.randn(2, 5).astype(np.float32)
        dx, _ = _run_softmax_backward(net, x, dy)
        dx_sums = np.sum(dx, axis=1)
        np.testing.assert_allclose(dx_sums, np.zeros_like(dx_sums), atol=1e-6)

    def test_gradient_when_dy_equals_y(self):
        """If dy = y (gradient proportional to probabilities):
        dot = sum(y_j * y_j) = ||y||^2
        dx = y * (y - ||y||^2) = y^2 - ||y||^2 * y
        For uniform y=1/C: ||y||^2 = C*(1/C^2) = 1/C
        dx = (1/C)*(1/C - 1/C) = 0 when y uniform? Let's check with numpy.
        """
        net = _make_softmax_net((1, 3), axis=1)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = net.forward({"data": x})
        y = out["out"]
        dy = y.copy()  # dy = y
        dx, _ = _run_softmax_backward(net, x, dy)
        dx_np = _softmax_backward_np(y, dy, axis=1)
        np.testing.assert_allclose(dx, dx_np, rtol=1e-5)
        # Also verify sum(dx) = 0 property
        np.testing.assert_allclose(np.sum(dx, axis=1), [0.0], atol=1e-6)
