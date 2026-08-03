"""Bias layer Backward gradient tests.

Bias computes y = x + bias (broadcast addition along specified axis).
Gradients:
  dX[n,d,i]  = dy[n,d,i]           (gradient passes through directly)
  d_bias[d]  = sum over n,i of dy[n,d,i]

Covers:
  1. Known-value hand verification
  2. Analytical gradient (numpy reference vs caffe-ffi)
  3. Numerical gradient check (central finite differences for dX, d_bias)
  4. Multiple shapes (2D NxD, 4D NxCxHxW)
  5. Zero dy -> zero gradients
  6. Shape/finite/determinism checks
  7. Multi-axis bias (num_axes > 1, e.g. positional encoding)
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
# Numpy reference for Bias forward/backward
# ---------------------------------------------------------------------------

def bias_forward_np(x, bias, axis=1, num_axes=1):
    """Numpy reference: y = x + bias with broadcasting along axis."""
    x64 = x.astype(np.float64)
    ndim = x64.ndim
    out_shape = [1] * ndim
    for i in range(num_axes):
        out_shape[axis + i] = x64.shape[axis + i]
    b64 = bias.reshape(out_shape).astype(np.float64)
    return (x64 + b64).astype(np.float32)


def bias_backward_np(x, dy, axis=1, num_axes=1, need_dx=True, need_dbias=True):
    """Numpy reference for Bias backward."""
    dy64 = dy.astype(np.float64)
    ndim = dy64.ndim

    dX = None
    if need_dx:
        dX = dy64.astype(np.float32)

    # Determine reduction axes (all except the bias axes)
    bias_axes = tuple(range(axis, axis + num_axes))
    reduce_axes = tuple(i for i in range(ndim) if i not in bias_axes)

    dbias = None
    if need_dbias:
        dbias = dy64.sum(axis=reduce_axes).astype(np.float32)

    return dX, dbias


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_bias_prototxt(input_dims, axis=1, num_axes=1, filler_value=0.0):
    """Create Input -> Bias prototxt."""
    dims_lines = "\n".join(f"input_dim: {d}" for d in input_dims)
    return textwrap.dedent(f"""\
        name: "test_bias_bw"
        input: "data"
        {dims_lines}
        layer {{
          name: "bias"
          type: "Bias"
          bottom: "data"
          top: "bias_out"
          bias_param {{
            axis: {axis}
            num_axes: {num_axes}
            filler {{ type: "constant" value: {filler_value} }}
          }}
        }}
    """)


def _make_bias_net(input_dims, axis=1, num_axes=1, filler_value=0.0):
    proto = _make_bias_prototxt(input_dims, axis, num_axes, filler_value)
    return Net(proto)


# ---------------------------------------------------------------------------
# Helper: set bias blobs and run backward
# ---------------------------------------------------------------------------

def _set_bias_blob(net, bias):
    """Set bias layer blobs[0] = bias (preserve shape for multi-axis bias)."""
    bias_layer = net.layer_by_name("bias")
    bias_layer.blobs[0].from_numpy(bias.astype(np.float32))


def _run_bias_backward(net, x, dy, bias):
    """Run forward then backward, return (dX, d_bias)."""
    _set_bias_blob(net, bias)
    net.forward({"data": x.astype(np.float32)})
    net.backward({"bias_out": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    d_bias = net.layer_by_name("bias").blobs[0].diff
    return dX, d_bias


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestBiasBackwardKnownValues:
    """L1: Hand-computed known-value verification."""

    def test_forward_zero_bias(self):
        """Forward with bias=0: y == x."""
        N, D = 2, 4
        net = _make_bias_net((N, D), axis=1, num_axes=1, filler_value=0.0)
        rng = np.random.RandomState(42)
        x = rng.randn(N, D).astype(np.float32)
        out = net.forward({"data": x})["bias_out"]
        np.testing.assert_array_equal(out, x)

    def test_forward_known_bias(self):
        """Forward with known bias: y = x + b per-channel."""
        N, C, H, W = 1, 2, 1, 1
        net = _make_bias_net((N, C, H, W), axis=1, filler_value=0.0)
        bias = np.array([3.0, -1.0], dtype=np.float32)
        _set_bias_blob(net, bias)
        x = np.array([[[[1.0]], [[2.0]]]], dtype=np.float32)
        out = net.forward({"data": x})["bias_out"]
        expected = np.array([[[[4.0]], [[1.0]]]], dtype=np.float32)  # 1+3=4, 2+(-1)=1
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_backward_dx_passes_through(self):
        """Backward dX = dy (bias is addition, gradient identity)."""
        N, C, H, W = 1, 2, 1, 1
        net = _make_bias_net((N, C, H, W), axis=1, filler_value=0.0)
        bias = np.array([3.0, -1.0], dtype=np.float32)
        x = np.array([[[[1.0]], [[2.0]]]], dtype=np.float32)
        dy = np.array([[[[5.0]], [[-3.0]]]], dtype=np.float32)
        dX, _ = _run_bias_backward(net, x, dy, bias)
        np.testing.assert_array_equal(dX, dy)

    def test_backward_dbias_known_values(self):
        """Backward d_bias = sum(dy) over broadcast dims."""
        N, D = 2, 3
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        bias = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        x = np.zeros((N, D), dtype=np.float32)
        dy = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], dtype=np.float32)
        _, d_bias = _run_bias_backward(net, x, dy, bias)
        expected = dy.sum(axis=0)
        np.testing.assert_allclose(d_bias, expected, rtol=1e-5)


@require_cpp_extension
class TestBiasBackwardAnalytical:
    """L2: Analytical gradient comparison with numpy reference."""

    @pytest.mark.parametrize("N,D", [(2, 4), (4, 8), (1, 16)])
    def test_dx_vs_numpy(self, N, D):
        """dX = dy matches numpy reference (should be exact)."""
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(N * 100 + D)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        bias = rng.randn(D).astype(np.float32) * 0.5
        dX, _ = _run_bias_backward(net, x, dy, bias)
        dX_ref, _ = bias_backward_np(x, dy, axis=1)
        np.testing.assert_allclose(dX, dX_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("N,D", [(2, 4), (4, 8)])
    def test_dbias_vs_numpy(self, N, D):
        """d_bias matches numpy reference."""
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(N * 100 + D + 1)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        bias = rng.randn(D).astype(np.float32) * 0.5
        _, d_bias = _run_bias_backward(net, x, dy, bias)
        _, d_bias_ref = bias_backward_np(x, dy, axis=1, need_dx=False)
        np.testing.assert_allclose(d_bias, d_bias_ref, rtol=RTOL, atol=ATOL)

    def test_4d_spatial_shape(self):
        """4D (N,C,H,W) shape with per-channel bias."""
        N, C, H, W = 2, 3, 4, 4
        net = _make_bias_net((N, C, H, W), axis=1, filler_value=0.0)
        rng = np.random.RandomState(123)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        bias = rng.randn(C).astype(np.float32) * 0.5
        dX, d_bias = _run_bias_backward(net, x, dy, bias)
        dX_ref, d_bias_ref = bias_backward_np(x, dy, axis=1)
        np.testing.assert_allclose(dX, dX_ref, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(d_bias, d_bias_ref, rtol=RTOL, atol=ATOL)

    def test_multi_axis_bias(self):
        """num_axes=2 (e.g. positional encoding bias shape (T,D) on axis=1)."""
        N, T, D = 2, 3, 4
        net = _make_bias_net((N, T, D), axis=1, num_axes=2, filler_value=0.0)
        rng = np.random.RandomState(456)
        x = rng.randn(N, T, D).astype(np.float32)
        dy = rng.randn(N, T, D).astype(np.float32)
        bias = rng.randn(T, D).astype(np.float32) * 0.3
        dX, d_bias = _run_bias_backward(net, x, dy, bias)
        dX_ref, d_bias_ref = bias_backward_np(x, dy, axis=1, num_axes=2)
        np.testing.assert_allclose(dX, dX_ref, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(d_bias.ravel(), d_bias_ref.ravel(), rtol=RTOL, atol=ATOL)


@require_cpp_extension
class TestBiasBackwardNumerical:
    """L3: Numerical gradient via central finite differences."""

    def test_numerical_grad_dx(self):
        """Numerical gradient for dX matches analytical."""
        N, D = 2, 4
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(42)
        x = rng.randn(N, D).astype(np.float32) * 0.5
        bias = rng.randn(D).astype(np.float32) * 0.3
        dy = rng.randn(N, D).astype(np.float32)

        _set_bias_blob(net, bias)
        net.forward({"data": x})
        net.backward({"bias_out": dy})
        analytic_dx = net.blob_by_name("data").diff.copy()

        num_dx = numerical_grad_for_input(
            net, "data", x, "bias_out", dy, h=EPS, name="bias_dX",
        )
        assert_grad_close(analytic_dx, num_dx, name="bias_dX", rtol=RTOL, atol=ATOL*10)

    def test_numerical_grad_dbias(self):
        """Numerical gradient for d_bias matches analytical (perturb bias blob)."""
        N, D = 2, 4
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(43)
        x = rng.randn(N, D).astype(np.float32) * 0.5
        bias = rng.randn(D).astype(np.float32) * 0.3
        dy = rng.randn(N, D).astype(np.float32)

        _set_bias_blob(net, bias)
        net.forward({"data": x})
        net.backward({"bias_out": dy})
        analytic_db = net.layer_by_name("bias").blobs[0].diff.copy()

        num_db = numerical_grad_for_blob(
            net, "bias", 0, {"data": x}, "bias_out", dy, h=EPS, name="bias_dBias",
        )
        assert_grad_close(analytic_db, num_db, name="bias_dBias", rtol=RTOL, atol=ATOL*10)


@require_cpp_extension
class TestBiasBackwardProperties:
    """Property-based tests: zero gradients, shapes, determinism, forward preserved."""

    def test_zero_dy_gives_zero_gradients(self):
        """Zero dy produces zero dX and d_bias."""
        N, D = 2, 4
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(50)
        x = rng.randn(N, D).astype(np.float32)
        dy = np.zeros_like(x)
        bias = rng.randn(D).astype(np.float32) * 0.5
        dX, d_bias = _run_bias_backward(net, x, dy, bias)
        np.testing.assert_allclose(dX, np.zeros_like(x), atol=1e-7)
        np.testing.assert_allclose(d_bias, np.zeros(D), atol=1e-7)

    def test_gradient_shapes(self):
        """Gradient shapes match blob/param shapes."""
        N, C, H, W = 2, 3, 4, 5
        net = _make_bias_net((N, C, H, W), axis=1, filler_value=0.0)
        rng = np.random.RandomState(51)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        bias = rng.randn(C).astype(np.float32)
        dX, d_bias = _run_bias_backward(net, x, dy, bias)
        assert dX.shape == (N, C, H, W)
        assert d_bias.shape == (C,)

    def test_determinism(self):
        """Running backward twice with same inputs gives same gradients."""
        N, D = 2, 4
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(52)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        bias = rng.randn(D).astype(np.float32) * 0.5
        dX1, db1 = _run_bias_backward(net, x, dy, bias)
        dX2, db2 = _run_bias_backward(net, x, dy, bias)
        np.testing.assert_array_equal(dX1, dX2)
        np.testing.assert_array_equal(db1, db2)

    def test_forward_preserved_after_backward(self):
        """Forward output is unchanged after backward."""
        N, D = 2, 4
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(53)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        bias = rng.randn(D).astype(np.float32) * 0.5
        _set_bias_blob(net, bias)
        out1 = net.forward({"data": x})["bias_out"].copy()
        net.backward({"bias_out": dy})
        out2 = net.blob_by_name("bias_out").data.copy()
        np.testing.assert_array_equal(out1, out2)

    def test_dx_exact_copy_of_dy(self):
        """dX is an exact copy of dy (bias is pure addition, ∂y/∂x=1)."""
        N, C, H, W = 3, 4, 5, 6
        net = _make_bias_net((N, C, H, W), axis=1, filler_value=0.0)
        rng = np.random.RandomState(54)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        bias = rng.randn(C).astype(np.float32) * 0.5
        dX, _ = _run_bias_backward(net, x, dy, bias)
        np.testing.assert_array_equal(dX, dy)

    def test_finite_values(self):
        """All gradients are finite (no NaN/Inf)."""
        N, D = 4, 8
        net = _make_bias_net((N, D), axis=1, filler_value=0.0)
        rng = np.random.RandomState(55)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        bias = rng.randn(D).astype(np.float32) * 0.5
        dX, d_bias = _run_bias_backward(net, x, dy, bias)
        assert np.all(np.isfinite(dX))
        assert np.all(np.isfinite(d_bias))
