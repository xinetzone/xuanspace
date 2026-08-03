"""Dropout layer Backward gradient tests (inference mode).

In the current caffe-ffi implementation Dropout operates in inference mode:
  - Forward: y = x  (identity pass-through)
  - Backward: dx = dy (identity, since dy/dx = 1)

Note: Training-mode Dropout (random neuron dropping with inverted scaling) is
not implemented; only inference identity is supported, which is sufficient for
inference/evaluation and simple layer-composition tests.

Covers:
  1. Known-value verification (identity forward + backward)
  2. Analytical gradient (dx == dy exactly, no scaling)
  3. Numerical gradient check (central finite differences via _grad_check_utils)
  4. Multiple dropout_ratio values (0.0, 0.3, 0.5, 0.7)
  5. Multiple input shapes (2D, 4D)
  6. Zero dy -> zero dX
  7. Shape/dtype/finite/determinism checks
  8. Forward output preserved after backward

Dropout has NO learnable parameters (no weight/bias blobs), so only dX needs
verification.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import (
    assert_grad_close,
    numerical_grad_for_input,
)

EPS_NUMERICAL = 1e-3


# ---------------------------------------------------------------------------
# Numpy reference for Dropout backward (inference identity)
# ---------------------------------------------------------------------------

def dropout_backward_np(dy, dropout_ratio=0.5):
    """Numpy reference for inference-mode Dropout backward.

    In inference mode Dropout is identity: dx = dy (dropout_ratio is ignored).
    """
    return dy.astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_dropout_prototxt(input_dims, dropout_ratio=0.5):
    """Create Input -> Dropout prototxt."""
    dims_lines = "\n".join(f"input_dim: {d}" for d in input_dims)
    return textwrap.dedent(f"""\
        name: "test_dropout_bw"
        input: "data"
        {dims_lines}
        layer {{
          name: "drop"
          type: "Dropout"
          bottom: "data"
          top: "drop"
          dropout_param {{
            dropout_ratio: {dropout_ratio}
          }}
        }}
    """)


def _make_dropout_net(input_dims, dropout_ratio=0.5):
    proto = _make_dropout_prototxt(input_dims, dropout_ratio)
    return Net(proto)


def _run_dropout_backward(net, x, dy):
    """Run forward then backward, return (y, dX)."""
    out = net.forward({"data": x.astype(np.float32)})
    net.backward({"drop": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    return out["drop"], dX


# ---------------------------------------------------------------------------
# Test Class 1: Identity forward/backward known values
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestDropoutIdentity:
    """Inference Dropout is exact identity for both forward and backward."""

    @pytest.mark.parametrize("ratio", [0.0, 0.3, 0.5, 0.7])
    def test_forward_is_identity(self, ratio):
        """Forward output must exactly equal input (y = x)."""
        N, D = 2, 8
        net = _make_dropout_net((N, D), dropout_ratio=ratio)
        rng = np.random.RandomState(42)
        x = rng.randn(N, D).astype(np.float32)
        out = net.forward({"data": x})
        np.testing.assert_array_equal(out["drop"], x)

    @pytest.mark.parametrize("ratio", [0.0, 0.3, 0.5, 0.7])
    def test_backward_dx_equals_dy(self, ratio):
        """Backward dX must exactly equal dy (dx = dy)."""
        N, D = 2, 8
        net = _make_dropout_net((N, D), dropout_ratio=ratio)
        rng = np.random.RandomState(43)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        _, dX = _run_dropout_backward(net, x, dy)
        np.testing.assert_array_equal(dX, dy)

    def test_known_values_small(self):
        """Tiny 2-element tensor: hand-verified identity."""
        net = _make_dropout_net((1, 2), dropout_ratio=0.5)
        x = np.array([[3.0, -7.0]], dtype=np.float32)
        dy = np.array([[0.1, -0.2]], dtype=np.float32)
        y, dX = _run_dropout_backward(net, x, dy)
        np.testing.assert_array_equal(y, x)
        np.testing.assert_array_equal(dX, dy)


# ---------------------------------------------------------------------------
# Test Class 2: 4D tensor (NCHW) backward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestDropout4DBackward:
    """Dropout backward on 4D NCHW tensors (typical convnet use)."""

    def test_4d_analytical_dx(self):
        """4D NCHW: caffe-ffi dX == dy exactly."""
        rng = np.random.RandomState(44)
        N, C, H, W = 2, 3, 4, 4
        net = _make_dropout_net((N, C, H, W), dropout_ratio=0.5)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        _, dX = _run_dropout_backward(net, x, dy)
        expected_dx = dropout_backward_np(dy, dropout_ratio=0.5)
        np.testing.assert_array_equal(dX, expected_dx)

    def test_4d_numerical_dx(self):
        """4D NCHW: numerical gradient check."""
        rng = np.random.RandomState(45)
        N, C, H, W = 1, 2, 3, 3
        net = _make_dropout_net((N, C, H, W), dropout_ratio=0.5)
        x = rng.randn(N, C, H, W).astype(np.float32) * 0.5
        dy = rng.randn(N, C, H, W).astype(np.float32) * 0.1

        net.forward({"data": x})
        net.backward({"drop": dy})
        analytic_dX = net.blob_by_name("data").diff

        numerical_dX = numerical_grad_for_input(
            net, "data", x, "drop", dy, h=EPS_NUMERICAL,
            name="dropout_4d_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, numerical_dX, name="dX(Dropout 4D)",
                          rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 3: Numerical gradient for 2D (InnerProduct-style)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestDropout2DNumerical:
    """Numerical gradient check on 2D (N,D) tensors."""

    @pytest.mark.parametrize("ratio", [0.0, 0.5])
    def test_2d_numerical_dx(self, ratio):
        """2D: numerical gradient for ratio=0.0 and ratio=0.5."""
        rng = np.random.RandomState(100 + int(ratio * 10))
        N, D = 2, 6
        net = _make_dropout_net((N, D), dropout_ratio=ratio)
        x = rng.randn(N, D).astype(np.float32) * 0.5
        dy = rng.randn(N, D).astype(np.float32) * 0.1

        net.forward({"data": x})
        net.backward({"drop": dy})
        analytic_dX = net.blob_by_name("data").diff

        numerical_dX = numerical_grad_for_input(
            net, "data", x, "drop", dy, h=EPS_NUMERICAL,
            name=f"dropout_2d_r{ratio}_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, numerical_dX, name=f"dX(Dropout 2D r={ratio})",
                          rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 4: Edge cases
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestDropoutEdgeCases:
    """Zero gradients, determinism, shape/dtype/finite checks."""

    def test_zero_dy_zero_dx(self):
        """Zero dy must produce zero dX."""
        N, D = 2, 8
        net = _make_dropout_net((N, D), dropout_ratio=0.5)
        x = np.random.RandomState(0).randn(N, D).astype(np.float32)
        dy = np.zeros((N, D), dtype=np.float32)
        _, dX = _run_dropout_backward(net, x, dy)
        np.testing.assert_array_equal(dX, np.zeros_like(dX))

    def test_deterministic(self):
        """Same input -> same dX (deterministic)."""
        rng = np.random.RandomState(99)
        N, D = 2, 8
        net1 = _make_dropout_net((N, D), dropout_ratio=0.5)
        net2 = _make_dropout_net((N, D), dropout_ratio=0.5)
        x = rng.randn(N, D).astype(np.float32)
        dy = rng.randn(N, D).astype(np.float32)
        _, dX1 = _run_dropout_backward(net1, x, dy)
        _, dX2 = _run_dropout_backward(net2, x, dy)
        np.testing.assert_array_equal(dX1, dX2)

    @pytest.mark.parametrize("shape", [(1, 10), (2, 3, 4), (2, 3, 4, 5)])
    def test_dx_shape_dtype(self, shape):
        """dX has correct shape, dtype, and finite values."""
        net = _make_dropout_net(shape, dropout_ratio=0.5)
        x = np.random.RandomState(0).randn(*shape).astype(np.float32)
        dy = np.random.RandomState(1).randn(*shape).astype(np.float32)
        _, dX = _run_dropout_backward(net, x, dy)
        assert dX.shape == shape, f"wrong dX shape: {dX.shape} vs {shape}"
        assert dX.dtype == np.float32, f"wrong dtype: {dX.dtype}"
        assert np.all(np.isfinite(dX)), "non-finite values in dX"

    def test_forward_preserved_after_backward(self):
        """Forward output blob data is unchanged after Backward."""
        N, D = 2, 8
        net = _make_dropout_net((N, D), dropout_ratio=0.5)
        x = np.random.RandomState(50).randn(N, D).astype(np.float32)
        out = net.forward({"data": x})
        y_before = out["drop"].copy()
        dy = np.random.RandomState(51).randn(N, D).astype(np.float32)
        net.backward({"drop": dy})
        y_after = net.blob_by_name("drop").data
        np.testing.assert_array_equal(y_before, y_after)

    def test_inplace_safe(self):
        """Dropout works correctly when top == bottom (inplace operation)."""
        # Create net with inplace Dropout (top == bottom name)
        proto = textwrap.dedent("""\
            name: "test_dropout_inplace"
            input: "data"
            input_dim: 1
            input_dim: 4
            layer {
              name: "drop"
              type: "Dropout"
              bottom: "data"
              top: "data"
              dropout_param { dropout_ratio: 0.5 }
            }
        """)
        net = Net(proto)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        dy = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        out = net.forward({"data": x})
        # Forward inplace: output replaces input
        np.testing.assert_array_equal(out["data"], x)
        net.backward({"data": dy})
        dX = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dX, dy)
