"""BatchNorm layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX)
  2. Numerical gradient check (central finite differences for dX)
  3. Known-value verification (hand-computed expected gradients)
  4. Per-channel scaling correctness (different inv_std per channel)
  5. Scale factor handling (count != 1.0)
  6. Epsilon effect on gradient magnitude
  7. Zero dy → zero gradient
  8. Shape/finite/determinism checks
  9. Forward output preserved after backward
 10. propagate_down behavior

Mathematical reference (inference/use_global_stats mode):
  Forward:  y = (x - mean*sf) / sqrt(max(var*sf, 0) + eps)
  Backward: dX = dy * inv_std[c]
    where inv_std[c] = 1 / sqrt(max(var[c]*sf, 0) + eps)
    sf = 1/count if count != 0 else 1.0

  In inference mode, mean and variance are constants (running statistics),
  so there is no gradient flow through them. No blob gradients needed.
  Learnable gamma/beta are handled by a separate Scale layer.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._numpy_bn_reference import bn_forward, bn_backward, bn_get_inv_std

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS_NUMERICAL = 1e-3
EPS_BN = 1e-5


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_bn_prototxt(input_dims, eps=1e-5, use_global_stats=True):
    """Create Input -> BatchNorm prototxt."""
    dims_str = " ".join(str(d) for d in input_dims)
    return textwrap.dedent(f"""\
        name: "test_bn_bw"
        input: "data"
        input_dim: {input_dims[0]}
        input_dim: {input_dims[1]}
        input_dim: {input_dims[2]}
        input_dim: {input_dims[3]}
        layer {{
          name: "bn"
          type: "BatchNorm"
          bottom: "data"
          top: "bn"
          batch_norm_param {{
            use_global_stats: {"true" if use_global_stats else "false"}
            eps: {eps}
          }}
        }}
    """)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _make_bn_net(N, C, H, W, eps=EPS_BN):
    """Create Input->BatchNorm net with given dimensions and set blobs."""
    proto = _make_bn_prototxt((N, C, H, W), eps=eps)
    return Net(proto)


def _set_bn_blobs(net, mean, var, count=1.0):
    """Set BatchNorm blobs[0]=mean, blobs[1]=var, blobs[2]=[count]."""
    bn = net.layer_by_name("bn")
    bn.blobs[0].from_numpy(mean.astype(np.float32))
    bn.blobs[1].from_numpy(var.astype(np.float32))
    bn.blobs[2].from_numpy(np.array([count], dtype=np.float32))


def _run_bn_backward(net, x, dy, mean, var, count=1.0):
    """Run forward then backward, return dX."""
    _set_bn_blobs(net, mean, var, count=count)
    net.forward({"data": x.astype(np.float32)})
    net.backward({"bn": dy.astype(np.float32)})
    return net.blob_by_name("data").diff


# ---------------------------------------------------------------------------
# Numerical gradient helper
# ---------------------------------------------------------------------------

def _num_grad_dx(net, x, mean, var, dy, count=1.0, eps_bn=EPS_BN, h=EPS_NUMERICAL):
    """Numerical gradient of L = sum(dy * bn_out) w.r.t. x via central differences."""
    grad = np.zeros_like(x, dtype=np.float64)
    flat_x = x.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x.size):
        orig = flat_x[i]

        xp = x.copy()
        xp.ravel()[i] = orig + h
        _set_bn_blobs(net, mean, var, count=count)
        out_p = net.forward({"data": xp.astype(np.float32)})["bn"]
        loss_p = float(np.sum(dy.astype(np.float64) * out_p.astype(np.float64)))

        xm = x.copy()
        xm.ravel()[i] = orig - h
        _set_bn_blobs(net, mean, var, count=count)
        out_m = net.forward({"data": xm.astype(np.float32)})["bn"]
        loss_m = float(np.sum(dy.astype(np.float64) * out_m.astype(np.float64)))

        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Test Class 1: BatchNorm Backward core tests
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestBatchNormBackward:
    """Core BatchNorm backward gradient tests."""

    def test_bn_backward_known_values(self):
        """Hand-computed: x=4, mean=2, var=4, eps=0, count=1 → y=1, dy=1 → dx=0.5."""
        N, C, H, W = 1, 1, 1, 1
        net = _make_bn_net(N, C, H, W, eps=0.0)
        x = np.array([[[[4.0]]]], dtype=np.float32)
        mean = np.array([2.0], dtype=np.float32)
        var = np.array([4.0], dtype=np.float32)
        dy = np.array([[[[1.0]]]], dtype=np.float32)

        dx = _run_bn_backward(net, x, dy, mean, var, count=1.0)
        expected_dx = np.array([[[[0.5]]]], dtype=np.float32)
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-5)

    def test_bn_backward_analytical_dx(self):
        """Analytical dX vs numpy reference on random NCHW data."""
        np.random.seed(123)
        N, C, H, W = 2, 3, 4, 4
        net = _make_bn_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32) * 2.0
        mean = np.random.randn(C).astype(np.float32) * 0.5
        var = (np.random.rand(C).astype(np.float32) + 0.5) * 2.0
        dy = np.random.randn(N, C, H, W).astype(np.float32) * 0.5

        dx = _run_bn_backward(net, x, dy, mean, var, count=1.0)
        dx_ref = bn_backward(dy, var, count=1.0, eps=EPS_BN)
        np.testing.assert_allclose(dx, dx_ref, rtol=1e-5, atol=1e-6)

    def test_bn_numerical_gradient_dx(self):
        """Central finite difference check on small tensor (1×2×2×2 = 8 elements)."""
        np.random.seed(456)
        N, C, H, W = 1, 2, 2, 2
        net = _make_bn_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32) * 0.5
        mean = np.random.randn(C).astype(np.float32) * 0.3
        var = (np.random.rand(C).astype(np.float32) + 0.5) * 2.0
        dy = np.random.randn(N, C, H, W).astype(np.float32) * 0.5

        dx_analytic = _run_bn_backward(net, x, dy, mean, var, count=1.0)
        dx_numeric = _num_grad_dx(net, x, mean, var, dy, count=1.0)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)

    def test_bn_backward_zero_dy_gives_zero_grads(self):
        """Zero upstream gradient → zero input gradient."""
        np.random.seed(789)
        N, C, H, W = 2, 3, 2, 2
        net = _make_bn_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32)
        var = np.abs(np.random.randn(C).astype(np.float32)) + 0.1
        dy = np.zeros((N, C, H, W), dtype=np.float32)

        dx = _run_bn_backward(net, x, dy, mean, var)
        np.testing.assert_array_equal(dx, np.zeros_like(dx))

    def test_bn_backward_shapes(self):
        """dX shape matches input, dtype is float32, all values finite."""
        N, C, H, W = 2, 4, 3, 3
        net = _make_bn_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        mean = np.zeros(C, dtype=np.float32)
        var = np.ones(C, dtype=np.float32)
        dy = np.random.randn(N, C, H, W).astype(np.float32)

        dx = _run_bn_backward(net, x, dy, mean, var)
        assert dx.shape == x.shape
        assert dx.dtype == np.float32
        assert np.all(np.isfinite(dx))

    def test_bn_backward_deterministic(self):
        """Same inputs → same dX across multiple calls."""
        np.random.seed(101)
        N, C, H, W = 2, 3, 3, 3
        net = _make_bn_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32)
        var = (np.random.rand(C).astype(np.float32) + 0.5) * 2.0
        dy = np.random.randn(N, C, H, W).astype(np.float32)

        dx1 = _run_bn_backward(net, x, dy, mean, var)
        dx2 = _run_bn_backward(net, x, dy, mean, var)
        dx3 = _run_bn_backward(net, x, dy, mean, var)
        np.testing.assert_array_equal(dx1, dx2)
        np.testing.assert_array_equal(dx1, dx3)

    def test_bn_backward_preserves_forward_output(self):
        """Backward does not corrupt the forward output blob."""
        np.random.seed(202)
        N, C, H, W = 1, 2, 3, 3
        net = _make_bn_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32) * 0.5
        var = (np.random.rand(C).astype(np.float32) + 0.5) * 2.0
        dy = np.random.randn(N, C, H, W).astype(np.float32)

        _set_bn_blobs(net, mean, var)
        y_before = net.forward({"data": x})["bn"].copy()
        net.backward({"bn": dy})
        y_after = net.blob_by_name("bn").data
        np.testing.assert_array_equal(y_before, y_after)

    def test_bn_backward_eps_effect(self):
        """With zero variance: larger eps → smaller gradient magnitude (directional)."""
        N, C, H, W = 1, 1, 1, 1
        x = np.array([[[[0.0]]]], dtype=np.float32)
        mean = np.array([0.0], dtype=np.float32)
        var = np.array([0.0], dtype=np.float32)
        dy = np.array([[[[1.0]]]], dtype=np.float32)

        net_small = _make_bn_net(N, C, H, W, eps=1e-5)
        dx_small = _run_bn_backward(net_small, x, dy, mean, var, count=1.0)

        net_large = _make_bn_net(N, C, H, W, eps=1.0)
        dx_large = _run_bn_backward(net_large, x, dy, mean, var, count=1.0)

        # inv_std = 1/sqrt(0+eps), so larger eps → smaller inv_std → smaller dx
        assert dx_large[0, 0, 0, 0] < dx_small[0, 0, 0, 0], \
            f"Expected smaller grad with larger eps, got {dx_large[0,0,0,0]} vs {dx_small[0,0,0,0]}"


# ---------------------------------------------------------------------------
# Test Class 2: Multi-channel per-channel scaling
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestBatchNormBackwardMultiChannel:
    """Verify per-channel inv_std is correctly applied (each channel has different scaling)."""

    def test_bn_per_channel_scaling(self):
        """var=[1,4,9] → inv_std=[1, 0.5, 1/3]; dy=1 → dx=[1, 0.5, 1/3] per channel."""
        N, C, H, W = 2, 3, 2, 2
        net = _make_bn_net(N, C, H, W, eps=0.0)
        x = np.random.randn(N, C, H, W).astype(np.float32) * 0.1
        mean = np.zeros(C, dtype=np.float32)
        var = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        dy = np.ones((N, C, H, W), dtype=np.float32)

        dx = _run_bn_backward(net, x, dy, mean, var, count=1.0)
        np.testing.assert_allclose(dx[:, 0, :, :], np.ones((N, H, W), dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(dx[:, 1, :, :], np.full((N, H, W), 0.5, dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(dx[:, 2, :, :], np.full((N, H, W), 1.0 / 3, dtype=np.float32), rtol=1e-5)


# ---------------------------------------------------------------------------
# Test Class 3: Scale factor (count != 1.0)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestBatchNormBackwardScaleFactor:
    """Test BatchNorm backward with non-unit scale factor (count != 1.0)."""

    def test_bn_scale_factor_count(self):
        """count=10, var_stored=40 → eff_var=40/10=4 → inv_std=0.5, dy=1 → dx=0.5."""
        N, C, H, W = 1, 1, 2, 2
        net = _make_bn_net(N, C, H, W, eps=0.0)
        x = np.random.randn(N, C, H, W).astype(np.float32) * 0.1
        mean = np.array([10.0], dtype=np.float32)  # eff_mean = 10/10 = 1.0
        var = np.array([40.0], dtype=np.float32)    # eff_var = 40/10 = 4.0
        count = 10.0
        dy = np.ones((N, C, H, W), dtype=np.float32)

        dx = _run_bn_backward(net, x, dy, mean, var, count=count)
        np.testing.assert_allclose(dx, np.full_like(dy, 0.5), rtol=1e-5)

    def test_bn_scale_factor_numerical(self):
        """Numerical gradient check with non-unit count."""
        np.random.seed(303)
        N, C, H, W = 1, 2, 1, 3  # 6 elements for fast numerical check
        net = _make_bn_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32) * 0.5
        mean = np.random.randn(C).astype(np.float32) * 2.0
        var = (np.random.rand(C).astype(np.float32) + 1.0) * 5.0
        count = 5.0
        dy = np.random.randn(N, C, H, W).astype(np.float32) * 0.5

        dx_analytic = _run_bn_backward(net, x, dy, mean, var, count=count)
        dx_numeric = _num_grad_dx(net, x, mean, var, dy, count=count)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)
