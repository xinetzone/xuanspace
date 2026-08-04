"""InstanceNorm layer Forward/Backward gradient tests.

Covers the instance-normalization layer ``y = (x - mean) / sqrt(var + eps)``
applied per-(n, c) plane, optionally with per-channel affine ``* gamma + beta``:

  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX)
  2. Numerical gradient check (central finite differences) for dX
  3. Affine parameter gradients (dgamma / dbeta) incl. numerical check
  4. Known-value verification (single channel, mean/var normalization)
  5. affine=false vs affine=true configurations
  6. Shape/finite/determinism checks

InstanceNorm is smooth (no kink), so standard tolerance (rtol=1e-3) is used.

Mathematical reference (per (n, c) group of size M):
  mean = sum(x)/M;  var = sum((x-mean)^2)/M;  inv_std = 1/sqrt(var+eps)
  y = (x - mean) * inv_std * gamma + beta
  d_one = sum(dy*gamma)/M;  d_two = sum(dy*gamma*(x-mean))/M
  dx = inv_std * (dy*gamma - d_one - (x-mean)*inv_std^2*d_two)
  dgamma = sum(dy * (x-mean) * inv_std);  dbeta = sum(dy)
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import numerical_grad_for_blob

EPS = 1e-3       # central finite-difference step
NUM_RTOL = 1e-2  # float32 central-difference precision (matches _grad_check_utils)
NUM_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_instnorm_prototxt(N=1, C=1, H=4, W=5, eps=1e-5, affine=False):
    """Create a minimal Input(N,C,H,W) -> InstanceNorm prototxt."""
    return textwrap.dedent(f"""\
        name: "test_instnorm_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "instnorm"
          type: "InstanceNorm"
          bottom: "data"
          top: "out"
          instance_norm_param {{ eps: {eps} affine: {str(affine).lower()} }}
        }}
    """)


def _make_instnorm_net(N=1, C=1, H=4, W=5, eps=1e-5, affine=False):
    return Net(_make_instnorm_prototxt(N, C, H, W, eps=eps, affine=affine))


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _instnorm_forward_np(x, gamma=None, beta=None, eps=1e-5):
    """Forward reference: per-(n,c) mean/var normalization + optional affine."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    y = (x - mean) * inv_std
    if gamma is not None:
        y = y * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)
    return y


def _instnorm_backward_np(x, dy, gamma=None, eps=1e-5):
    """Backward reference: returns (dx, dgamma, dbeta)."""
    x = np.asarray(x, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    N, C, H, W = x.shape
    M = H * W
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)

    if gamma is None:
        gamma = np.ones(C)

    # dgamma, dbeta (per channel)
    dgamma = np.zeros(C)
    dbeta = np.zeros(C)
    yn = (x - mean) * inv_std  # normalized (before affine)
    for c in range(C):
        dgamma[c] = np.sum(dy[:, c, :, :] * yn[:, c, :, :])
        dbeta[c] = np.sum(dy[:, c, :, :])

    # dx: batchnorm-style per (n,c) group
    dy_g = dy * gamma.reshape(1, -1, 1, 1)
    sum_dy = np.sum(dy_g, axis=(2, 3), keepdims=True)
    sum_dy_x = np.sum(dy_g * (x - mean), axis=(2, 3), keepdims=True)
    d_one = sum_dy / M
    d_two = sum_dy_x / M
    dx = inv_std * (dy_g - d_one - (x - mean) * inv_std ** 2 * d_two)
    return dx, dgamma, dbeta


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
# Tests: InstanceNorm Forward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInstanceNormForward:
    """InstanceNorm forward correctness (known values + normalization)."""

    def test_forward_known_values(self):
        """Single (n,c) plane: mean/var normalization yields zero-mean, unit-var."""
        net = _make_instnorm_net(N=1, C=1, H=1, W=4, affine=False)
        # x = [1, 2, 3, 4]; mean=2.5; var=1.25; inv_std=1/sqrt(1.25+eps)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 1, 1, 4)
        out = net.forward({"data": x})["out"]
        out = np.asarray(out).reshape(-1)
        np.testing.assert_allclose(np.mean(out), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.var(out), 1.0, rtol=1e-4, atol=1e-4)

    def test_forward_zero_mean_unit_var(self):
        """Each (n,c) plane has zero mean and unit variance after normalization."""
        rng = np.random.RandomState(0)
        N, C, H, W = 2, 3, 4, 5
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=False)
        x = rng.randn(N, C, H, W).astype(np.float32) * 3.0 + 2.0
        out = np.asarray(net.forward({"data": x})["out"])
        for n in range(N):
            for c in range(C):
                plane = out[n, c]
                np.testing.assert_allclose(np.mean(plane), 0.0, atol=1e-5)
                np.testing.assert_allclose(np.var(plane), 1.0, rtol=1e-4, atol=1e-4)

    def test_forward_affine(self):
        """Affine=true: y = normalized * gamma + beta."""
        rng = np.random.RandomState(1)
        N, C, H, W = 2, 3, 4, 5
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=True)
        x = rng.randn(N, C, H, W).astype(np.float32)
        out = net.forward({"data": x})["out"]
        expected = _instnorm_forward_np(x, gamma=np.ones(C), beta=np.zeros(C)).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_forward_matches_numpy(self):
        """Forward matches numpy reference on random data."""
        rng = np.random.RandomState(2)
        N, C, H, W = 2, 3, 4, 5
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=False)
        x = rng.randn(N, C, H, W).astype(np.float32)
        out = net.forward({"data": x})["out"]
        expected = _instnorm_forward_np(x).astype(np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: InstanceNorm Backward (analytical)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInstanceNormGradient:
    """InstanceNorm backward analytical gradient tests."""

    def test_backward_analytical_no_affine(self):
        """dx matches numpy reference (affine=false)."""
        rng = np.random.RandomState(42)
        N, C, H, W = 2, 3, 4, 5
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=False)
        x = rng.randn(N, C, H, W).astype(np.float32) * 2.0
        dy = rng.randn(N, C, H, W).astype(np.float32)
        out = net.forward({"data": x})["out"]
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected, _, _ = _instnorm_backward_np(x, dy)
        np.testing.assert_allclose(dx, expected.astype(np.float32), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, _instnorm_forward_np(x).astype(np.float32),
                                   rtol=1e-5, atol=1e-6)

    def test_backward_analytical_affine(self):
        """dx matches numpy reference (affine=true, default gamma=1, beta=0)."""
        rng = np.random.RandomState(43)
        N, C, H, W = 2, 3, 4, 5
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=True)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected, _, _ = _instnorm_backward_np(x, dy, gamma=np.ones(C))
        np.testing.assert_allclose(dx, expected.astype(np.float32), rtol=1e-5, atol=1e-6)

    def test_backward_affine_gamma_beta(self):
        """dgamma and dbeta match numpy reference."""
        rng = np.random.RandomState(44)
        N, C, H, W = 2, 3, 4, 5
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=True)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        gamma = net.layer_by_name("instnorm").blobs[0].data.copy()
        beta = net.layer_by_name("instnorm").blobs[1].data.copy()
        _, dgamma_ref, dbeta_ref = _instnorm_backward_np(x, dy, gamma=gamma, eps=1e-5)
        dgamma = net.layer_by_name("instnorm").blobs[0].diff
        dbeta = net.layer_by_name("instnorm").blobs[1].diff
        np.testing.assert_allclose(dgamma, dgamma_ref.astype(np.float32), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dbeta, dbeta_ref.astype(np.float32), rtol=1e-5, atol=1e-6)

    def test_backward_runs_without_crash(self):
        """Backward should complete without errors on random data."""
        net = _make_instnorm_net(N=2, C=3, H=4, W=5, affine=True)
        x = np.random.RandomState(1).randn(2, 3, 4, 5).astype(np.float32)
        dy = np.ones_like(x)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == x.shape
        assert dx.dtype == np.float32
        assert np.all(np.isfinite(dx))


# ---------------------------------------------------------------------------
# Tests: InstanceNorm Numerical Gradient
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInstanceNormNumericalGradient:
    """Central finite-difference gradient checks (smooth, standard tolerance)."""

    def test_numerical_gradient_dx(self):
        """Numerical check for dX (affine=false)."""
        rng = np.random.RandomState(7)
        N, C, H, W = 1, 3, 3, 4
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=False)
        x = rng.randn(N, C, H, W).astype(np.float32) * 2.0
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad(net, x, dy)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_dgamma(self):
        """Numerical check for dgamma via parameter perturbation."""
        rng = np.random.RandomState(13)
        N, C, H, W = 1, 3, 3, 4
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=True)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dgamma_analytic = net.layer_by_name("instnorm").blobs[0].diff
        dgamma_num = numerical_grad_for_blob(
            net, "instnorm", 0, {"data": x}, "out", dy, h=EPS,
            name="instnorm.gamma",
        )
        np.testing.assert_allclose(dgamma_analytic, dgamma_num, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_dbeta(self):
        """Numerical check for dbeta via parameter perturbation."""
        rng = np.random.RandomState(21)
        N, C, H, W = 1, 3, 3, 4
        net = _make_instnorm_net(N=N, C=C, H=H, W=W, affine=True)
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})
        dbeta_analytic = net.layer_by_name("instnorm").blobs[1].diff
        dbeta_num = numerical_grad_for_blob(
            net, "instnorm", 1, {"data": x}, "out", dy, h=EPS,
            name="instnorm.beta",
        )
        np.testing.assert_allclose(dbeta_analytic, dbeta_num, rtol=NUM_RTOL, atol=NUM_ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])