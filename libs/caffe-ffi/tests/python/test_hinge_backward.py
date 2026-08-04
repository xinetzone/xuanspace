"""HingeLayer Forward/Backward gradient tests.

Covers the multi-class hinge loss:

  For each sample (i, j) with truth label y:
    z_c = max(0, 1 + score_c - score_y)   for all c != y
    loss_i = sum_c z_c            (L1 norm)
    loss_i = sum_c z_c^2          (L2 norm)
  scalar loss = mean_i loss_i

Tests:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX)
  2. Numerical gradient check (central finite differences)
  3. Known-value verification (single-sample, hand-computed)
  4. L1 vs L2 norm configurations
  5. loss_weight scaling (via upstream top diff)
  6. Label bottom never receives gradients; shape/finite checks

Mathematical reference (N samples, loss_weight L):
  dX_c    += L * (1/N) * [ z_c > 0 ? 1 : 0 ]        (L1)
  dX_c    += L * (1/N) * [ z_c > 0 ? 2*z_c : 0 ]    (L2)
  dX_y    -=  sum_{c != y} dX_c
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

EPS = 1e-3       # central finite-difference step
NUM_RTOL = 1e-2  # float32 central-difference precision (matches _grad_check_utils)
NUM_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_hinge_prototxt(N=1, C=3, H=1, W=1, norm="L1", axis=1):
    """Create Input(score) -> Input(label) -> Hinge prototxt."""
    return textwrap.dedent(f"""\
        name: "test_hinge_bw"
        layer {{
          name: "score"
          type: "Input"
          top: "score"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "label"
          type: "Input"
          top: "label"
          input_param {{ shape {{ dim: {N} dim: 1 dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "hinge"
          type: "Hinge"
          bottom: "score"
          bottom: "label"
          top: "loss"
          hinge_param {{ norm: {norm} axis: {axis} }}
        }}
    """)


def _make_hinge_net(N=1, C=3, H=1, W=1, norm="L1", axis=1):
    return Net(_make_hinge_prototxt(N, C, H, W, norm=norm, axis=axis))


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _hinge_forward_np(score, label, norm="L1"):
    """Forward reference: mean over samples of sum_c z_c (or z_c^2)."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    N, C, H, W = score.shape
    samples = N * H * W
    total = 0.0
    for i in range(N):
        for j in range(H * W):
            y = int(label[i, 0, j // W, j % W])
            z = relu(1.0 + score[i, :, j // W, j % W] - score[i, y, j // W, j % W])
            z[y] = 0.0  # exclude truth class
            if norm == "L2":
                total += np.sum(z ** 2)
            else:
                total += np.sum(z)
    return total / samples


def relu(x):
    return np.maximum(x, 0.0)


def _hinge_backward_np(score, label, norm="L1", loss_weight=1.0):
    """Backward reference: returns dX."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    N, C, H, W = score.shape
    samples = N * H * W
    scale = loss_weight / samples
    dx = np.zeros_like(score, dtype=np.float64)
    for i in range(N):
        for j in range(H * W):
            y = int(label[i, 0, j // W, j % W])
            for c in range(C):
                if c == y:
                    continue
                z = 1.0 + score[i, c, j // W, j % W] - score[i, y, j // W, j % W]
                if z > 0.0:
                    dz = scale * (2.0 * z if norm == "L2" else 1.0)
                    dx[i, c, j // W, j % W] += dz
                    dx[i, y, j // W, j % W] -= dz
    return dx


def _num_grad(net, score, label, loss_weight=1.0, h=EPS):
    """Numerical gradient of loss_weight * loss w.r.t. score (central differences)."""
    grad = np.zeros_like(score, dtype=np.float64)
    flat_score = score.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_score.size):
        orig = flat_score[i]
        sp = score.copy()
        sp.ravel()[i] = orig + h
        out_p = net.forward({"score": sp.astype(np.float32), "label": label})
        loss_p = float(out_p["loss"].flat[0] * loss_weight)
        sm = score.copy()
        sm.ravel()[i] = orig - h
        out_m = net.forward({"score": sm.astype(np.float32), "label": label})
        loss_m = float(out_m["loss"].flat[0] * loss_weight)
        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# Tests: Hinge Forward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestHingeForward:
    """Hinge forward correctness (known values)."""

    def test_forward_known_values_l1(self):
        """Single sample, L1: hand-computed."""
        net = _make_hinge_net(N=1, C=3, H=1, W=1, norm="L1")
        # score = [3, 1, 0]; truth = 0
        # c=1: z = 1 + 1 - 3 = -1 -> 0
        # c=2: z = 1 + 0 - 3 = -2 -> 0
        # loss = 0
        score = np.array([3.0, 1.0, 0.0], dtype=np.float32).reshape(1, 3, 1, 1)
        label = np.array([0], dtype=np.float32).reshape(1, 1, 1, 1)
        out = net.forward({"score": score, "label": label})["loss"]
        np.testing.assert_allclose(float(out.flat[0]), 0.0, atol=1e-6)

    def test_forward_known_values_l1_violation(self):
        """Single sample, L1 with a violation: hand-computed."""
        net = _make_hinge_net(N=1, C=3, H=1, W=1, norm="L1")
        # score = [0, 3, 0]; truth = 0
        # c=1: z = 1 + 3 - 0 = 4 -> 4
        # c=2: z = 1 + 0 - 0 = 1 -> 1
        # loss = 4 + 1 = 5
        score = np.array([0.0, 3.0, 0.0], dtype=np.float32).reshape(1, 3, 1, 1)
        label = np.array([0], dtype=np.float32).reshape(1, 1, 1, 1)
        out = net.forward({"score": score, "label": label})["loss"]
        np.testing.assert_allclose(float(out.flat[0]), 5.0, rtol=1e-6)

    def test_forward_known_values_l2(self):
        """Single sample, L2: hand-computed."""
        net = _make_hinge_net(N=1, C=3, H=1, W=1, norm="L2")
        # score = [0, 3, 0]; truth = 0
        # c=1: z = 4 -> 16
        # c=2: z = 1 -> 1
        # loss = 16 + 1 = 17
        score = np.array([0.0, 3.0, 0.0], dtype=np.float32).reshape(1, 3, 1, 1)
        label = np.array([0], dtype=np.float32).reshape(1, 1, 1, 1)
        out = net.forward({"score": score, "label": label})["loss"]
        np.testing.assert_allclose(float(out.flat[0]), 17.0, rtol=1e-6)

    def test_forward_matches_numpy(self):
        """Forward matches numpy reference on random data (L1 and L2)."""
        rng = np.random.RandomState(0)
        N, C, H, W = 2, 4, 2, 3
        for norm in ("L1", "L2"):
            net = _make_hinge_net(N=N, C=C, H=H, W=W, norm=norm)
            score = rng.randn(N, C, H, W).astype(np.float32)
            label = rng.randint(0, C, size=(N, 1, H, W)).astype(np.float32)
            out = net.forward({"score": score, "label": label})["loss"]
            expected = _hinge_forward_np(score, label, norm=norm)
            np.testing.assert_allclose(float(out.flat[0]), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Hinge Backward (analytical)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestHingeGradient:
    """Hinge backward analytical gradient tests."""

    def _run(self, net, score, label, loss_weight=1.0):
        out = net.forward({"score": score, "label": label})
        loss = float(out["loss"].flat[0])
        net.backward({"loss": np.array([loss_weight], dtype=np.float32)})
        dx = net.blob_by_name("score").diff
        return loss, dx

    def test_backward_analytical_l1(self):
        """dx matches numpy reference (L1)."""
        rng = np.random.RandomState(42)
        N, C, H, W = 2, 4, 2, 3
        net = _make_hinge_net(N=N, C=C, H=H, W=W, norm="L1")
        score = rng.randn(N, C, H, W).astype(np.float32)
        label = rng.randint(0, C, size=(N, 1, H, W)).astype(np.float32)
        loss, dx = self._run(net, score, label)
        dx_ref = _hinge_backward_np(score, label, norm="L1")
        np.testing.assert_allclose(dx, dx_ref.astype(np.float32), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(loss, _hinge_forward_np(score, label, "L1"), rtol=1e-5)

    def test_backward_analytical_l2(self):
        """dx matches numpy reference (L2)."""
        rng = np.random.RandomState(43)
        N, C, H, W = 2, 4, 2, 3
        net = _make_hinge_net(N=N, C=C, H=H, W=W, norm="L2")
        score = rng.randn(N, C, H, W).astype(np.float32)
        label = rng.randint(0, C, size=(N, 1, H, W)).astype(np.float32)
        loss, dx = self._run(net, score, label)
        dx_ref = _hinge_backward_np(score, label, norm="L2")
        np.testing.assert_allclose(dx, dx_ref.astype(np.float32), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(loss, _hinge_forward_np(score, label, "L2"), rtol=1e-5)

    def test_backward_loss_weight_scaling(self):
        """Doubling loss_weight doubles the gradients."""
        rng = np.random.RandomState(44)
        N, C, H, W = 1, 4, 2, 3
        net = _make_hinge_net(N=N, C=C, H=H, W=W, norm="L1")
        score = rng.randn(N, C, H, W).astype(np.float32)
        label = rng.randint(0, C, size=(N, 1, H, W)).astype(np.float32)
        _, dx_w1 = self._run(net, score, label, loss_weight=1.0)
        _, dx_w2 = self._run(net, score, label, loss_weight=2.0)
        np.testing.assert_allclose(dx_w2, 2.0 * dx_w1, rtol=1e-5, atol=1e-6)

    def test_backward_label_no_gradient(self):
        """Label bottom diff stays zero (labels never receive gradients)."""
        net = _make_hinge_net(N=1, C=3, H=1, W=1, norm="L1")
        score = np.array([0.0, 3.0, 0.0], dtype=np.float32).reshape(1, 3, 1, 1)
        label = np.array([0], dtype=np.float32).reshape(1, 1, 1, 1)
        net.forward({"score": score, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dlabel = net.blob_by_name("label").diff
        np.testing.assert_allclose(dlabel, 0.0, atol=1e-8)

    def test_backward_runs_without_crash(self):
        """Backward should complete without errors on random data."""
        net = _make_hinge_net(N=2, C=4, H=2, W=3, norm="L2")
        score = np.random.RandomState(1).randn(2, 4, 2, 3).astype(np.float32)
        label = np.random.randint(0, 4, size=(2, 1, 2, 3)).astype(np.float32)
        net.forward({"score": score, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dx = net.blob_by_name("score").diff
        assert dx.shape == score.shape
        assert dx.dtype == np.float32
        assert np.all(np.isfinite(dx))


# ---------------------------------------------------------------------------
# Tests: Hinge Numerical Gradient
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestHingeNumericalGradient:
    """Central finite-difference gradient checks."""

    def test_numerical_gradient_l1(self):
        """Numerical check for dX (L1)."""
        rng = np.random.RandomState(7)
        N, C, H, W = 1, 4, 2, 3
        net = _make_hinge_net(N=N, C=C, H=H, W=W, norm="L1")
        score = rng.randn(N, C, H, W).astype(np.float32)
        label = rng.randint(0, C, size=(N, 1, H, W)).astype(np.float32)
        net.forward({"score": score, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dx_analytic = net.blob_by_name("score").diff
        dx_num = _num_grad(net, score, label, loss_weight=1.0)
        np.testing.assert_allclose(dx_analytic, dx_num, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_l2(self):
        """Numerical check for dX (L2)."""
        rng = np.random.RandomState(13)
        N, C, H, W = 1, 4, 2, 3
        net = _make_hinge_net(N=N, C=C, H=H, W=W, norm="L2")
        score = rng.randn(N, C, H, W).astype(np.float32)
        label = rng.randint(0, C, size=(N, 1, H, W)).astype(np.float32)
        net.forward({"score": score, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dx_analytic = net.blob_by_name("score").diff
        dx_num = _num_grad(net, score, label, loss_weight=1.0)
        np.testing.assert_allclose(dx_analytic, dx_num, rtol=NUM_RTOL, atol=NUM_ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])