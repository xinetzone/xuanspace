"""MarginRankingLayer Forward/Backward gradient tests.

Covers the pairwise margin ranking loss ``loss_i = max(0, -y_i*(x1_i - x2_i) + margin)``
with three bottoms (x1, x2, label in {-1, +1}):

  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX)
  2. Numerical gradient check (central finite differences) for dX1 and dX2
  3. Known-value verification (satisfied / violated pairs)
  4. margin and sign configurations
  5. loss_weight scaling (via upstream top diff)
  6. Label bottom never receives gradients; shape/finite checks

The scalar loss is the mean over all elements, optionally scaled by `sign`.

Mathematical reference (N elements, loss_weight L):
  loss_i = max(0, -y_i*(x1_i - x2_i) + margin);  loss = sign * mean(loss_i)
  dx1_i = L * sign * (-y_i * mask_i) / N
  dx2_i = L * sign * ( y_i * mask_i) / N
  mask_i = 1 if loss_i > 0 else 0
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

EPS = 1e-3       # central finite-difference step
NUM_RTOL = 1e-3
NUM_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_margin_prototxt(N=1, C=1, H=1, W=4, margin=1.0, sign=1):
    """Create Input(x1) -> Input(x2) -> Input(label) -> MarginRanking prototxt."""
    return textwrap.dedent(f"""\
        name: "test_margin_bw"
        layer {{
          name: "x1"
          type: "Input"
          top: "x1"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "x2"
          type: "Input"
          top: "x2"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "label"
          type: "Input"
          top: "label"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "margin"
          type: "MarginRanking"
          bottom: "x1"
          bottom: "x2"
          bottom: "label"
          top: "loss"
          margin_ranking_param {{ margin: {margin} sign: {sign} }}
        }}
    """)


def _make_margin_net(N=1, C=1, H=1, W=4, margin=1.0, sign=1):
    return Net(_make_margin_prototxt(N, C, H, W, margin=margin, sign=sign))


# ---------------------------------------------------------------------------
# Reference numpy implementations
# ---------------------------------------------------------------------------

def _margin_forward_np(x1, x2, y, margin=1.0, sign=1):
    """Forward reference: loss = sign * mean_i(max(0, -y*(x1-x2)+margin))."""
    v = -y * (x1 - x2) + margin
    loss_i = np.maximum(0.0, v)
    return sign * np.mean(loss_i)


def _margin_backward_np(x1, x2, y, margin=1.0, sign=1, loss_weight=1.0):
    """Backward reference: returns (dx1, dx2)."""
    v = -y * (x1 - x2) + margin
    mask = (v > 0.0).astype(np.float64)
    N = x1.size
    scale = loss_weight * sign / N
    dx1 = scale * (-y * mask)
    dx2 = scale * (y * mask)
    return dx1, dx2


def _num_grad(net, x1, x2, y, loss_weight=1.0, margin=1.0, sign=1, h=EPS):
    """Numerical gradient of loss_weight * loss w.r.t. x1 (central differences)."""
    grad = np.zeros_like(x1, dtype=np.float64)
    flat_x1 = x1.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x1.size):
        orig = flat_x1[i]
        xp = x1.copy()
        xp.ravel()[i] = orig + h
        out_p = net.forward({"x1": xp.astype(np.float32), "x2": x2, "label": y})
        loss_p = float(out_p["loss"].flat[0] * loss_weight)
        xm = x1.copy()
        xm.ravel()[i] = orig - h
        out_m = net.forward({"x1": xm.astype(np.float32), "x2": x2, "label": y})
        loss_m = float(out_m["loss"].flat[0] * loss_weight)
        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# Tests: MarginRanking Forward
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestMarginRankingForward:
    """MarginRanking forward correctness (known values)."""

    def test_forward_known_values(self):
        """Hand-computed loss for mixed satisfied/violated pairs."""
        net = _make_margin_net(N=1, C=1, H=1, W=4, margin=1.0, sign=1)
        # Pairs (x1, x2, y):
        #   (1, 0, 1): v = -1*(1-0)+1 = 0        -> loss 0
        #   (0, 1, 1): v = -1*(0-1)+1 = 2        -> loss 2
        #   (1, 0, -1): v = 1*(1-0)+1 = 2        -> loss 2
        #   (0, 0, 1): v = -1*0+1 = 1            -> loss 1
        # mean = (0+2+2+1)/4 = 1.25
        x1 = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32).reshape(1, 1, 1, 4)
        x2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 1, 4)
        y = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32).reshape(1, 1, 1, 4)
        out = net.forward({"x1": x1, "x2": x2, "label": y})["loss"]
        np.testing.assert_allclose(float(out.flat[0]), 1.25, rtol=1e-6)

    def test_forward_all_satisfied(self):
        """All pairs satisfy the margin: loss = 0."""
        net = _make_margin_net(N=1, C=1, H=1, W=4, margin=1.0, sign=1)
        # x1 far above x2 with y=1 => margin satisfied
        x1 = np.ones((1, 1, 1, 4), dtype=np.float32) * 5.0
        x2 = np.zeros((1, 1, 1, 4), dtype=np.float32)
        y = np.ones((1, 1, 1, 4), dtype=np.float32)
        out = net.forward({"x1": x1, "x2": x2, "label": y})["loss"]
        np.testing.assert_allclose(float(out.flat[0]), 0.0, atol=1e-6)

    def test_forward_sign_negative(self):
        """sign=-1 flips the loss sign."""
        x1 = np.array([0.0, 0.0], dtype=np.float32).reshape(1, 1, 1, 2)
        x2 = np.array([-1.0, -1.0], dtype=np.float32).reshape(1, 1, 1, 2)
        y = np.ones((1, 1, 1, 2), dtype=np.float32)
        net_pos = _make_margin_net(N=1, C=1, H=1, W=2, margin=1.0, sign=1)
        net_neg = _make_margin_net(N=1, C=1, H=1, W=2, margin=1.0, sign=-1)
        loss_pos = float(net_pos.forward({"x1": x1, "x2": x2, "label": y})["loss"].flat[0])
        loss_neg = float(net_neg.forward({"x1": x1, "x2": x2, "label": y})["loss"].flat[0])
        np.testing.assert_allclose(loss_neg, -loss_pos, rtol=1e-6)

    def test_forward_matches_numpy(self):
        """Forward matches numpy reference on random data."""
        rng = np.random.RandomState(0)
        N, C, H, W = 2, 3, 4, 5
        net = _make_margin_net(N=N, C=C, H=H, W=W, margin=0.5, sign=1)
        x1 = rng.randn(N, C, H, W).astype(np.float32)
        x2 = rng.randn(N, C, H, W).astype(np.float32)
        y = rng.choice([-1.0, 1.0], size=(N, C, H, W)).astype(np.float32)
        out = net.forward({"x1": x1, "x2": x2, "label": y})["loss"]
        expected = _margin_forward_np(x1, x2, y, margin=0.5, sign=1)
        np.testing.assert_allclose(float(out.flat[0]), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: MarginRanking Backward (analytical)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestMarginRankingGradient:
    """MarginRanking backward analytical gradient tests."""

    def _run(self, net, x1, x2, y, loss_weight=1.0):
        out = net.forward({"x1": x1, "x2": x2, "label": y})
        loss = float(out["loss"].flat[0])
        net.backward({"loss": np.array([loss_weight], dtype=np.float32)})
        dx1 = net.blob_by_name("x1").diff
        dx2 = net.blob_by_name("x2").diff
        return loss, dx1, dx2

    def test_backward_analytical(self):
        """dx1/dx2 match numpy reference."""
        rng = np.random.RandomState(42)
        N, C, H, W = 2, 3, 4, 5
        net = _make_margin_net(N=N, C=C, H=H, W=W, margin=0.5, sign=1)
        x1 = rng.randn(N, C, H, W).astype(np.float32)
        x2 = rng.randn(N, C, H, W).astype(np.float32)
        y = rng.choice([-1.0, 1.0], size=(N, C, H, W)).astype(np.float32)
        loss, dx1, dx2 = self._run(net, x1, x2, y)
        dx1_ref, dx2_ref = _margin_backward_np(x1, x2, y, margin=0.5, sign=1)
        np.testing.assert_allclose(dx1, dx1_ref.astype(np.float32), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dx2, dx2_ref.astype(np.float32), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(loss, _margin_forward_np(x1, x2, y, margin=0.5), rtol=1e-5)

    def test_backward_loss_weight_scaling(self):
        """Doubling loss_weight doubles the gradients."""
        rng = np.random.RandomState(43)
        N, C, H, W = 1, 3, 3, 4
        net = _make_margin_net(N=N, C=C, H=H, W=W, margin=0.5, sign=1)
        x1 = rng.randn(N, C, H, W).astype(np.float32)
        x2 = rng.randn(N, C, H, W).astype(np.float32)
        y = rng.choice([-1.0, 1.0], size=(N, C, H, W)).astype(np.float32)
        _, dx1_w1, _ = self._run(net, x1, x2, y, loss_weight=1.0)
        _, dx1_w2, _ = self._run(net, x1, x2, y, loss_weight=2.0)
        np.testing.assert_allclose(dx1_w2, 2.0 * dx1_w1, rtol=1e-5, atol=1e-6)

    def test_backward_sign_negative(self):
        """sign=-1 flips gradient signs."""
        rng = np.random.RandomState(44)
        N, C, H, W = 1, 3, 3, 4
        net_pos = _make_margin_net(N=N, C=C, H=H, W=W, margin=0.5, sign=1)
        net_neg = _make_margin_net(N=N, C=C, H=H, W=W, margin=0.5, sign=-1)
        x1 = rng.randn(N, C, H, W).astype(np.float32)
        x2 = rng.randn(N, C, H, W).astype(np.float32)
        y = rng.choice([-1.0, 1.0], size=(N, C, H, W)).astype(np.float32)
        _, dx1_pos, _ = self._run(net_pos, x1, x2, y)
        _, dx1_neg, _ = self._run(net_neg, x1, x2, y)
        np.testing.assert_allclose(dx1_neg, -dx1_pos, rtol=1e-5, atol=1e-6)

    def test_backward_label_no_gradient(self):
        """Label bottom diff stays zero (labels never receive gradients)."""
        net = _make_margin_net(N=1, C=1, H=1, W=4, margin=1.0, sign=1)
        x1 = np.random.RandomState(1).randn(1, 1, 1, 4).astype(np.float32)
        x2 = np.random.RandomState(2).randn(1, 1, 1, 4).astype(np.float32)
        y = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32).reshape(1, 1, 1, 4)
        net.forward({"x1": x1, "x2": x2, "label": y})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dlabel = net.blob_by_name("label").diff
        np.testing.assert_allclose(dlabel, 0.0, atol=1e-8)

    def test_backward_runs_without_crash(self):
        """Backward should complete without errors on random data."""
        net = _make_margin_net(N=2, C=3, H=4, W=5, margin=0.5, sign=1)
        x1 = np.random.RandomState(1).randn(2, 3, 4, 5).astype(np.float32)
        x2 = np.random.RandomState(2).randn(2, 3, 4, 5).astype(np.float32)
        y = np.random.choice([-1.0, 1.0], size=(2, 3, 4, 5)).astype(np.float32)
        net.forward({"x1": x1, "x2": x2, "label": y})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dx1 = net.blob_by_name("x1").diff
        dx2 = net.blob_by_name("x2").diff
        assert dx1.shape == x1.shape and dx2.shape == x2.shape
        assert np.all(np.isfinite(dx1)) and np.all(np.isfinite(dx2))


# ---------------------------------------------------------------------------
# Tests: MarginRanking Numerical Gradient
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestMarginRankingNumericalGradient:
    """Central finite-difference gradient checks."""

    def test_numerical_gradient_dx1(self):
        """Numerical check for dX1 (avoiding the exact kink boundary)."""
        rng = np.random.RandomState(7)
        N, C, H, W = 1, 3, 3, 4
        net = _make_margin_net(N=N, C=C, H=H, W=W, margin=0.5, sign=1)
        x1 = rng.randn(N, C, H, W).astype(np.float32)
        x2 = rng.randn(N, C, H, W).astype(np.float32)
        # Ensure the margin is not exactly satisfied (v != 0) to avoid kink straddling.
        x2 = x2 + 0.3
        y = rng.choice([-1.0, 1.0], size=(N, C, H, W)).astype(np.float32)
        net.forward({"x1": x1, "x2": x2, "label": y})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dx1_analytic = net.blob_by_name("x1").diff
        dx1_num = _num_grad(net, x1, x2, y, loss_weight=1.0, margin=0.5, sign=1)
        np.testing.assert_allclose(dx1_analytic, dx1_num, rtol=NUM_RTOL, atol=NUM_ATOL)

    def test_numerical_gradient_dx2(self):
        """Numerical check for dX2 (via perturbation of x1 note: reuse x1 numeric)."""
        rng = np.random.RandomState(13)
        N, C, H, W = 1, 3, 3, 4
        net = _make_margin_net(N=N, C=C, H=H, W=W, margin=0.5, sign=1)
        x1 = rng.randn(N, C, H, W).astype(np.float32)
        x2 = rng.randn(N, C, H, W).astype(np.float32) + 0.3
        y = rng.choice([-1.0, 1.0], size=(N, C, H, W)).astype(np.float32)
        net.forward({"x1": x1, "x2": x2, "label": y})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        dx2_analytic = net.blob_by_name("x2").diff
        # Numerical gradient w.r.t. x2: perturb x2, hold x1 fixed.
        grad = np.zeros_like(x2, dtype=np.float64)
        flat_x2 = x2.ravel()
        flat_grad = grad.ravel()
        for i in range(flat_x2.size):
            orig = flat_x2[i]
            xp = x2.copy()
            xp.ravel()[i] = orig + EPS
            out_p = net.forward({"x1": x1, "x2": xp.astype(np.float32), "label": y})
            loss_p = float(out_p["loss"].flat[0])
            xm = x2.copy()
            xm.ravel()[i] = orig - EPS
            out_m = net.forward({"x1": x1, "x2": xm.astype(np.float32), "label": y})
            loss_m = float(out_m["loss"].flat[0])
            flat_grad[i] = (loss_p - loss_m) / (2.0 * EPS)
        np.testing.assert_allclose(dx2_analytic, grad.astype(np.float32), rtol=NUM_RTOL, atol=NUM_ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])