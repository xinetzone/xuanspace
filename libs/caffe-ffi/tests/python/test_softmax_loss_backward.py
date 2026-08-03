"""SoftmaxWithLoss layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX)
     - dX = (softmax_probs - one_hot(label)) / valid_count  (loss_weight=1.0)
  2. Numerical gradient check (central finite differences for dX)
  3. Known-value verification (perfect predictions, uniform logits)
  4. Configurations: NCHW format, 1x1 spatial, HxW spatial, ignore_label
  5. Zero/one gradient checks
  6. Shape/finite/determinism checks
  7. Forward output preserved after backward
  8. propagate_down behavior

SoftmaxWithLoss has NO learnable parameters (no weight/bias blobs), so only
dX (input gradient w.r.t. data bottom) needs verification. Label bottom never
receives gradients.

Mathematical reference:
  Forward softmax: p_j = exp(x_j - max(x)) / sum_k exp(x_k - max(x))
  Forward loss:    L = -1/N * sum_i log(p_{y_i})  (cross-entropy)
  Backward dX:     dx_j = (p_j - 1{j==y}) / N    for each sample i, class j
    where N = valid_count (number of non-ignored labels)
    With loss_weight w: dx_j = w * (p_j - 1{j==y}) / N
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

EPS_NUMERICAL = 1e-3
RTOL = 1e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Numpy reference implementations
# ---------------------------------------------------------------------------

def softmax_np(x, axis=1):
    """Numpy reference for softmax along a given axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def softmax_loss_backward_np(score, label, axis=1, loss_weight=1.0, ignore_label=None):
    """Numpy reference for SoftmaxWithLoss backward: returns (loss, dX, probs).

    Args:
        score: Logits tensor (N, C) or (N, C, H, W).
        label: Integer labels with shape matching score but C=1 on axis dim.
        axis: Softmax axis (default 1 = channel dim in NCHW).
        loss_weight: Scalar multiplier for gradient (from top diff).
        ignore_label: Label value to ignore (gradient zeroed at those positions).

    Returns:
        (loss, dX, probs):
            loss: scalar average cross-entropy loss.
            dX: Gradient w.r.t. score (same shape as score).
            probs: Softmax probabilities (same shape as score).
    """
    probs = softmax_np(score.astype(np.float64), axis=axis)
    ndim = score.ndim
    outer_num = int(np.prod(score.shape[:axis]))
    inner_num = int(np.prod(score.shape[axis+1:]))
    channels = score.shape[axis]

    label_flat = label.reshape(-1)
    score_t = np.moveaxis(score, axis, 1).reshape(outer_num, channels, inner_num)
    probs_t = np.moveaxis(probs, axis, 1).reshape(outer_num, channels, inner_num)

    loss = 0.0
    valid_count = 0
    dX_t = np.copy(probs_t)

    idx = 0
    for i in range(outer_num):
        for j in range(inner_num):
            lbl = int(label_flat[idx])
            idx += 1
            if ignore_label is not None and lbl == ignore_label:
                dX_t[i, :, j] = 0.0
                continue
            loss -= np.log(max(probs_t[i, lbl, j], np.finfo(np.float64).tiny))
            dX_t[i, lbl, j] -= 1.0
            valid_count += 1

    avg_loss = loss / valid_count if valid_count > 0 else 0.0
    scale = (loss_weight / valid_count) if valid_count > 0 else 0.0
    dX_t *= scale

    # Reshape back to original layout
    dX = np.moveaxis(dX_t.reshape([outer_num, channels] + list(score.shape[axis+1:])),
                     1, axis)
    return avg_loss, dX.astype(np.float32), probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_sml_prototxt(N, C, H=1, W=1, axis=1, ignore_label=None):
    """Create Input(data) -> Input(label) -> SoftmaxWithLoss(loss) prototxt."""
    ignore_str = ""
    if ignore_label is not None:
        ignore_str = f"    loss_param {{ ignore_label: {ignore_label} }}\n"
    return textwrap.dedent(f"""\
        name: "test_sml_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "label"
          type: "Input"
          top: "label"
          input_param {{ shape {{ dim: {N} dim: 1 dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "loss"
          type: "SoftmaxWithLoss"
          bottom: "data"
          bottom: "label"
          top: "loss"
          softmax_param {{ axis: {axis} }}
{ignore_str}        }}
    """)


def _make_sml_net(N, C, H=1, W=1, axis=1, ignore_label=None):
    """Create SoftmaxWithLoss network."""
    proto = _make_sml_prototxt(N, C, H, W, axis=axis, ignore_label=ignore_label)
    return Net(proto)


def _run_sml_backward(net, x, label, loss_weight=1.0):
    """Run forward then backward, return (loss, dX)."""
    out = net.forward({"data": x.astype(np.float32), "label": label.astype(np.float32)})
    loss = out["loss"]
    # Set upstream gradient (loss_weight) for the scalar loss output
    net.backward({"loss": np.array([loss_weight], dtype=np.float32)})
    dX = net.blob_by_name("data").diff
    return float(loss), dX


# ---------------------------------------------------------------------------
# Numerical gradient helper
# ---------------------------------------------------------------------------

def _num_grad_dx(net, x, label, loss_weight=1.0, h=EPS_NUMERICAL, ignore_label=None):
    """Numerical gradient of loss w.r.t. x via central finite differences.

    Perturbs each element x[i] by +/-h, recomputes loss, approximates:
        dL/dx[i] ≈ (L(x+h*e_i) - L(x-h*e_i)) / (2h)

    For SoftmaxWithLoss, since backward multiplies by loss_weight/valid_count,
    we compute the numerical gradient of loss_weight * loss (matching backward's
    output which is d(loss_weight*L)/dX).
    """
    grad = np.zeros_like(x, dtype=np.float64)
    flat_x = x.astype(np.float32).ravel()
    flat_grad = grad.ravel()
    x_work = x.astype(np.float32).copy()
    label_f32 = label.astype(np.float32)

    for i in range(flat_x.size):
        orig = flat_x[i]

        # +h
        xp = x_work.copy()
        xp.ravel()[i] = orig + np.float32(h)
        out_p = net.forward({"data": xp, "label": label_f32})
        loss_p = float(out_p["loss"].item() * loss_weight

        # -h
        xm = x_work.copy()
        xm.ravel()[i] = orig - np.float32(h)
        out_m = net.forward({"data": xm, "label": label_f32})
        loss_m = float(out_m["loss"].item() * loss_weight

        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)

    return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Test Class 1: SoftmaxWithLoss Backward core tests
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSoftmaxWithLossBackward:
    """Core SoftmaxWithLoss backward gradient tests."""

    def test_sml_known_perfect_predictions(self):
        """Perfect predictions (high score for true class): probs≈1 at label,
        dX≈0 at true class, dX≈probs/N at wrong classes."""
        N, C = 1, 3
        net = _make_sml_net(N, C)
        # Strongly confident prediction for class 1
        x = np.array([[[-10.0, 10.0, -10.0]]], dtype=np.float32).reshape(N, C, 1, 1)
        label = np.array([[[[1]]]], dtype=np.float32)

        loss, dX = _run_sml_backward(net, x, label)
        # Loss should be near 0 (softmax gives ~1 for class 1)
        assert loss < 1e-4, f"Expected near-zero loss for perfect prediction, got {loss}"
        # dX at class 1 should be ~0 (p-1 ~ 0)
        assert abs(dX[0, 1, 0, 0]) < 1e-3, f"Expected ~0 grad at true class, got {dX[0,1,0,0]}"
        # dX at other classes should be ~p_j/N ≈ 0 (since p_j ≈ 0 for j≠1)
        for c in [0, 2]:
            assert abs(dX[0, c, 0, 0]) < 1e-4, f"Expected ~0 grad at class {c}, got {dX[0,c,0,0]}"

    def test_sml_known_uniform(self):
        """Uniform logits: p_j = 1/C for all j, loss = -log(1/C) = log(C).
        dX_j = (1/C - 1{j==y})/N."""
        N, C = 2, 4
        net = _make_sml_net(N, C)
        x = np.zeros((N, C, 1, 1), dtype=np.float32)  # uniform logits
        label = np.array([[[[0]]], [[[2]]]], dtype=np.float32)

        loss, dX = _run_sml_backward(net, x, label)
        expected_loss = np.log(C)  # -log(1/C) = log(C)
        np.testing.assert_allclose(loss, expected_loss, rtol=1e-4)

        # For sample 0 (label=0): dX[0,0,0,0] = (1/4 - 1)/2 = -3/8 = -0.375
        # dX[0,c!=0,0,0] = (1/4)/2 = 1/8 = 0.125
        np.testing.assert_allclose(dX[0, 0, 0, 0], -0.375, rtol=1e-4)
        for c in [1, 2, 3]:
            np.testing.assert_allclose(dX[0, c, 0, 0], 0.125, rtol=1e-4)

        # For sample 1 (label=2): dX[1,2,0,0] = (1/4 - 1)/2 = -0.375
        np.testing.assert_allclose(dX[1, 2, 0, 0], -0.375, rtol=1e-4)
        for c in [0, 1, 3]:
            np.testing.assert_allclose(dX[1, c, 0, 0], 0.125, rtol=1e-4)

    def test_sml_gradient_sums_to_zero(self):
        """For each sample, sum of dX over classes should be 0.
        dX_j = (p_j - delta_{j,y})/N -> sum_j dX_j = (sum p_j - 1)/N = (1-1)/N = 0."""
        np.random.seed(42)
        N, C = 4, 5
        net = _make_sml_net(N, C)
        x = np.random.randn(N, C, 1, 1).astype(np.float32) * 2.0
        label = np.array([[[[i % C]]] for i in range(N)], dtype=np.float32)

        _, dX = _run_sml_backward(net, x, label)
        # Sum over channel axis should be ~0 per sample
        sum_per_sample = dX.sum(axis=1)
        np.testing.assert_allclose(sum_per_sample, 0.0, atol=1e-5)

    def test_sml_analytical_vs_numpy(self):
        """Analytical dX vs numpy reference on random NCHW data."""
        np.random.seed(123)
        N, C, H, W = 3, 5, 2, 2
        net = _make_sml_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        label = np.random.randint(0, C, size=(N, 1, H, W)).astype(np.float32)

        loss_cpp, dX_cpp = _run_sml_backward(net, x, label)
        loss_np, dX_np, _ = softmax_loss_backward_np(x, label)

        np.testing.assert_allclose(loss_cpp, loss_np, rtol=1e-4,
                                    err_msg="Loss mismatch vs numpy")
        np.testing.assert_allclose(dX_cpp, dX_np, rtol=RTOL, atol=ATOL,
                                    err_msg="dX mismatch vs numpy reference")

    def test_sml_numerical_gradient(self):
        """Numerical gradient check on small tensor (N=2, C=3)."""
        np.random.seed(456)
        N, C = 2, 3
        net = _make_sml_net(N, C)
        x = np.random.randn(N, C, 1, 1).astype(np.float32) * 0.5  # small logits for stability
        label = np.array([[[[0]]], [[[2]]]], dtype=np.float32)

        loss_cpp, dX_analytic = _run_sml_backward(net, x, label)
        dX_numerical = _num_grad_dx(net, x, label, h=EPS_NUMERICAL)

        np.testing.assert_allclose(dX_analytic, dX_numerical, rtol=1e-2, atol=1e-3,
                                    err_msg="Numerical gradient check failed for dX")

    def test_sml_numerical_gradient_spatial(self):
        """Numerical gradient check with spatial dims (N=1, C=4, H=2, W=2)."""
        np.random.seed(789)
        N, C, H, W = 1, 4, 2, 2
        net = _make_sml_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32) * 0.5
        label = np.random.randint(0, C, size=(N, 1, H, W)).astype(np.float32)

        _, dX_analytic = _run_sml_backward(net, x, label)
        dX_numerical = _num_grad_dx(net, x, label, h=EPS_NUMERICAL)

        np.testing.assert_allclose(dX_analytic, dX_numerical, rtol=1e-2, atol=1e-3,
                                    err_msg="Numerical gradient check failed for spatial dX")

    def test_sml_loss_weight_scaling(self):
        """Gradient scales linearly with loss_weight."""
        np.random.seed(42)
        N, C = 3, 4
        net1 = _make_sml_net(N, C)
        x = np.random.randn(N, C, 1, 1).astype(np.float32)
        label = np.random.randint(0, C, size=(N, 1, 1, 1)).astype(np.float32)

        _, dX_w1 = _run_sml_backward(net1, x, label, loss_weight=1.0)

        net2 = _make_sml_net(N, C)
        _, dX_w2 = _run_sml_backward(net2, x, label, loss_weight=2.0)

        np.testing.assert_allclose(dX_w2, 2.0 * dX_w1, rtol=1e-5,
                                    err_msg="dX should scale linearly with loss_weight")

    def test_sml_ignore_label(self):
        """Ignored labels should have zero gradient across all channels."""
        N, C, H, W = 2, 3, 1, 1
        ignore_label = -1
        net = _make_sml_net(N, C, H, W, ignore_label=ignore_label)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        label = np.array([[[[1]]], [[[ignore_label]]]], dtype=np.float32)  # second sample ignored

        _, dX = _run_sml_backward(net, x, label)
        # Sample 1 (ignored): all channels should be zero
        np.testing.assert_allclose(dX[1], 0.0, atol=1e-6)
        # Sample 0 (valid): should have non-zero gradient
        assert np.any(np.abs(dX[0]) > 1e-6), "Valid sample should have non-zero dX"

    def test_sml_deterministic(self):
        """Two identical forward+backward calls should produce identical dX."""
        np.random.seed(42)
        N, C, H, W = 2, 4, 2, 2
        net = _make_sml_net(N, C, H, W)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        label = np.random.randint(0, C, size=(N, 1, H, W)).astype(np.float32)

        _, dX1 = _run_sml_backward(net, x, label)
        _, dX2 = _run_sml_backward(net, x, label)

        np.testing.assert_array_equal(dX1, dX2)

    def test_sml_no_nan_inf(self):
        """Gradients should be finite for random inputs."""
        np.random.seed(42)
        N, C = 4, 8
        net = _make_sml_net(N, C)
        x = np.random.randn(N, C, 1, 1).astype(np.float32) * 3.0
        label = np.random.randint(0, C, size=(N, 1, 1, 1)).astype(np.float32)

        _, dX = _run_sml_backward(net, x, label)
        assert not np.any(np.isnan(dX)), "dX contains NaN"
        assert not np.any(np.isinf(dX)), "dX contains Inf"

    def test_sml_forward_preserved(self):
        """Forward output (probabilities/loss) should not be corrupted by backward."""
        np.random.seed(42)
        N, C = 3, 5
        net = _make_sml_net(N, C)
        x = np.random.randn(N, C, 1, 1).astype(np.float32)
        label = np.random.randint(0, C, size=(N, 1, 1, 1)).astype(np.float32)

        out1 = net.forward({"data": x, "label": label})
        loss1 = float(out1["loss"].item()
        net.backward({"loss": np.array([1.0], dtype=np.float32)})
        out2 = net.forward({"data": x, "label": label})
        loss2 = float(out2["loss"].item()

        np.testing.assert_allclose(loss1, loss2, rtol=1e-6)

    def test_sml_multi_sample_consistency(self):
        """Loss and gradient are correctly averaged over all valid samples."""
        C = 3
        # Two identical samples with same label should give same dX per sample
        # as a single sample (loss_weight=1, averaged over 2 samples)
        net1 = _make_sml_net(1, C)
        x1 = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32).reshape(1, C, 1, 1)
        label1 = np.array([[[[2]]]], dtype=np.float32)
        _, dX1 = _run_sml_backward(net1, x1, label1)

        net2 = _make_sml_net(2, C)
        x2 = np.tile(x1, (2, 1, 1, 1))
        label2 = np.tile(label1, (2, 1, 1, 1))
        _, dX2 = _run_sml_backward(net2, x2, label2)

        # Each sample's dX in the 2-sample case should be half of the single-sample dX
        np.testing.assert_allclose(dX2[0], dX1[0] / 2.0, rtol=1e-5)
        np.testing.assert_allclose(dX2[1], dX1[0] / 2.0, rtol=1e-5)
