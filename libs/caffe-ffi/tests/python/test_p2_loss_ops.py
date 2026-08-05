"""P2 loss-layer unit tests: ContrastiveLoss, InfogainLoss, MultinomialLogisticLoss.

Covers for each op:
  1. Forward numerical correctness (numpy reference vs caffe-ffi scalar loss)
  2. Backward gradient correctness (analytical vs numpy reference)
  3. Numerical gradient check (central finite differences)
  4. Registration / instantiation
  5. Branch-specific configs (normalization modes, infogain matrix, margin)

Mathematical references (from the C++ sources):

  ContrastiveLoss (contrastive_loss_layer.cpp):
    For each sample i with diff = a_i - b_i, dist_sq = ||diff||^2, y = label[i]:
      y == 1 : L_i = dist_sq
      y == 0 : if dist_sq < margin: L_i = (margin - dist_sq)^2   (non-legacy)
    forward: loss = sum(L_i) / max(1, normalizer)
    backward: dX = alpha*diff, dY = -alpha*diff, where
      alpha = 2*scale                    for y == 1
      alpha = -4*(margin - dist_sq)*scale for y == 0 && dist_sq < margin  (non-legacy)
      scale = loss_weight / normalizer

  InfogainLoss (infogain_loss_layer.cpp):
    Forward computes a stable softmax of bottom[0] over the softmax axis, then
      loss = -sum_k H[gt,k] * log(p_k)   (with H = identity when no infogain bottom)
    backward: dL/dx_j = p_j * H_row_sum - H[gt,j]  (identity: p_j - delta_{gt,j})

  MultinomialLogisticLoss (multinomial_logistic_loss_layer.cpp):
    forward: loss = -log(p_gt) ; normalizer default BATCH_SIZE
    backward: dL/dp[gt] = -scale/p_gt, others = 0
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

RTOL = 1e-3
ATOL = 1e-4
EPS_NUMERICAL = 1e-3


# ---------------------------------------------------------------------------
# Numpy reference implementations
# ---------------------------------------------------------------------------

def _normalizer(kind: str, outer_num: int, inner_num: int, valid_count: int) -> float:
    """Normalizer for a loss layer given the LossParameter.normalization mode."""
    if kind == "FULL":
        return float(outer_num * inner_num)
    if kind == "VALID":
        return float(valid_count)
    if kind == "BATCH_SIZE":
        return float(outer_num)
    if kind == "NONE":
        return 1.0
    raise ValueError(f"Unknown normalization: {kind}")


def contrastive_loss_np(a, b, label, margin=1.0, normalization="BATCH_SIZE",
                        legacy=False):
    """Numpy reference for ContrastiveLoss forward scalar loss."""
    a = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    b = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    label = np.asarray(label, dtype=np.float64).ravel()
    n = a.shape[0]
    diff = a - b
    dist_sq = np.sum(diff * diff, axis=1)
    total = 0.0
    valid_count = 0
    for i in range(n):
        y = label[i]
        if y == 1.0:
            total += dist_sq[i]
        else:
            if legacy:
                d = np.sqrt(dist_sq[i])
                if d < margin:
                    total += (margin - d) ** 2
            else:
                if dist_sq[i] < margin:
                    total += (margin - dist_sq[i]) ** 2
        valid_count += 1
    norm = _normalizer(normalization, n, 1, valid_count)
    return total / max(1.0, norm)


def contrastive_backward_np(a, b, label, loss_weight=1.0, margin=1.0,
                            normalization="BATCH_SIZE", legacy=False):
    """Numpy reference for ContrastiveLoss backward (dA, dB)."""
    a = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    b = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    label = np.asarray(label, dtype=np.float64).ravel()
    n = a.shape[0]
    diff = a - b
    dist_sq = np.sum(diff * diff, axis=1)
    norm = max(1.0, _normalizer(normalization, n, 1, n))
    scale = loss_weight / norm
    da = np.zeros_like(a)
    db = np.zeros_like(b)
    for i in range(n):
        y = label[i]
        alpha = 0.0
        if y == 1.0:
            alpha = 2.0 * scale
        else:
            if legacy:
                d = np.sqrt(dist_sq[i])
                if 0 < d < margin:
                    alpha = -2.0 * (margin - d) / d * scale
            else:
                if dist_sq[i] < margin:
                    alpha = -4.0 * (margin - dist_sq[i]) * scale
        da[i] = alpha * diff[i]
        db[i] = -alpha * diff[i]
    return da.astype(np.float32), db.astype(np.float32)


def _softmax_rows(x):
    """Row-wise stable softmax over the last axis; x shape (N, C)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def infogain_loss_np(logits, label, infogain=None, normalization="BATCH_SIZE"):
    """Numpy reference for InfogainLoss forward scalar loss (axis=1, 2D input)."""
    p = _softmax_rows(logits)
    label = np.asarray(label, dtype=np.int64).ravel()
    n, c = p.shape
    total = 0.0
    valid_count = 0
    for i in range(n):
        gt = label[i]
        if infogain is None:
            total -= np.log(max(p[i, gt], np.finfo(np.float64).tiny))
        else:
            h_row = np.asarray(infogain, dtype=np.float64)[gt]
            total -= np.sum(h_row * np.log(np.maximum(p[i], np.finfo(np.float64).tiny)))
        valid_count += 1
    norm = _normalizer(normalization, n, 1, valid_count)
    return total / max(1.0, norm)


def infogain_backward_np(logits, label, infogain=None, loss_weight=1.0,
                         normalization="BATCH_SIZE"):
    """Numpy reference for InfogainLoss backward dL/dlogits (identity/H matrix)."""
    p = _softmax_rows(logits)
    label = np.asarray(label, dtype=np.int64).ravel()
    n, c = p.shape
    norm = max(1.0, _normalizer(normalization, n, 1, n))
    scale = loss_weight / norm
    grad = np.zeros_like(p)
    for i in range(n):
        gt = label[i]
        if infogain is None:
            h_row_sum = 1.0
            h_gt = np.zeros(c)
            h_gt[gt] = 1.0
        else:
            h = np.asarray(infogain, dtype=np.float64)
            h_gt = h[gt]
            h_row_sum = h_gt.sum()
        grad[i] = p[i] * h_row_sum - h_gt
    return (grad * scale).astype(np.float32)


def multinomial_loss_np(prob, label, normalization="BATCH_SIZE"):
    """Numpy reference for MultinomialLogisticLoss forward scalar loss."""
    prob = np.asarray(prob, dtype=np.float64).reshape(prob.shape[0], -1)
    label = np.asarray(label, dtype=np.int64).ravel()
    n = prob.shape[0]
    total = 0.0
    valid_count = 0
    for i in range(n):
        gt = label[i]
        total -= np.log(max(prob[i, gt], np.finfo(np.float64).tiny))
        valid_count += 1
    norm = _normalizer(normalization, n, 1, valid_count)
    return total / max(1.0, norm)


def multinomial_backward_np(prob, label, loss_weight=1.0, normalization="BATCH_SIZE"):
    """Numpy reference for MultinomialLogisticLoss backward dL/dprob."""
    prob = np.asarray(prob, dtype=np.float64).reshape(prob.shape[0], -1)
    label = np.asarray(label, dtype=np.int64).ravel()
    n = prob.shape[0]
    norm = max(1.0, _normalizer(normalization, n, 1, n))
    scale = loss_weight / norm
    grad = np.zeros_like(prob)
    for i in range(n):
        gt = label[i]
        grad[i, gt] = -scale / max(prob[i, gt], np.finfo(np.float64).tiny)
    return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Prototxt builders
# ---------------------------------------------------------------------------

def _make_contrastive_prototxt(N, D, margin=1.0, normalization=None):
    loss_param = ""
    if normalization is not None:
        loss_param = f"  loss_param {{ normalization: {normalization} }}\n"
    return textwrap.dedent(f"""\
        name: "test_contrastive"
        layer {{
          name: "a"
          type: "Input"
          top: "a"
          input_param {{ shape {{ dim: {N} dim: {D} }} }}
        }}
        layer {{
          name: "b"
          type: "Input"
          top: "b"
          input_param {{ shape {{ dim: {N} dim: {D} }} }}
        }}
        layer {{
          name: "label"
          type: "Input"
          top: "label"
          input_param {{ shape {{ dim: {N} }} }}
        }}
        layer {{
          name: "loss"
          type: "ContrastiveLoss"
          bottom: "a"
          bottom: "b"
          bottom: "label"
          top: "loss"
          contrastive_loss_param {{ margin: {margin} }}
{loss_param}        }}
    """)


def _make_infogain_prototxt(N, C, normalization=None, with_infogain=False):
    loss_param = ""
    if normalization is not None:
        loss_param = f"  loss_param {{ normalization: {normalization} }}\n"
    infogain_layer = ""
    infogain_bottom = ""
    if with_infogain:
        infogain_layer = textwrap.dedent(f"""\
            layer {{
              name: "infogain"
              type: "Input"
              top: "infogain"
              input_param {{ shape {{ dim: {C} dim: {C} }} }}
            }}
        """)
        infogain_bottom = '  bottom: "infogain"\n'
    return textwrap.dedent(f"""\
        name: "test_infogain"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {N} dim: {C} }} }}
        }}
        layer {{
          name: "label"
          type: "Input"
          top: "label"
          input_param {{ shape {{ dim: {N} }} }}
        }}
        {infogain_layer}layer {{
          name: "loss"
          type: "InfogainLoss"
          bottom: "data"
          bottom: "label"
{infogain_bottom}          top: "loss"
{loss_param}        }}
    """)


def _make_multinomial_prototxt(N, C, normalization=None):
    loss_param = ""
    if normalization is not None:
        loss_param = f"  loss_param {{ normalization: {normalization} }}\n"
    return textwrap.dedent(f"""\
        name: "test_multinomial"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {N} dim: {C} }} }}
        }}
        layer {{
          name: "label"
          type: "Input"
          top: "label"
          input_param {{ shape {{ dim: {N} }} }}
        }}
        layer {{
          name: "loss"
          type: "MultinomialLogisticLoss"
          bottom: "data"
          bottom: "label"
          top: "loss"
{loss_param}        }}
    """)


def _forward_loss(net, inputs):
    """Run forward and return the scalar loss."""
    out = net.Forward(inputs)
    return float(out["loss"].data.flat[0])


def _run_backward(net, inputs, loss_weight=1.0, diff_blobs=("data",)):
    """Run forward then backward; return dict {blob_name: diff numpy}."""
    net.Forward(inputs)
    net.backward({"loss": np.array([loss_weight], dtype=np.float32)})
    return {name: net.blob_by_name(name).diff.copy() for name in diff_blobs}


def _loss_num_grad(net, blob_name, inputs, h=EPS_NUMERICAL):
    """Numerical gradient of the scalar loss w.r.t. one input blob."""
    x0 = inputs[blob_name]
    grad = np.zeros_like(x0, dtype=np.float64)
    flat_x = x0.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x.size):
        orig = flat_x[i]
        xp = x0.copy()
        xp.ravel()[i] = orig + np.float32(h)
        lp = _forward_loss(net, {**inputs, blob_name: xp})
        xm = x0.copy()
        xm.ravel()[i] = orig - np.float32(h)
        lm = _forward_loss(net, {**inputs, blob_name: xm})
        flat_grad[i] = (lp - lm) / (2.0 * h)
    return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Test Class: ContrastiveLoss
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestContrastiveLoss:
    def test_registration(self):
        net = Net(_make_contrastive_prototxt(2, 3))
        types = [l.type for l in net.layers_array()]
        assert "ContrastiveLoss" in types

    def test_forward_known(self):
        np.random.seed(0)
        N, D = 4, 3
        net = Net(_make_contrastive_prototxt(N, D))
        a = np.random.randn(N, D).astype(np.float32)
        b = np.random.randn(N, D).astype(np.float32)
        label = np.array([1, 0, 1, 0], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"a": a, "b": b, "label": label})
        loss_np = contrastive_loss_np(a, b, label)
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL)

    def test_forward_margin_boundary(self):
        # y=0 samples with dist_sq < margin contribute (margin-dist_sq)^2.
        np.random.seed(1)
        N, D = 3, 2
        net = Net(_make_contrastive_prototxt(N, D, margin=1.0))
        a = np.random.randn(N, D).astype(np.float32) * 0.1   # small dists
        b = np.random.randn(N, D).astype(np.float32) * 0.1
        label = np.array([0, 0, 0], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"a": a, "b": b, "label": label})
        loss_np = contrastive_loss_np(a, b, label, margin=1.0)
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL)

    def test_forward_far_apart_negatives(self):
        # y=0 samples with dist_sq >= margin contribute nothing.
        N, D = 2, 2
        net = Net(_make_contrastive_prototxt(N, D, margin=1.0))
        a = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32)
        b = np.zeros((N, D), dtype=np.float32)
        label = np.array([0, 0], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"a": a, "b": b, "label": label})
        np.testing.assert_allclose(loss_cpp, 0.0, atol=1e-6)

    def test_backward_analytical_vs_numpy(self):
        np.random.seed(2)
        N, D = 4, 3
        net = Net(_make_contrastive_prototxt(N, D))
        a = np.random.randn(N, D).astype(np.float32) * 0.3
        b = np.random.randn(N, D).astype(np.float32) * 0.3
        label = np.array([1, 0, 1, 0], dtype=np.float32)
        diffs = _run_backward(net, {"a": a, "b": b, "label": label},
                              diff_blobs=("a", "b"))
        da_np, db_np = contrastive_backward_np(a, b, label)
        np.testing.assert_allclose(diffs["a"], da_np, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(diffs["b"], db_np, rtol=RTOL, atol=ATOL)

    def test_backward_label_no_gradient(self):
        np.random.seed(3)
        N, D = 3, 2
        net = Net(_make_contrastive_prototxt(N, D))
        a = np.random.randn(N, D).astype(np.float32)
        b = np.random.randn(N, D).astype(np.float32)
        label = np.array([1, 0, 1], dtype=np.float32)
        dlabel = _run_backward(net, {"a": a, "b": b, "label": label},
                               diff_blobs=("label",))["label"]
        np.testing.assert_allclose(dlabel, 0.0, atol=1e-6)

    def test_numerical_gradient(self):
        np.random.seed(4)
        N, D = 2, 2
        net = Net(_make_contrastive_prototxt(N, D))
        a = np.random.randn(N, D).astype(np.float32) * 0.2
        b = np.random.randn(N, D).astype(np.float32) * 0.2
        label = np.array([1, 0], dtype=np.float32)
        inputs = {"a": a, "b": b, "label": label}
        diffs = _run_backward(net, inputs, diff_blobs=("a", "b"))
        ga_num = _loss_num_grad(net, "a", inputs)
        gb_num = _loss_num_grad(net, "b", inputs)
        np.testing.assert_allclose(diffs["a"], ga_num, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(diffs["b"], gb_num, rtol=1e-2, atol=1e-3)

    def test_normalization_modes(self):
        np.random.seed(5)
        N, D = 3, 2
        a = np.random.randn(N, D).astype(np.float32) * 0.2
        b = np.random.randn(N, D).astype(np.float32) * 0.2
        label = np.array([1, 0, 1], dtype=np.float32)
        for mode in ["FULL", "VALID", "BATCH_SIZE", "NONE"]:
            net = Net(_make_contrastive_prototxt(N, D, normalization=mode))
            loss_cpp = _forward_loss(net, {"a": a, "b": b, "label": label})
            loss_np = contrastive_loss_np(a, b, label, normalization=mode)
            np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL,
                                       err_msg=f"normalization={mode}")

    def test_loss_weight_scaling(self):
        np.random.seed(6)
        N, D = 3, 2
        net = Net(_make_contrastive_prototxt(N, D))
        a = np.random.randn(N, D).astype(np.float32) * 0.2
        b = np.random.randn(N, D).astype(np.float32) * 0.2
        label = np.array([1, 0, 1], dtype=np.float32)
        inputs = {"a": a, "b": b, "label": label}
        d1 = _run_backward(net, inputs, loss_weight=1.0, diff_blobs=("a",))["a"]
        d2 = _run_backward(net, inputs, loss_weight=2.0, diff_blobs=("a",))["a"]
        np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test Class: InfogainLoss
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInfogainLoss:
    def test_registration(self):
        net = Net(_make_infogain_prototxt(3, 4))
        types = [l.type for l in net.layers_array()]
        assert "InfogainLoss" in types

    def test_forward_identity(self):
        np.random.seed(10)
        N, C = 4, 5
        net = Net(_make_infogain_prototxt(N, C))
        logits = np.random.randn(N, C).astype(np.float32)
        label = np.array([0, 2, 1, 4], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"data": logits, "label": label})
        loss_np = infogain_loss_np(logits, label)
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL)

    def test_forward_with_infogain_matrix(self):
        np.random.seed(11)
        N, C = 3, 3
        net = Net(_make_infogain_prototxt(N, C, with_infogain=True))
        logits = np.random.randn(N, C).astype(np.float32)
        label = np.array([0, 1, 2], dtype=np.float32)
        # A non-identity infogain matrix (positive weights).
        infogain = np.array([[1.0, 0.5, 0.2],
                             [0.3, 1.0, 0.4],
                             [0.1, 0.2, 1.0]], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"data": logits, "label": label,
                                       "infogain": infogain})
        loss_np = infogain_loss_np(logits, label, infogain=infogain)
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL)

    def test_backward_identity_analytical_vs_numpy(self):
        np.random.seed(12)
        N, C = 4, 3
        net = Net(_make_infogain_prototxt(N, C))
        logits = np.random.randn(N, C).astype(np.float32)
        label = np.array([0, 1, 2, 0], dtype=np.float32)
        inputs = {"data": logits, "label": label}
        d = _run_backward(net, inputs, diff_blobs=("data",))["data"]
        d_np = infogain_backward_np(logits, label)
        np.testing.assert_allclose(d, d_np, rtol=RTOL, atol=ATOL)

    def test_backward_with_infogain_matrix(self):
        np.random.seed(13)
        N, C = 3, 3
        net = Net(_make_infogain_prototxt(N, C, with_infogain=True))
        logits = np.random.randn(N, C).astype(np.float32)
        label = np.array([0, 2, 1], dtype=np.float32)
        infogain = np.array([[1.0, 0.5, 0.2],
                             [0.3, 1.0, 0.4],
                             [0.1, 0.2, 1.0]], dtype=np.float32)
        inputs = {"data": logits, "label": label, "infogain": infogain}
        d = _run_backward(net, inputs, diff_blobs=("data",))["data"]
        d_np = infogain_backward_np(logits, label, infogain=infogain)
        np.testing.assert_allclose(d, d_np, rtol=RTOL, atol=ATOL)

    def test_numerical_gradient(self):
        np.random.seed(14)
        N, C = 2, 3
        net = Net(_make_infogain_prototxt(N, C))
        logits = np.random.randn(N, C).astype(np.float32) * 0.5
        label = np.array([0, 2], dtype=np.float32)
        inputs = {"data": logits, "label": label}
        d = _run_backward(net, inputs, diff_blobs=("data",))["data"]
        d_num = _loss_num_grad(net, "data", inputs)
        np.testing.assert_allclose(d, d_num, rtol=1e-2, atol=1e-3)

    def test_perfect_prediction_low_loss(self):
        N, C = 2, 3
        net = Net(_make_infogain_prototxt(N, C))
        # Strong logits for the ground-truth class -> low loss.
        logits = np.array([[10.0, 0.0, 0.0], [0.0, -10.0, 10.0]], dtype=np.float32)
        label = np.array([0, 2], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"data": logits, "label": label})
        assert loss_cpp < 1e-3, f"Expected low loss, got {loss_cpp}"


# ---------------------------------------------------------------------------
# Test Class: MultinomialLogisticLoss
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestMultinomialLogisticLoss:
    def test_registration(self):
        net = Net(_make_multinomial_prototxt(3, 4))
        types = [l.type for l in net.layers_array()]
        assert "MultinomialLogisticLoss" in types

    def test_forward_known(self):
        np.random.seed(20)
        N, C = 4, 5
        net = Net(_make_multinomial_prototxt(N, C))
        # Valid probability rows (positive, not necessarily summing to 1).
        prob = np.abs(np.random.randn(N, C)).astype(np.float32) + 0.1
        label = np.array([0, 3, 1, 2], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"data": prob, "label": label})
        loss_np = multinomial_loss_np(prob, label)
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL)

    def test_forward_softmax_rows(self):
        N, C = 2, 3
        net = Net(_make_multinomial_prototxt(N, C))
        prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float32)
        label = np.array([0, 2], dtype=np.float32)
        loss_cpp = _forward_loss(net, {"data": prob, "label": label})
        loss_np = -np.log(np.array([0.7, 0.6])).mean()
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=1e-5)

    def test_backward_analytical_vs_numpy(self):
        np.random.seed(21)
        N, C = 4, 3
        net = Net(_make_multinomial_prototxt(N, C))
        prob = np.abs(np.random.randn(N, C)).astype(np.float32) + 0.1
        label = np.array([0, 1, 2, 0], dtype=np.float32)
        inputs = {"data": prob, "label": label}
        d = _run_backward(net, inputs, diff_blobs=("data",))["data"]
        d_np = multinomial_backward_np(prob, label)
        np.testing.assert_allclose(d, d_np, rtol=RTOL, atol=ATOL)

    def test_numerical_gradient(self):
        np.random.seed(22)
        N, C = 2, 3
        net = Net(_make_multinomial_prototxt(N, C))
        prob = (np.abs(np.random.randn(N, C)).astype(np.float32) + 0.5)
        label = np.array([0, 2], dtype=np.float32)
        inputs = {"data": prob, "label": label}
        d = _run_backward(net, inputs, diff_blobs=("data",))["data"]
        d_num = _loss_num_grad(net, "data", inputs)
        np.testing.assert_allclose(d, d_num, rtol=1e-2, atol=1e-3)

    def test_backward_only_gt_channel_nonzero(self):
        N, C = 2, 3
        net = Net(_make_multinomial_prototxt(N, C))
        prob = (np.abs(np.random.randn(N, C)).astype(np.float32) + 0.5)
        label = np.array([1, 2], dtype=np.float32)
        inputs = {"data": prob, "label": label}
        d = _run_backward(net, inputs, diff_blobs=("data",))["data"]
        # Only the ground-truth channel entries receive gradients.
        non_gt = d.copy()
        non_gt[0, int(label[0])] = 0.0
        non_gt[1, int(label[1])] = 0.0
        np.testing.assert_allclose(non_gt, 0.0, atol=1e-6)
        assert d[0, int(label[0])] < 0.0 and d[1, int(label[1])] < 0.0

    def test_normalization_modes(self):
        np.random.seed(23)
        N, C = 3, 2
        prob = (np.abs(np.random.randn(N, C)).astype(np.float32) + 0.5)
        label = np.array([0, 1, 0], dtype=np.float32)
        for mode in ["FULL", "VALID", "BATCH_SIZE", "NONE"]:
            net = Net(_make_multinomial_prototxt(N, C, normalization=mode))
            loss_cpp = _forward_loss(net, {"data": prob, "label": label})
            loss_np = multinomial_loss_np(prob, label, normalization=mode)
            np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL,
                                       err_msg=f"normalization={mode}")