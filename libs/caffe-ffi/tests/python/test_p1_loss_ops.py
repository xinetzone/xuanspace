"""P1 loss layer ops: SigmoidCrossEntropyLoss and EuclideanLoss.

Covers for each op:
  1. Forward numerical correctness (numpy reference vs caffe-ffi scalar loss)
  2. Backward gradient correctness (analytical vs numpy reference)
  3. Numerical gradient check (central finite differences)
  4. Known-value checks (perfect prediction, all-0.5 predictions, zero diff)
  5. Branch-specific configs (normalization modes / ignore_label for sigmoid-CE;
     batch-size scaling for Euclidean)
  6. Registration / instantiation

Mathematical references (from the C++ sources):

  SigmoidCrossEntropyLoss (sigmoid_cross_entropy_loss_layer.cpp):
    For each element:
      stable per-element loss L_i = max(x,0) - x*y + log1p(exp(-|x|))
      (numerically equivalent to -[ x*(y - (x>=0)) - log(1+exp(x - 2*x*(x>=0))) ])
    Sum over non-ignored elements, then divide by normalizer where the
    normalizer depends on LossParameter.normalization:
      FULL        -> outer_num * inner_num
      VALID       -> number of valid (non-ignored) elements
      BATCH_SIZE  -> outer_num  (default)
      NONE        -> 1.0
    forward: loss = sum(L_i) / max(1, normalizer)
    backward: dX = (sigmoid(x) - y) * loss_weight / normalizer, zeroed at
    ignored positions.  Only bottom[0] receives gradients.

  EuclideanLoss (euclidean_loss_layer.cpp):
    forward: loss = ||x - y||^2 / (2 * N),  N = bottom[0]->shape(0)
    backward: dX = (x - y) * loss_weight / N,  dY = -(x - y) * loss_weight / N
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

def sigmoid_ce_loss_np(x, y, normalization="BATCH_SIZE", ignore_label=None):
    """Numpy reference for SigmoidCrossEntropyLoss forward scalar loss."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    outer_num = int(x.shape[0])
    inner_num = int(np.prod(x.shape[1:]))

    per_elem = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
    valid = np.ones_like(x, dtype=np.float64)
    if ignore_label is not None:
        valid = (y != ignore_label).astype(np.float64)
    total = np.sum(per_elem * valid)
    valid_count = int(np.sum(valid))

    if normalization == "FULL":
        normalizer = float(outer_num * inner_num)
    elif normalization == "VALID":
        normalizer = float(valid_count)
    elif normalization == "BATCH_SIZE":
        normalizer = float(outer_num)
    elif normalization == "NONE":
        normalizer = 1.0
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    normalizer = max(1.0, normalizer)
    return total / normalizer


def sigmoid_ce_backward_np(x, y, loss_weight=1.0, normalization="BATCH_SIZE",
                           ignore_label=None):
    """Numpy reference for SigmoidCrossEntropyLoss backward dX."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-x))
    dx = sig - y
    if ignore_label is not None:
        dx = dx * (y != ignore_label).astype(np.float64)

    outer_num = int(x.shape[0])
    inner_num = int(np.prod(x.shape[1:]))
    valid = (y != ignore_label).astype(np.float64) if ignore_label is not None else \
        np.ones_like(x, dtype=np.float64)
    valid_count = int(np.sum(valid))

    if normalization == "FULL":
        normalizer = float(outer_num * inner_num)
    elif normalization == "VALID":
        normalizer = float(valid_count)
    elif normalization == "BATCH_SIZE":
        normalizer = float(outer_num)
    elif normalization == "NONE":
        normalizer = 1.0
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    normalizer = max(1.0, normalizer)
    return (dx * loss_weight / normalizer).astype(np.float32)


def euclidean_loss_np(x, y):
    """Numpy reference for EuclideanLoss forward scalar loss."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = int(x.shape[0])
    diff = x - y
    return float(np.sum(diff * diff) / (2.0 * N))


def euclidean_backward_np(x, y, loss_weight=1.0):
    """Numpy reference for EuclideanLoss backward (dX, dY)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = int(x.shape[0])
    diff = x - y
    dx = (diff * loss_weight / N).astype(np.float32)
    dy = (-diff * loss_weight / N).astype(np.float32)
    return dx, dy


# ---------------------------------------------------------------------------
# Prototxt builders
# ---------------------------------------------------------------------------

def _make_sce_prototxt(N, C, H=1, W=1, normalization=None, ignore_label=None):
    """Input(data) -> Input(label) -> SigmoidCrossEntropyLoss(loss)."""
    loss_param = ""
    parts = []
    if ignore_label is not None:
        parts.append(f"    ignore_label: {ignore_label}")
    if normalization is not None:
        parts.append(f"    normalization: {normalization}")
    if parts:
        loss_param = "  loss_param {\n" + "\n".join(parts) + "\n  }\n"
    return textwrap.dedent(f"""\
        name: "test_sce"
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
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "loss"
          type: "SigmoidCrossEntropyLoss"
          bottom: "data"
          bottom: "label"
          top: "loss"
{loss_param}        }}
    """)


def _make_euclid_prototxt(N, C, H=1, W=1):
    """Input(data) -> Input(label) -> EuclideanLoss(loss)."""
    return textwrap.dedent(f"""\
        name: "test_euclid"
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
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "loss"
          type: "EuclideanLoss"
          bottom: "data"
          bottom: "label"
          top: "loss"
        }}
    """)


def _forward_loss(net, x, label):
    """Run forward and return the scalar loss (from net.Forward output)."""
    out = net.Forward({"data": x.astype(np.float32), "label": label.astype(np.float32)})
    return float(out["loss"].data.flat[0])


def _run_backward(net, x, label, loss_weight=1.0):
    """Run forward then backward, return (dx, dy) diffs for data/label blobs."""
    net.Forward({"data": x.astype(np.float32), "label": label.astype(np.float32)})
    net.backward({"loss": np.array([loss_weight], dtype=np.float32)})
    dx = net.blob_by_name("data").diff.copy()
    dy = net.blob_by_name("label").diff.copy()
    return dx, dy


# ---------------------------------------------------------------------------
# Numerical gradient helper
# ---------------------------------------------------------------------------

def _num_grad_wrt(net, blob_name, base_inputs, h=EPS_NUMERICAL):
    """Numerical gradient of the scalar loss w.r.t. one input blob.

    base_inputs: dict {blob_name -> float32 ndarray} for all net inputs.
    Returns float32 array of the same shape as the perturbed blob.
    """
    x0 = base_inputs[blob_name]
    grad = np.zeros_like(x0, dtype=np.float64)
    flat_x = x0.astype(np.float32).ravel()
    flat_grad = grad.ravel()

    for i in range(flat_x.size):
        orig = flat_x[i]

        xp = x0.copy()
        xp.ravel()[i] = orig + np.float32(h)
        inputs = dict(base_inputs)
        inputs[blob_name] = xp
        loss_p = _forward_loss(net, inputs["data"], inputs["label"])

        xm = x0.copy()
        xm.ravel()[i] = orig - np.float32(h)
        inputs = dict(base_inputs)
        inputs[blob_name] = xm
        loss_m = _forward_loss(net, inputs["data"], inputs["label"])

        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)

    return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Test Class 1: SigmoidCrossEntropyLoss
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSigmoidCrossEntropyLoss:
    """Forward / backward / branch behavior for SigmoidCrossEntropyLoss."""

    def test_sce_registration(self):
        """Layer instantiates and is registered with the correct type."""
        net = Net(_make_sce_prototxt(2, 3))
        types = [layer.type for layer in net.layers_array()]
        assert "SigmoidCrossEntropyLoss" in types, f"layer not registered: {types}"

    def test_sce_forward_default_batch_size(self):
        """Default normalization (BATCH_SIZE) matches numpy on random data."""
        np.random.seed(0)
        N, C, H, W = 3, 4, 2, 2
        net = Net(_make_sce_prototxt(N, C, H, W))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.rand(N, C, H, W).astype(np.float32)  # targets in [0,1]

        loss_cpp = _forward_loss(net, x, y)
        loss_np = sigmoid_ce_loss_np(x, y, normalization="BATCH_SIZE")
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL,
                                   err_msg="default BATCH_SIZE loss mismatch")

    def test_sce_forward_normalization_modes(self):
        """Each normalization mode (FULL/VALID/BATCH_SIZE/NONE) matches numpy."""
        np.random.seed(1)
        N, C, H, W = 2, 3, 1, 1
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.rand(N, C, H, W).astype(np.float32)

        for mode in ["FULL", "VALID", "BATCH_SIZE", "NONE"]:
            net = Net(_make_sce_prototxt(N, C, H, W, normalization=mode))
            loss_cpp = _forward_loss(net, x, y)
            loss_np = sigmoid_ce_loss_np(x, y, normalization=mode)
            np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL,
                                       err_msg=f"normalization={mode} loss mismatch")

    def test_sce_known_all_half_predictions(self):
        """x=0 logits with y=0.5 targets give per-element loss of ln(2)."""
        # Single element: loss = 0*0.5 - 0*0.5 + log1p(exp(0)) = ln(2)
        net = Net(_make_sce_prototxt(1, 1, 1, 1))
        x = np.zeros((1, 1, 1, 1), dtype=np.float32)
        y = np.full((1, 1, 1, 1), 0.5, dtype=np.float32)
        loss_cpp = _forward_loss(net, x, y)
        np.testing.assert_allclose(loss_cpp, np.log(2.0), rtol=1e-5)

    def test_sce_known_perfect_predictions(self):
        """Perfect sigmoid predictions drive loss ~0."""
        net = Net(_make_sce_prototxt(2, 2, 1, 1))
        # logits strongly positive where target=1, strongly negative where target=0
        x = np.array([[[[10.0, -10.0]]], [[[-10.0, 10.0]]]], dtype=np.float32)
        y = np.array([[[[1.0, 0.0]]], [[[0.0, 1.0]]]], dtype=np.float32)
        loss_cpp = _forward_loss(net, x, y)
        assert loss_cpp < 1e-4, f"Expected near-zero loss, got {loss_cpp}"

    def test_sce_backward_analytical_vs_numpy(self):
        """Analytical dX (batch_size norm) vs numpy reference on random data."""
        np.random.seed(2)
        N, C, H, W = 3, 4, 2, 2
        net = Net(_make_sce_prototxt(N, C, H, W))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.rand(N, C, H, W).astype(np.float32)

        dx, _ = _run_backward(net, x, y)
        dx_np = sigmoid_ce_backward_np(x, y, normalization="BATCH_SIZE")
        np.testing.assert_allclose(dx, dx_np, rtol=RTOL, atol=ATOL,
                                   err_msg="dX mismatch vs numpy reference")

    def test_sce_backward_label_has_no_gradient(self):
        """SigmoidCrossEntropyLoss does not backprop to the label input."""
        np.random.seed(3)
        N, C = 2, 3
        net = Net(_make_sce_prototxt(N, C))
        x = np.random.randn(N, C, 1, 1).astype(np.float32)
        y = np.random.rand(N, C, 1, 1).astype(np.float32)
        _, dy = _run_backward(net, x, y)
        np.testing.assert_allclose(dy, 0.0, atol=1e-6,
                                   err_msg="label bottom should receive no gradient")

    def test_sce_numerical_gradient(self):
        """Numerical gradient of loss vs analytical dX (small tensor)."""
        np.random.seed(4)
        N, C = 2, 3
        net = Net(_make_sce_prototxt(N, C))
        x = np.random.randn(N, C, 1, 1).astype(np.float32) * 0.5
        y = np.random.rand(N, C, 1, 1).astype(np.float32)

        dx, _ = _run_backward(net, x, y)
        dx_num = _num_grad_wrt(net, "data", {"data": x, "label": y})
        np.testing.assert_allclose(dx, dx_num, rtol=1e-2, atol=1e-3,
                                   err_msg="numerical gradient check failed for dX")

    def test_sce_ignore_label_forward(self):
        """Ignored elements are excluded from the loss sum and the VALID normalizer."""
        np.random.seed(5)
        N, C, H, W = 2, 3, 1, 1
        ignore_label = -1
        net = Net(_make_sce_prototxt(N, C, H, W, normalization="VALID",
                                     ignore_label=ignore_label))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.rand(N, C, H, W).astype(np.float32)
        # Force a few elements to the ignore label target
        y = y.copy()
        y[0, 0, 0, 0] = float(ignore_label)
        y[1, 2, 0, 0] = float(ignore_label)

        loss_cpp = _forward_loss(net, x, y)
        loss_np = sigmoid_ce_loss_np(x, y, normalization="VALID",
                                     ignore_label=ignore_label)
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL,
                                   err_msg="ignore_label VALID loss mismatch")

    def test_sce_ignore_label_backward(self):
        """Ignored elements have zero gradient in dX."""
        np.random.seed(6)
        N, C, H, W = 2, 3, 1, 1
        ignore_label = -1
        net = Net(_make_sce_prototxt(N, C, H, W, normalization="VALID",
                                     ignore_label=ignore_label))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.rand(N, C, H, W).astype(np.float32)
        y = y.copy()
        y[0, 0, 0, 0] = float(ignore_label)
        y[1, 2, 0, 0] = float(ignore_label)

        dx, _ = _run_backward(net, x, y)
        dx_np = sigmoid_ce_backward_np(x, y, normalization="VALID",
                                       ignore_label=ignore_label)
        np.testing.assert_allclose(dx, dx_np, rtol=RTOL, atol=ATOL,
                                   err_msg="ignore_label dX mismatch")
        # Ignored positions must be exactly zero
        assert dx[0, 0, 0, 0] == 0.0
        assert dx[1, 2, 0, 0] == 0.0

    def test_sce_loss_weight_scaling(self):
        """Gradient scales linearly with loss_weight."""
        np.random.seed(7)
        N, C = 3, 4
        net = Net(_make_sce_prototxt(N, C))
        x = np.random.randn(N, C, 1, 1).astype(np.float32)
        y = np.random.rand(N, C, 1, 1).astype(np.float32)

        dx1, _ = _run_backward(net, x, y, loss_weight=1.0)
        dx2, _ = _run_backward(net, x, y, loss_weight=2.0)
        np.testing.assert_allclose(dx2, 2.0 * dx1, rtol=1e-5,
                                   err_msg="dX should scale linearly with loss_weight")

    def test_sce_deterministic(self):
        """Two identical forward passes give identical loss."""
        np.random.seed(8)
        N, C, H, W = 2, 4, 2, 2
        net = Net(_make_sce_prototxt(N, C, H, W))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.rand(N, C, H, W).astype(np.float32)
        l1 = _forward_loss(net, x, y)
        l2 = _forward_loss(net, x, y)
        np.testing.assert_allclose(l1, l2, rtol=1e-6)

    def test_sce_finite(self):
        """Forward loss is finite for wide-range logits."""
        np.random.seed(9)
        N, C = 4, 8
        net = Net(_make_sce_prototxt(N, C))
        x = np.random.randn(N, C, 1, 1).astype(np.float32) * 5.0
        y = np.random.rand(N, C, 1, 1).astype(np.float32)
        loss_cpp = _forward_loss(net, x, y)
        assert np.isfinite(loss_cpp), f"loss is not finite: {loss_cpp}"


# ---------------------------------------------------------------------------
# Test Class 2: EuclideanLoss
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestEuclideanLoss:
    """Forward / backward / batch behavior for EuclideanLoss."""

    def test_euclid_registration(self):
        """Layer instantiates and is registered with the correct type."""
        net = Net(_make_euclid_prototxt(2, 3))
        types = [layer.type for layer in net.layers_array()]
        assert "EuclideanLoss" in types, f"layer not registered: {types}"

    def test_euclid_forward_known(self):
        """Forward matches ||x-y||^2 / (2N) on random data."""
        np.random.seed(10)
        N, C, H, W = 3, 2, 2, 2
        net = Net(_make_euclid_prototxt(N, C, H, W))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.randn(N, C, H, W).astype(np.float32)

        loss_cpp = _forward_loss(net, x, y)
        loss_np = euclidean_loss_np(x, y)
        np.testing.assert_allclose(loss_cpp, loss_np, rtol=RTOL,
                                   err_msg="EuclideanLoss forward mismatch")

    def test_euclid_forward_perfect_prediction(self):
        """x == y gives zero loss."""
        np.random.seed(11)
        N, C, H, W = 2, 3, 2, 2
        net = Net(_make_euclid_prototxt(N, C, H, W))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        loss_cpp = _forward_loss(net, x, x)
        np.testing.assert_allclose(loss_cpp, 0.0, atol=1e-6,
                                   err_msg="x==y should give zero loss")

    def test_euclid_forward_batch_scaling(self):
        """Loss is the batch mean of per-sample L2 losses (1/N scaling)."""
        C = 2
        # Two distinct samples with known per-sample half-squared losses.
        xa = np.array([[[[1.0, 2.0]]]], dtype=np.float32)  # sample 0
        ya = np.array([[[[0.0, 0.0]]]], dtype=np.float32)
        xb = np.array([[[[3.0, 4.0]]]], dtype=np.float32)  # sample 1
        yb = np.array([[[[3.0, 3.0]]]], dtype=np.float32)

        # per-sample losses:
        #   loss_a = ||xa-ya||^2 / 2 = (1+4)/2 = 2.5
        #   loss_b = ||xb-yb||^2 / 2 = (0+1)/2 = 0.5
        # batch mean = (2.5 + 0.5) / 2 = 1.5
        x = np.concatenate([xa, xb], axis=0)
        y = np.concatenate([ya, yb], axis=0)
        net = Net(_make_euclid_prototxt(2, C, 1, 1))
        loss_cpp = _forward_loss(net, x, y)
        np.testing.assert_allclose(loss_cpp, 1.5, rtol=1e-5,
                                   err_msg="batch loss should be the per-sample mean")

        # Identical samples across the batch keep the batch-mean loss unchanged
        # (both the squared-diff sum and N scale together).
        net1 = Net(_make_euclid_prototxt(1, C, 1, 1))
        loss1 = _forward_loss(net1, xa, ya)
        net3 = Net(_make_euclid_prototxt(3, C, 1, 1))
        x3 = np.tile(xa, (3, 1, 1, 1))
        y3 = np.tile(ya, (3, 1, 1, 1))
        loss3 = _forward_loss(net3, x3, y3)
        np.testing.assert_allclose(loss3, loss1, rtol=1e-5,
                                   err_msg="duplicating samples should preserve the mean loss")

    def test_euclid_backward_analytical_vs_numpy(self):
        """Analytical (dX, dY) vs numpy reference on random data."""
        np.random.seed(12)
        N, C, H, W = 3, 4, 2, 2
        net = Net(_make_euclid_prototxt(N, C, H, W))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.randn(N, C, H, W).astype(np.float32)

        dx, dy = _run_backward(net, x, y)
        dx_np, dy_np = euclidean_backward_np(x, y)
        np.testing.assert_allclose(dx, dx_np, rtol=RTOL, atol=ATOL,
                                   err_msg="Euclidean dX mismatch")
        np.testing.assert_allclose(dy, dy_np, rtol=RTOL, atol=ATOL,
                                   err_msg="Euclidean dY mismatch")

    def test_euclid_numerical_gradient(self):
        """Numerical gradient of loss vs analytical dX and dY."""
        np.random.seed(13)
        N, C = 2, 3
        net = Net(_make_euclid_prototxt(N, C))
        x = np.random.randn(N, C, 1, 1).astype(np.float32)
        y = np.random.randn(N, C, 1, 1).astype(np.float32)

        dx, dy = _run_backward(net, x, y)
        base = {"data": x, "label": y}
        dx_num = _num_grad_wrt(net, "data", base)
        dy_num = _num_grad_wrt(net, "label", base)
        np.testing.assert_allclose(dx, dx_num, rtol=1e-2, atol=1e-3,
                                   err_msg="numerical gradient check failed for dX")
        np.testing.assert_allclose(dy, dy_num, rtol=1e-2, atol=1e-3,
                                   err_msg="numerical gradient check failed for dY")

    def test_euclid_loss_weight_scaling(self):
        """Gradients scale linearly with loss_weight."""
        np.random.seed(14)
        N, C = 3, 4
        net = Net(_make_euclid_prototxt(N, C))
        x = np.random.randn(N, C, 1, 1).astype(np.float32)
        y = np.random.randn(N, C, 1, 1).astype(np.float32)

        dx1, dy1 = _run_backward(net, x, y, loss_weight=1.0)
        dx2, dy2 = _run_backward(net, x, y, loss_weight=2.0)
        np.testing.assert_allclose(dx2, 2.0 * dx1, rtol=1e-5)
        np.testing.assert_allclose(dy2, 2.0 * dy1, rtol=1e-5)

    def test_euclid_deterministic(self):
        """Two identical forward passes give identical loss."""
        np.random.seed(15)
        N, C, H, W = 2, 4, 2, 2
        net = Net(_make_euclid_prototxt(N, C, H, W))
        x = np.random.randn(N, C, H, W).astype(np.float32)
        y = np.random.randn(N, C, H, W).astype(np.float32)
        l1 = _forward_loss(net, x, y)
        l2 = _forward_loss(net, x, y)
        np.testing.assert_allclose(l1, l2, rtol=1e-6)

    def test_euclid_finite(self):
        """Forward loss is finite for random inputs."""
        np.random.seed(16)
        N, C = 4, 8
        net = Net(_make_euclid_prototxt(N, C))
        x = np.random.randn(N, C, 1, 1).astype(np.float32) * 3.0
        y = np.random.randn(N, C, 1, 1).astype(np.float32) * 3.0
        loss_cpp = _forward_loss(net, x, y)
        assert np.isfinite(loss_cpp), f"loss is not finite: {loss_cpp}"