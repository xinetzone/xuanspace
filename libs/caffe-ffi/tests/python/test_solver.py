"""Tests for the caffe-ffi training-engineering layer (``caffe_ffi.solver``).

Covers the four building blocks of Task 33 (P4-训练工程化):

1. **Optimizers** — ``SGD`` / ``Adam`` update rules verified against independent
   numpy reference implementations (plain momentum, Nesterov, Adam bias
   correction, L2 weight decay).
2. **Learning-rate schedulers** — ``StepLR`` / ``MultiStepLR`` /
   ``ExponentialLR`` / ``CosineAnnealingLR`` math and the ``optimizer.lr``
   write-back contract.
3. **Optimizer checkpointing** — ``state_dict`` / ``load_state_dict`` round-trips.
4. **Solver** (native only) — a full ``fit`` loop on a tiny MLP + SoftmaxWithLoss
   verifies loss decreases, history is populated, and validate/train-mode toggling
   behaves.

The pure-Python optimizer/scheduler tests use a lightweight single-learnable-blob
net stub (built on real ``Blob`` objects) so they run identically with and without
the C++ extension. The end-to-end ``Solver`` loop requires the native extension
and is gated with ``@require_cpp_extension``.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from caffe_ffi import Blob
from caffe_ffi.solver import (
    Adam,
    CosineAnnealingLR,
    ExponentialLR,
    LRScheduler,
    MultiStepLR,
    SGD,
    StepLR,
    Solver,
)

from .conftest import require_cpp_extension


# ---------------------------------------------------------------------------
# Helpers: a single-learnable-blob net stub backed by real Blob objects
# ---------------------------------------------------------------------------

def make_single_param_net(shape):
    """Return ``(net, blob)`` where ``net`` has one learnable layer ``fc``.

    ``net.layers_array()`` yields a single layer named ``fc`` whose ``blobs`` is
    a one-element list holding a real :class:`Blob`. This is enough for the
    optimizer update math to run identically in native and stub modes.
    """
    blob = Blob(list(shape))
    layer = types.SimpleNamespace(name="fc", type="InnerProduct", blobs=[blob])
    net = types.SimpleNamespace(layers_array=lambda: [layer])
    return net, blob


def make_two_param_net(shape):
    """Two learnable layers (``fc0``/``fc1``) to exercise multi-param state dicts."""
    b0 = Blob(list(shape))
    b1 = Blob(list(shape))
    l0 = types.SimpleNamespace(name="fc0", type="InnerProduct", blobs=[b0])
    l1 = types.SimpleNamespace(name="fc1", type="InnerProduct", blobs=[b1])
    net = types.SimpleNamespace(layers_array=lambda: [l0, l1])
    return net, b0, b1


def make_optimizer_dummy():
    """A minimal optimizer-like object for scheduler tests (only ``lr`` matters)."""
    return types.SimpleNamespace(lr=0.1)


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class TestSGD:
    def test_no_momentum_plain_sgd(self):
        net, blob = make_single_param_net((2, 3))
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        diff = np.full((2, 3), 0.5, dtype=np.float32)
        blob.data_tensor[:] = data
        blob.diff_tensor[:] = diff

        opt = SGD(lr=0.1)
        opt.step(net)

        expected = data - 0.1 * diff
        np.testing.assert_allclose(blob.data_tensor, expected, rtol=1e-6)

    def test_weight_decay_adds_l2_to_grad(self):
        net, blob = make_single_param_net((4,))
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        diff = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        blob.data_tensor[:] = data
        blob.diff_tensor[:] = diff

        wd = 0.5
        opt = SGD(lr=0.1, weight_decay=wd)
        opt.step(net)

        # effective grad = diff + wd * data
        expected = data - 0.1 * (diff + wd * data)
        np.testing.assert_allclose(blob.data_tensor, expected, rtol=1e-6)

    def test_momentum_matches_numpy_reference(self):
        net, blob = make_single_param_net((3,))
        data = np.array([1.0, -1.0, 2.0], dtype=np.float32)
        diff = np.array([0.2, -0.3, 0.4], dtype=np.float32)
        blob.data_tensor[:] = data
        blob.diff_tensor[:] = diff

        opt = SGD(lr=0.1, momentum=0.9)
        v = np.zeros(3, dtype=np.float32)
        expected = data.copy()
        for _ in range(3):
            opt.step(net)
            v = 0.9 * v - 0.1 * diff
            expected += v
            np.testing.assert_allclose(blob.data_tensor, expected, rtol=1e-6)

    def test_nesterov_matches_numpy_reference(self):
        net, blob = make_single_param_net((3,))
        blob.data_tensor[:] = np.array([1.0, -1.0, 2.0], dtype=np.float32)
        blob.diff_tensor[:] = np.array([0.2, -0.3, 0.4], dtype=np.float32)

        opt = SGD(lr=0.1, momentum=0.9, nesterov=True)
        v = np.zeros(3, dtype=np.float32)
        init = np.array([1.0, -1.0, 2.0], dtype=np.float32)
        for _ in range(3):
            grad = blob.diff_tensor.copy()
            opt.step(net)
            v_prev = v.copy()
            v = 0.9 * v - 0.1 * grad
            init = init - 0.9 * v_prev + (1 + 0.9) * v
            np.testing.assert_allclose(blob.data_tensor, init, rtol=1e-6)

    def test_state_dict_roundtrip(self):
        net, blob = make_single_param_net((2, 2))
        blob.data_tensor[:] = np.arange(4, dtype=np.float32).reshape(2, 2)
        blob.diff_tensor[:] = np.ones((2, 2), dtype=np.float32)

        opt = SGD(lr=0.1, momentum=0.9)
        opt.step(net)
        state = opt.state_dict()
        assert "fc:0" in state

        opt2 = SGD(lr=0.1, momentum=0.9)
        opt2.load_state_dict(state)
        np.testing.assert_array_equal(opt2._velocity[("fc", 0)], opt._velocity[("fc", 0)])

    def test_zero_grad_clears_diff(self):
        net, blob = make_single_param_net((2,))
        blob.diff_tensor[:] = np.array([5.0, 5.0], dtype=np.float32)
        SGD(lr=0.1).zero_grad(net)
        np.testing.assert_array_equal(blob.diff_tensor, np.zeros(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class TestAdam:
    def test_matches_numpy_reference(self):
        net, blob = make_single_param_net((3,))
        blob.data_tensor[:] = np.array([1.0, -1.0, 2.0], dtype=np.float32)
        blob.diff_tensor[:] = np.array([0.2, -0.3, 0.4], dtype=np.float32)

        opt = Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        m = np.zeros(3, dtype=np.float32)
        v = np.zeros(3, dtype=np.float32)
        data = blob.data_tensor.copy()
        for t in range(1, 4):
            grad = blob.diff_tensor.copy()
            opt.step(net)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad * grad)
            mhat = m / (1 - 0.9 ** t)
            vhat = v / (1 - 0.999 ** t)
            data = data - 0.001 * mhat / (np.sqrt(vhat) + 1e-8)
            np.testing.assert_allclose(blob.data_tensor, data, rtol=1e-5)

    def test_state_dict_roundtrip(self):
        net, b0, b1 = make_two_param_net((2,))
        b0.data_tensor[:] = np.array([1.0, 2.0], dtype=np.float32)
        b0.diff_tensor[:] = np.array([0.1, 0.1], dtype=np.float32)
        b1.data_tensor[:] = np.array([3.0, 4.0], dtype=np.float32)
        b1.diff_tensor[:] = np.array([0.2, 0.2], dtype=np.float32)

        opt = Adam(lr=0.001)
        opt.step(net)
        state = opt.state_dict()
        assert state["t"] == 1
        assert "fc0:0" in state["m"] and "fc1:0" in state["m"]

        opt2 = Adam(lr=0.001)
        opt2.load_state_dict(state)
        assert opt2._t == 1
        np.testing.assert_array_equal(opt2._m[("fc0", 0)], opt._m[("fc0", 0)])


# ---------------------------------------------------------------------------
# Learning-rate schedulers
# ---------------------------------------------------------------------------

class TestLRScheduler:
    def test_step_lr(self):
        opt = make_optimizer_dummy()
        sched = StepLR(opt, step_size=3, gamma=0.1)
        assert sched.step() == pytest.approx(0.1)
        assert sched.step() == pytest.approx(0.1)
        assert sched.step() == pytest.approx(0.1)
        assert sched.step() == pytest.approx(0.01)
        assert opt.lr == pytest.approx(0.01)

    def test_multistep_lr(self):
        opt = make_optimizer_dummy()
        sched = MultiStepLR(opt, milestones=[2, 5], gamma=0.5)
        expected = {0: 0.1, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.025, 6: 0.025}
        for epoch in range(7):
            assert sched.step() == pytest.approx(expected[epoch])

    def test_exponential_lr(self):
        opt = make_optimizer_dummy()
        sched = ExponentialLR(opt, gamma=0.9)
        for epoch in range(5):
            assert sched.step() == pytest.approx(0.1 * 0.9 ** epoch)

    def test_cosine_annealing_lr(self):
        opt = make_optimizer_dummy()
        sched = CosineAnnealingLR(opt, T_max=4, eta_min=0.0)
        for epoch in range(4):
            expected = 0.1 * (1 + math.cos(math.pi * epoch / 4)) / 2
            assert sched.step() == pytest.approx(expected, abs=1e-7)

    def test_base_class_requires_get_lr(self):
        opt = make_optimizer_dummy()
        sched = LRScheduler(opt)
        with pytest.raises(NotImplementedError):
            sched.get_lr()

    def test_writes_back_to_optimizer_lr(self):
        opt = make_optimizer_dummy()
        sched = StepLR(opt, step_size=1, gamma=0.5)
        sched.step()  # epoch 0 -> base_lr (0.1)
        assert opt.lr == pytest.approx(0.1)
        sched.step()  # epoch 1 -> 0.1 * 0.5
        assert opt.lr == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Solver (end-to-end, requires native C++ extension)
# ---------------------------------------------------------------------------

TINY_MLP_PROTO = '''
name: "tiny_mlp"
layer { name: "data" type: "Input" top: "data"
  input_param { shape { dim: 4 dim: 8 } } }
layer { name: "label" type: "Input" top: "label"
  input_param { shape { dim: 4 dim: 1 } } }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1"
  inner_product_param { num_output: 16 weight_filler { type: "msra" } } }
layer { name: "relu1" type: "ReLU" bottom: "fc1" top: "fc1" }
layer { name: "fc2" type: "InnerProduct" bottom: "fc1" top: "fc2"
  inner_product_param { num_output: 3 weight_filler { type: "msra" } } }
layer { name: "loss" type: "SoftmaxWithLoss" bottom: "fc2" bottom: "label" top: "loss" }
'''


class TestSolver:
    @require_cpp_extension
    def test_fit_decreases_loss_and_populates_history(self):
        from caffe_ffi import net_from_param, net_param_from_string

        net = net_from_param(net_param_from_string(TINY_MLP_PROTO))
        opt = SGD(lr=0.05, momentum=0.9)
        solver = Solver(net, opt, loss_blob="loss")

        rng = np.random.RandomState(0)

        def batches():
            for _ in range(20):
                yield {
                    "data": rng.randn(4, 8).astype(np.float32),
                    "label": rng.randint(0, 3, size=(4, 1)).astype(np.float32),
                }

        history = solver.fit(batches, epochs=2, log_interval=0)
        assert len(history["loss"]) == 40
        assert all(np.isfinite(l) for l in history["loss"])
        assert history["loss"][-1] < history["loss"][0]

    @require_cpp_extension
    def test_step_returns_loss_and_optimizer_updates_weights(self):
        from caffe_ffi import net_from_param, net_param_from_string

        net = net_from_param(net_param_from_string(TINY_MLP_PROTO))
        opt = SGD(lr=0.01)
        solver = Solver(net, opt, loss_blob="loss")

        rng = np.random.RandomState(0)
        # Break symmetry caused by msra filler stub (which initializes weights
        # to constant 1.0); with non-trivial weights, gradients are non-zero.
        for layer in net.layers_array():
            for blob in layer.blobs:
                blob.data_tensor[:] = rng.randn(*blob.shape).astype(np.float32) * 0.1

        weights_before = solver.net.layer_by_name("fc1").blobs[0].data.copy()
        loss = solver.step({
            "data": rng.randn(4, 8).astype(np.float32),
            "label": rng.randint(0, 3, size=(4, 1)).astype(np.float32),
        })
        assert np.isfinite(loss)
        weights_after = solver.net.layer_by_name("fc1").blobs[0].data
        assert not np.array_equal(weights_before, weights_after)

    @require_cpp_extension
    def test_scheduler_stepped_per_epoch(self):
        from caffe_ffi import net_from_param, net_param_from_string

        net = net_from_param(net_param_from_string(TINY_MLP_PROTO))
        opt = SGD(lr=0.1)
        sched = StepLR(opt, step_size=1, gamma=0.5)
        solver = Solver(net, opt, loss_blob="loss", scheduler=sched)

        rng = np.random.RandomState(1)

        def batches():
            for _ in range(5):
                yield {
                    "data": rng.randn(4, 8).astype(np.float32),
                    "label": rng.randint(0, 3, size=(4, 1)).astype(np.float32),
                }

        solver.fit(batches, epochs=2, log_interval=0)
        assert opt.lr == pytest.approx(0.025)  # 0.1 * 0.5^2
        assert solver.history["lr"] == pytest.approx([0.05, 0.025])

    @require_cpp_extension
    def test_validate_returns_metric(self):
        from caffe_ffi import net_from_param, net_param_from_string

        net = net_from_param(net_param_from_string(TINY_MLP_PROTO))
        solver = Solver(net, SGD(lr=0.01), loss_blob="loss")

        val_batches = [
            {"data": np.zeros((4, 8), dtype=np.float32), "label": np.zeros((4, 1), dtype=np.float32)}
            for _ in range(3)
        ]
        score = solver.validate(val_batches)
        assert np.isfinite(score)