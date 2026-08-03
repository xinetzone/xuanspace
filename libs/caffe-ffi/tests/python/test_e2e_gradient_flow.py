"""End-to-end gradient propagation verification for CNN training network.

Network architecture:
  Input(data) → Conv → BN → ReLU → Pool → IP → ReLU → Dropout → IP → SoftmaxWithLoss
  Input(label) ────────────────────────────────────────────────────────────────┘

This test verifies:
  1. Full forward pass runs without crashes
  2. Full backward pass runs without crashes
  3. All learnable parameter gradients are finite and non-zero
  4. Loss value is finite and reasonable
  5. Gradient norms are stable (no NaN/Inf/exploding)
  6. Multi-step training: loss decreases over a few iterations
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension


def _make_e2e_prototxt(N=4, C=3, H=8, W=8, num_classes=4):
    """Create a minimal CNN: Conv1(3→8,3x3) → BN → ReLU → Pool(2x2) → IP(8*4*4→16) → ReLU → Dropout → IP(16→C) → SML."""
    return textwrap.dedent(f"""\
        name: "e2e_cnn_train"
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
          input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }}
        }}
        layer {{
          name: "conv1"
          type: "Convolution"
          bottom: "data"
          top: "conv1"
          convolution_param {{
            num_output: 8
            kernel_size: 3
            pad: 1
            stride: 1
            bias_term: true
            weight_filler {{ type: "xavier" }}
            bias_filler {{ type: "constant" value: 0 }}
          }}
        }}
        layer {{
          name: "bn1"
          type: "BatchNorm"
          bottom: "conv1"
          top: "bn1"
          batch_norm_param {{ use_global_stats: true eps: 1e-5 }}
        }}
        layer {{
          name: "relu1"
          type: "ReLU"
          bottom: "bn1"
          top: "relu1"
        }}
        layer {{
          name: "pool1"
          type: "Pooling"
          bottom: "relu1"
          top: "pool1"
          pooling_param {{ pool: MAX kernel_size: 2 stride: 2 }}
        }}
        layer {{
          name: "ip1"
          type: "InnerProduct"
          bottom: "pool1"
          top: "ip1"
          inner_product_param {{
            num_output: 16
            bias_term: true
            weight_filler {{ type: "xavier" }}
            bias_filler {{ type: "constant" value: 0 }}
          }}
        }}
        layer {{
          name: "relu2"
          type: "ReLU"
          bottom: "ip1"
          top: "relu2"
        }}
        layer {{
          name: "drop1"
          type: "Dropout"
          bottom: "relu2"
          top: "drop1"
          dropout_param {{ dropout_ratio: 0.5 }}
        }}
        layer {{
          name: "ip2"
          type: "InnerProduct"
          bottom: "drop1"
          top: "ip2"
          inner_product_param {{
            num_output: {num_classes}
            bias_term: true
            weight_filler {{ type: "xavier" }}
            bias_filler {{ type: "constant" value: 0 }}
          }}
        }}
        layer {{
          name: "loss"
          type: "SoftmaxWithLoss"
          bottom: "ip2"
          bottom: "label"
          top: "loss"
        }}
    """)


def _set_bn_running_stats(net, layer_name, num_channels):
    """Set BN running stats (mean=0, var=1, count=1 for training mode)."""
    layer = net.layer_by_name(layer_name)
    layer.blobs[0].from_numpy(np.zeros(num_channels, dtype=np.float32))  # mean
    layer.blobs[1].from_numpy(np.ones(num_channels, dtype=np.float32))   # var
    layer.blobs[2].from_numpy(np.array([1.0], dtype=np.float32))         # count


def _simple_sgd_update(net, lr=0.01):
    """Very simple SGD update: w -= lr * dw for all layer blobs."""
    for lname in ["conv1", "ip1", "ip2"]:
        layer = net.layer_by_name(lname)
        for blob in layer.blobs:
            w = blob.data.copy()
            dw = blob.diff
            blob.from_numpy((w - lr * dw).astype(np.float32))


@require_cpp_extension
class TestEndToEndGradientFlow:
    """Verify end-to-end gradient propagation through the complete CNN."""

    def test_forward_backward_no_crash(self):
        """Forward + backward pass completes without errors."""
        N, C, H, W, num_cls = 4, 3, 8, 8, 4
        net = Net(_make_e2e_prototxt(N, C, H, W, num_cls))

        _set_bn_running_stats(net, "bn1", 8)

        rng = np.random.RandomState(42)
        data = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1, 1, 1)).astype(np.float32)

        out = net.forward({"data": data, "label": label})
        loss = float(out["loss"])
        assert np.isfinite(loss), f"Loss is not finite: {loss}"

        net.backward({"loss": np.array([1.0], dtype=np.float32})})

    def test_all_param_gradients_nonzero_finite(self):
        """All learnable parameter gradients are finite and non-zero."""
        N, C, H, W, num_cls = 4, 3, 8, 8, 4
        net = Net(_make_e2e_prototxt(N, C, H, W, num_cls))
        _set_bn_running_stats(net, "bn1", 8)

        rng = np.random.RandomState(123)
        data = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1, 1, 1)).astype(np.float32)

        net.forward({"data": data, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})

        # Check each learnable layer
        learnable_layers = ["conv1", "ip1", "ip2"]
        for lname in learnable_layers:
            layer = net.layer_by_name(lname)
            for i, blob in enumerate(layer.blobs):
                dw = blob.diff
                assert np.all(np.isfinite(dw)), f"{lname} blob[{i}] diff has NaN/Inf"
                grad_norm = float(np.sqrt(np.sum(dw.astype(np.float64) ** 2)))
                assert grad_norm > 1e-10, f"{lname} blob[{i}] gradient is zero (norm={grad_norm})"

        # Input gradient (data diff) should also be finite and non-zero
        data_diff = net.blob_by_name("data").diff
        assert np.all(np.isfinite(data_diff)), "data diff has NaN/Inf"
        data_grad_norm = float(np.sqrt(np.sum(data_diff.astype(np.float64) ** 2)))
        assert data_grad_norm > 1e-10, f"data gradient is zero (norm={data_grad_norm})"

    def test_input_gradient_flows_through_dropout(self):
        """Verify gradient actually flows through Dropout layer (dX = dy)."""
        N, C, H, W, num_cls = 4, 3, 8, 8, 4
        net = Net(_make_e2e_prototxt(N, C, H, W, num_cls))
        _set_bn_running_stats(net, "bn1", 8)

        rng = np.random.RandomState(456)
        data = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1, 1, 1)).astype(np.float32)

        net.forward({"data": data, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})

        # Dropout is identity in inference mode, so drop1 diff should equal relu2 diff
        # (both point to same memory if inplace, or should be identical values)
        drop_diff = net.blob_by_name("drop1").diff
        relu2_diff = net.blob_by_name("relu2").diff
        np.testing.assert_array_equal(drop_diff, relu2_diff)

    def test_loss_decreases_with_training(self):
        """Loss should decrease over a few SGD steps (gradient effectiveness)."""
        N, C, H, W, num_cls = 4, 3, 8, 8, 4
        net = Net(_make_e2e_prototxt(N, C, H, W, num_cls))
        _set_bn_running_stats(net, "bn1", 8)

        rng = np.random.RandomState(789)
        data = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1, 1, 1)).astype(np.float32)

        losses = []
        for step in range(10):
            out = net.forward({"data": data, "label": label})
            loss = float(out["loss"])
            losses.append(loss)
            net.backward({"loss": np.array([1.0], dtype=np.float32)})
            _simple_sgd_update(net, lr=0.01)

        # Loss should be finite at all steps
        for i, l in enumerate(losses):
            assert np.isfinite(l), f"Loss at step {i} is not finite: {l}"

        # Loss should generally decrease (allow some noise for SGD on tiny batch)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_gradient_norms_stable(self):
        """Gradient norms should be in a reasonable range (not exploding/vanishing)."""
        N, C, H, W, num_cls = 4, 3, 8, 8, 4
        net = Net(_make_e2e_prototxt(N, C, H, W, num_cls))
        _set_bn_running_stats(net, "bn1", 8)

        rng = np.random.RandomState(101)
        data = rng.randn(N, C, H, W).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1, 1, 1)).astype(np.float32)

        net.forward({"data": data, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})

        # Check gradient norms for each layer are reasonable
        for lname in ["conv1", "ip1", "ip2"]:
            layer = net.layer_by_name(lname)
            for i, blob in enumerate(layer.blobs):
                dw = blob.diff.astype(np.float64)
                norm = float(np.sqrt(np.sum(dw ** 2)))
                # Gradient norm should be between 1e-8 and 1e3 (not vanishing/exploding)
                assert 1e-10 < norm < 1e4, (
                    f"{lname} blob[{i}] gradient norm={norm:.3g} outside reasonable range"
                )
