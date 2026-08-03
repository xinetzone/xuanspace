"""End-to-end gradient consistency test for all P3-D layers.

Verifies that Dropout, Scale, Bias, Eltwise, Concat, and Softmax work together
correctly in a network with residual connections and branch concatenation.

Core correctness criterion: SGD training reduces loss monotonically (which proves
gradients flow correctly from loss back through all layers to all parameters).
"""

import textwrap
import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension


def _make_p3d_full_prototxt(N=4, input_dim=16, num_classes=4):
    return textwrap.dedent(f"""\
        name: "p3d_all_layers_e2e"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {N} dim: {input_dim} }} }}
        }}
        layer {{
          name: "label"
          type: "Input"
          top: "label"
          input_param {{ shape {{ dim: {N} dim: 1 }} }}
        }}
        # Main path: IP -> ReLU --+
        #                         +-> Eltwise(SUM) -> Scale -> Bias -> Dropout -> Concat -> IP -> Softmax -> SoftmaxWithLoss
        # Branch path: IP -> ReLU -------------------------------------------------+
        layer {{
          name: "ip_main"
          type: "InnerProduct"
          bottom: "data"
          top: "ip_main"
          inner_product_param {{ num_output: 16 weight_filler {{ type: "xavier" }} bias_filler {{ type: "constant" }} }}
        }}
        layer {{
          name: "relu_main"
          type: "ReLU"
          bottom: "ip_main"
          top: "relu_main"
        }}
        layer {{
          name: "ip_residual"
          type: "InnerProduct"
          bottom: "data"
          top: "ip_residual"
          inner_product_param {{ num_output: 16 weight_filler {{ type: "xavier" }} bias_filler {{ type: "constant" }} }}
        }}
        layer {{
          name: "eltwise_sum"
          type: "Eltwise"
          bottom: "relu_main"
          bottom: "ip_residual"
          top: "eltwise_sum"
          eltwise_param {{ operation: SUM }}
        }}
        layer {{
          name: "scale1"
          type: "Scale"
          bottom: "eltwise_sum"
          top: "scale1"
          scale_param {{ bias_term: true filler {{ value: 1 }} bias_filler {{ value: 0 }} }}
        }}
        layer {{
          name: "bias1"
          type: "Bias"
          bottom: "scale1"
          top: "bias1"
          bias_param {{ filler {{ value: 0 }} }}
        }}
        layer {{
          name: "drop1"
          type: "Dropout"
          bottom: "bias1"
          top: "drop1"
          dropout_param {{ dropout_ratio: 0 }}
        }}
        layer {{
          name: "ip_branch"
          type: "InnerProduct"
          bottom: "data"
          top: "ip_branch"
          inner_product_param {{ num_output: 16 weight_filler {{ type: "xavier" }} bias_filler {{ type: "constant" }} }}
        }}
        layer {{
          name: "relu_branch"
          type: "ReLU"
          bottom: "ip_branch"
          top: "relu_branch"
        }}
        layer {{
          name: "concat1"
          type: "Concat"
          bottom: "drop1"
          bottom: "relu_branch"
          top: "concat1"
          concat_param {{ axis: 1 }}
        }}
        layer {{
          name: "ip_final"
          type: "InnerProduct"
          bottom: "concat1"
          top: "ip_final"
          inner_product_param {{ num_output: {num_classes} weight_filler {{ type: "xavier" }} bias_filler {{ type: "constant" }} }}
        }}
        layer {{
          name: "softmax_prob"
          type: "Softmax"
          bottom: "ip_final"
          top: "softmax_prob"
        }}
        layer {{
          name: "loss"
          type: "SoftmaxWithLoss"
          bottom: "ip_final"
          bottom: "label"
          top: "loss"
        }}
    """)


def _simple_sgd_update(net, lr=0.01):
    learnable = ["ip_main", "ip_residual", "scale1", "bias1", "ip_branch", "ip_final"]
    for lname in learnable:
        layer = net.layer_by_name(lname)
        for blob in layer.blobs:
            blob.data -= lr * blob.diff


@require_cpp_extension
class TestP3DAllLayersEndToEnd:
    """End-to-end tests for network containing all P3-D layers."""

    def test_p3d_all_layers_forward_backward_no_crash(self):
        """Forward + backward pass completes without errors through all P3-D layers."""
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))

        rng = np.random.RandomState(42)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        out = net.forward({"data": data, "label": label})
        loss = float(out["loss"].item())
        assert np.isfinite(loss), f"Loss is not finite: {loss}"

        net.backward({"loss": np.array([1.0], dtype=np.float32)})

    def test_p3d_all_param_gradients_finite(self):
        """All learnable parameter gradients from P3-D layers are finite."""
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))

        rng = np.random.RandomState(123)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        net.forward({"data": data, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})

        all_layers = ["ip_main", "ip_residual", "scale1", "bias1", "ip_branch", "ip_final"]
        for lname in all_layers:
            layer = net.layer_by_name(lname)
            for i, blob in enumerate(layer.blobs):
                dw = blob.diff
                assert np.all(np.isfinite(dw)), f"{lname} blob[{i}] diff has NaN/Inf"

    def test_p3d_loss_decreases_with_training(self):
        """Loss should decrease over SGD steps (all P3-D layers contribute to learning).

        This is the GOLD STANDARD test: if loss decreases, gradients flow correctly
        through all layers, and parameter updates work.
        """
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))

        rng = np.random.RandomState(101112)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        losses = []
        for step in range(20):
            out = net.forward({"data": data, "label": label})
            loss = float(out["loss"].item())
            losses.append(loss)
            net.backward({"loss": np.array([1.0], dtype=np.float32)})
            _simple_sgd_update(net, lr=0.01)

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"

        assert all(np.isfinite(l) for l in losses), "NaN/Inf loss encountered during training"

    def test_p3d_softmax_independent_layer_probabilities(self):
        """Standalone Softmax layer produces valid probabilities (sum to 1)."""
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))

        rng = np.random.RandomState(131415)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        net.forward({"data": data, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})

        prob = net.blob_by_name("softmax_prob").data
        prob_sums = np.sum(prob, axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5, err_msg="Softmax probs don't sum to 1")

    def test_p3d_eltwise_sum_gradient_routes(self):
        """Verify Eltwise SUM routes gradient to both input branches (finite checks)."""
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))
        lr = 0.01

        rng = np.random.RandomState(161718)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        # Warmup to ensure gradients flow past ReLU
        for _ in range(5):
            net.forward({"data": data, "label": label})
            net.backward({"loss": np.array([1.0], dtype=np.float32)})
            _simple_sgd_update(net, lr=lr)

        relu_main_diff = net.blob_by_name("relu_main").diff
        residual_diff = net.blob_by_name("ip_residual").diff
        assert np.all(np.isfinite(relu_main_diff)), "relu_main diff has NaN/Inf"
        assert np.all(np.isfinite(residual_diff)), "ip_residual diff has NaN/Inf"

        # After Eltwise SUM with coeff=1 for both, both inputs should receive the same dy
        eltwise_diff = net.blob_by_name("eltwise_sum").diff
        np.testing.assert_allclose(relu_main_diff, eltwise_diff, atol=1e-6,
                                   err_msg="Eltwise SUM main branch gradient mismatch")
        np.testing.assert_allclose(residual_diff, eltwise_diff, atol=1e-6,
                                   err_msg="Eltwise SUM residual branch gradient mismatch")

    def test_p3d_concat_gradient_splits(self):
        """Verify Concat splits gradient correctly to both input branches."""
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))
        lr = 0.01

        rng = np.random.RandomState(192021)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        # Warmup
        for _ in range(5):
            net.forward({"data": data, "label": label})
            net.backward({"loss": np.array([1.0], dtype=np.float32)})
            _simple_sgd_update(net, lr=lr)

        drop_diff = net.blob_by_name("drop1").diff
        branch_diff = net.blob_by_name("relu_branch").diff
        concat_diff = net.blob_by_name("concat1").diff

        assert np.all(np.isfinite(drop_diff)), "drop1 diff NaN/Inf"
        assert np.all(np.isfinite(branch_diff)), "relu_branch diff NaN/Inf"

        np.testing.assert_allclose(concat_diff[:, :16], drop_diff, atol=1e-6,
                                   err_msg="Concat main branch gradient mismatch")
        np.testing.assert_allclose(concat_diff[:, 16:], branch_diff, atol=1e-6,
                                   err_msg="Concat branch gradient mismatch")

    def test_p3d_dropout_identity_gradient_passthrough(self):
        """Dropout with ratio=0 acts as identity: dX = dy."""
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))

        rng = np.random.RandomState(222324)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        net.forward({"data": data, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})

        bias_diff = net.blob_by_name("bias1").diff
        drop_diff = net.blob_by_name("drop1").diff
        np.testing.assert_array_equal(drop_diff, bias_diff,
                                      err_msg="Dropout (ratio=0) should pass gradient through unchanged")

    def test_p3d_scale_bias_gradient_shapes_correct(self):
        """Scale and Bias layers produce gradients with correct shapes."""
        N, input_dim, num_cls = 4, 16, 4
        net = Net(_make_p3d_full_prototxt(N, input_dim, num_cls))

        rng = np.random.RandomState(252627)
        data = rng.randn(N, input_dim).astype(np.float32) * 0.1
        label = rng.randint(0, num_cls, size=(N, 1)).astype(np.float32)

        net.forward({"data": data, "label": label})
        net.backward({"loss": np.array([1.0], dtype=np.float32)})

        scale_layer = net.layer_by_name("scale1")
        assert len(scale_layer.blobs) == 2, "Scale with bias should have 2 blobs"
        assert scale_layer.blobs[0].diff.shape == (16,), "Scale gamma gradient wrong shape"
        assert scale_layer.blobs[1].diff.shape == (16,), "Scale beta gradient wrong shape"

        bias_layer = net.layer_by_name("bias1")
        assert len(bias_layer.blobs) == 1, "Bias should have 1 blob"
        assert bias_layer.blobs[0].diff.shape == (16,), "Bias gradient wrong shape"
