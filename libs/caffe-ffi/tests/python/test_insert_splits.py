"""InsertSplits graph transformation — edge case tests.

Tests boundary conditions for the implicit Split insertion pass that runs during
Net initialization. Covered scenarios:

1.  Zero-consumer blob (dead end, no split)
2.  Single-consumer blob (no split)
3.  In-place ReLU with downstream multi-consumer (split named after last producer)
4.  Loss weight on intermediate blob counts as a consumer (triggers split)
5.  Chained splits (fan-out after fan-out)
6.  Idempotence (explicit splits not duplicated)
7.  Forward pass correctness with in-place + split
8.  Multiple external inputs (param.input()) with splits, correct ordering
9.  Linear chain (no fan-out, zero splits)
10. Double in-place chain (two consecutive in-place ops, split after last producer)
11. Mixed Input layer + param.input() external inputs
12. Caffe native naming convention alignment (with forward verification)
13. Split→Concat→Split nesting (Inception-style topology)
14. Multiple independent splits with correct positions
15. Empty network (zero layers, no crash)
16. Explicit Input layer with 3 consumers (split with 3 outputs)
17. Loss weight + multiple downstream consumers (3-output split)
18. Unknown bottom blob reference raises error

Run with:
    pytest tests/python/test_insert_splits.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import require_cpp_extension
from .caffe_test_helpers import (
    make_net, count_splits,
    assert_split_exists, assert_split_after_producer,
    assert_split_at_position, assert_split_order,
    assert_no_split, assert_exact_split_name,
    assert_forward_shapes,
)


# ──────────────────────────────────────────────────────────────────────
# Test class
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestInsertSplits:
    """Edge cases for automatic Split layer insertion (InsertSplits pass)."""

    def test_dead_end_no_split(self):
        """Zero-consumer blob (dead-end output) should not trigger a split."""
        prototxt = """
name: 'test_dead'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'out1'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data' top: 'dead_end'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'out1' top: 'out3'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        # 'data' → fc1+fc2 = 2 consumers → needs 1 split
        # 'dead_end' → 0 consumers → no split
        # 'out1' → fc3 only → no split
        assert count_splits(net) == 1

    def test_single_consumer_no_split(self):
        """Single-consumer blob should not trigger a split."""
        prototxt = """
name: 'test_single'
input: 'data'
input_shape { dim: 1 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        assert count_splits(net) == 0

    def test_inplace_relu_split_named_after_last_producer(self):
        """In-place ReLU → two consumers: split named after ReLU (last producer)."""
        prototxt = """
name: 'test_inplace'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 } }
layer { name: 'relu1' type: 'ReLU' bottom: 'fc1_out' top: 'fc1_out' }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc3_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        # Split must be named after the LAST producer (relu1), not fc1
        assert_split_exists(names, "fc1_out_relu1_0_split")
        # NO split after fc1 (only 1 consumer: relu)
        assert_no_split(names, "fc1_out_fc1_0_split")
        assert count_splits(net) == 1

    def test_loss_weight_triggers_split(self):
        """Loss weight on an intermediate blob counts as an extra consumer."""
        prototxt = """
name: 'test_loss'
force_backward: true
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 bias_term: false } loss_weight: 2.5 }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data' top: 'fc2_out'
  inner_product_param { num_output: 3 bias_term: false } }
layer { name: 'add' type: 'Eltwise' bottom: 'fc1_out' bottom: 'fc2_out' top: 'sum'
  eltwise_param { operation: SUM } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        # data → fc1+fc2 (2 consumers) → split
        # fc1_out → add(1) + loss_weight(1) = 2 → split
        assert_split_exists(names, "data_input_0_split")
        assert_split_exists(names, "fc1_out_fc1_0_split")
        assert count_splits(net) == 2

    def test_chained_splits(self):
        """Chained fan-out: data→3 consumers, fc2_out→2 consumers."""
        prototxt = """
name: 'test_chain'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'data' top: 'fc3_out'
  inner_product_param { num_output: 3 } }
layer { name: 'add12' type: 'Eltwise' bottom: 'fc1_out' bottom: 'fc2_out' top: 'sum12'
  eltwise_param { operation: SUM } }
layer { name: 'add23' type: 'Eltwise' bottom: 'fc2_out' bottom: 'fc3_out' top: 'sum23'
  eltwise_param { operation: SUM } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        assert_split_exists(names, "data_input_0_split")
        assert_split_exists(names, "fc2_out_fc2_0_split")
        assert count_splits(net) == 2

    def test_idempotent_no_duplicate_splits(self):
        """Pre-split network should not get additional splits inserted."""
        prototxt = """
name: 'test_idem'
input: 'data'
input_shape { dim: 1 dim: 4 }
layer { name: 'data_input_0_split' type: 'Split' bottom: 'data'
  top: 'data_input_0_split_0' top: 'data_input_0_split_1' }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data_input_0_split_0' top: 'out1'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data_input_0_split_1' top: 'out2'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        # Each split output has exactly 1 consumer → no extra splits
        assert count_splits(net) == 1

    def test_forward_correctness_inplace_split(self):
        """Forward pass produces correct outputs with in-place + split."""
        prototxt = """
name: 'test_fwd'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 } }
layer { name: 'relu' type: 'ReLU' bottom: 'fc1_out' top: 'fc1_out' }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'data' top: 'fc3_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        outputs = net.Forward({})
        assert_forward_shapes(outputs, {
            "fc2_out": (2, 3),
            "fc3_out": (2, 3),
        })

    def test_multiple_external_inputs_order(self):
        """Multiple param.input() sources both split; splits appear in input declaration order."""
        prototxt = """
name: 'test_multi_input'
input: 'data'
input: 'weight'
input_shape { dim: 1 dim: 4 }
input_shape { dim: 1 dim: 3 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'out1'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data' top: 'out2'
  inner_product_param { num_output: 3 } }
layer { name: 'scale1' type: 'Scale' bottom: 'out1' bottom: 'weight' top: 'scaled1'
  scale_param { axis: 0 } }
layer { name: 'scale2' type: 'Scale' bottom: 'out2' bottom: 'weight' top: 'scaled2'
  scale_param { axis: 0 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        assert_split_exists(names, "data_input_")
        assert_split_exists(names, "weight_input_")
        assert count_splits(net) == 2
        # Splits must be in input declaration order: data split before weight split
        assert_split_order(names, "data_input_", "weight_input_",
                           msg="data split should precede weight split")

    def test_linear_chain_zero_splits(self):
        """Linear chain (no fan-out) produces zero splits."""
        prototxt = """
name: 'test_linear'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 } }
layer { name: 'relu' type: 'ReLU' bottom: 'fc1_out' top: 'fc1_out' }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        assert count_splits(net) == 0
        assert len(names) == 3

    def test_double_inplace_split_after_last_producer(self):
        """Double in-place (fc→relu→relu) → two consumers: split after last in-place op."""
        prototxt = """
name: 'test_double_inplace'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'x'
  inner_product_param { num_output: 3 } }
layer { name: 'relu1' type: 'ReLU' bottom: 'x' top: 'x' }
layer { name: 'relu2' type: 'ReLU' bottom: 'x' top: 'x' }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'x' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'x' top: 'fc3_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        # Split named after relu2 (last producer), NOT relu1 or fc1
        assert_split_exists(names, "x_relu2_0_split")
        assert_no_split(names, "x_relu1_0_split")
        assert_no_split(names, "x_fc1_0_split")
        assert count_splits(net) == 1

    def test_mixed_input_layer_and_param_input(self):
        """Mixed explicit Input layer + param.input() external inputs.

        Scenario: param.input('data') has 2 consumers (fc1, fc2).
        An explicit Input-type layer ('aux') also has 2 consumers (concat1, concat2).
        Both need splits. The param.input split should appear first (at position 0),
        and the Input layer's split should appear right after the Input layer itself.
        """
        prototxt = """
name: 'test_mixed_input'
input: 'data'
input_shape { dim: 1 dim: 4 }
layer { name: 'aux' type: 'Input' top: 'aux'
  input_param { shape { dim: 1 dim: 3 } } }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'concat1' type: 'Concat' bottom: 'fc1_out' bottom: 'aux' top: 'cat1' }
layer { name: 'concat2' type: 'Concat' bottom: 'fc2_out' bottom: 'aux' top: 'cat2'
  concat_param { axis: 1 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        assert_split_exists(names, "data_input_0_split")
        assert_split_exists(names, "aux_aux_0_split")
        assert count_splits(net) == 2

        # data split must be at the very beginning (before any regular layer)
        assert_split_at_position(names, "data_input_0_split", 0)

        # aux split must appear right after the 'aux' Input layer
        assert_split_after_producer(names, "aux", "aux_aux_0_split")

    def test_split_output_names_match_caffe_native_convention(self):
        """Verify split naming exactly matches native Caffe TestWithInPlace convention.

        Native Caffe expected names (from test_split_layer.cpp TestWithInPlace):
        - data layer (type Input, top: data) → data_data_0_split for fan-out
        - innerprod1 → innerprod1 → relu1 (in-place) → innerprod1_relu1_0_split
          (when relu1's output is consumed by 2+ layers)
        """
        prototxt = """
name: 'TestNetwork'
layer { name: 'data' type: 'Input' top: 'data'
  input_param { shape { dim: 2 dim: 4 } } }
layer { name: 'innerprod1' type: 'InnerProduct' bottom: 'data' top: 'innerprod1'
  inner_product_param { num_output: 3 } }
layer { name: 'relu1' type: 'ReLU' bottom: 'innerprod1' top: 'innerprod1' }
layer { name: 'innerprod2' type: 'InnerProduct' bottom: 'innerprod1' top: 'innerprod2'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'innerprod1' top: 'fc3_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        # data is consumed by innerprod1 and fc2 → split named data_data_0_split
        assert_exact_split_name(names, "data_data_0_split")
        # innerprod1 after relu1 is consumed by innerprod2 and fc3 → split
        assert_exact_split_name(names, "innerprod1_relu1_0_split")
        assert count_splits(net) == 2

        # Forward pass
        inp_data = np.random.randn(2, 4).astype(np.float32)
        outputs = net.Forward({"data": inp_data})
        assert_forward_shapes(outputs, {
            "innerprod2": (2, 3),
            "fc2_out": (2, 3),
            "fc3_out": (2, 3),
        })

    def test_split_concat_split_nested(self):
        """Split→Concat→Split nesting (Inception-style topology).

        Topology:
            data → split → fc_a, fc_b
            fc_a → concat
            fc_b → concat
            concat → split → fc_c, fc_d
        Both data and concat outputs need splits.
        """
        prototxt = """
name: 'test_inception'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc_a' type: 'InnerProduct' bottom: 'data' top: 'fc_a_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc_b' type: 'InnerProduct' bottom: 'data' top: 'fc_b_out'
  inner_product_param { num_output: 3 } }
layer { name: 'cat' type: 'Concat' bottom: 'fc_a_out' bottom: 'fc_b_out' top: 'cat_out'
  concat_param { axis: 1 } }
layer { name: 'fc_c' type: 'InnerProduct' bottom: 'cat_out' top: 'fc_c_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc_d' type: 'InnerProduct' bottom: 'cat_out' top: 'fc_d_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        # data → fc_a + fc_b = 2 consumers → split
        # cat_out → fc_c + fc_d = 2 consumers → split
        assert_split_exists(names, "data_input_0_split")
        assert_split_exists(names, "cat_out_cat_0_split")
        assert count_splits(net) == 2

        # Verify split positions: data split at position 0,
        # cat split immediately after cat layer
        assert_split_at_position(names, "data_input_0_split", 0)
        assert_split_after_producer(names, "cat", "cat_out_cat_0_split")

        # Forward correctness
        inp = np.random.randn(2, 4).astype(np.float32)
        outputs = net.Forward({"data": inp})
        assert_forward_shapes(outputs, {
            "fc_c_out": (2, 3),
            "fc_d_out": (2, 3),
        })

    def test_multiple_layers_need_splits_positions(self):
        """Multiple independent splits: each inserted right after its producer."""
        prototxt = """
name: 'test_multi_split'
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc3_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc4' type: 'InnerProduct' bottom: 'fc2_out' top: 'fc4_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc5' type: 'InnerProduct' bottom: 'fc2_out' top: 'fc5_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        # data has 1 consumer (fc1) → no split
        # fc1_out → fc2 + fc3 = 2 → split
        # fc2_out → fc4 + fc5 = 2 → split
        assert_no_split(names, "data_input_0_split")
        assert_split_exists(names, "fc1_out_fc1_0_split")
        assert_split_exists(names, "fc2_out_fc2_0_split")
        assert count_splits(net) == 2

        # Each split immediately follows its producer
        assert_split_after_producer(names, "fc1", "fc1_out_fc1_0_split")
        assert_split_after_producer(names, "fc2", "fc2_out_fc2_0_split")

    def test_empty_network_no_crash(self):
        """Network with zero layers should not crash and produce zero splits."""
        prototxt = """
name: 'test_empty'
input: 'data'
input_shape { dim: 1 dim: 4 }
"""
        net = make_net(prototxt)
        assert count_splits(net) == 0
        assert len(list(net.layer_names())) == 0

    def test_input_layer_three_consumers(self):
        """Explicit Input layer with 3 consumers → split with 3 outputs."""
        prototxt = """
name: 'test_input3'
layer { name: 'data' type: 'Input' top: 'data'
  input_param { shape { dim: 2 dim: 4 } } }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'data' top: 'fc2_out'
  inner_product_param { num_output: 3 } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'data' top: 'fc3_out'
  inner_product_param { num_output: 3 } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        assert_exact_split_name(names, "data_data_0_split")
        assert count_splits(net) == 1
        assert_split_after_producer(names, "data", "data_data_0_split")

        # Forward: all three branches produce output
        inp = np.random.randn(2, 4).astype(np.float32)
        outputs = net.Forward({"data": inp})
        assert_forward_shapes(outputs, {
            "fc1_out": (2, 3),
            "fc2_out": (2, 3),
            "fc3_out": (2, 3),
        })

    def test_loss_weight_plus_multiple_consumers(self):
        """A layer top with loss_weight + 2 downstream consumers → split with 3 outputs."""
        prototxt = """
name: 'test_loss3'
force_backward: true
input: 'data'
input_shape { dim: 2 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 bias_term: false } loss_weight: 1.0 }
layer { name: 'fc2' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc2_out'
  inner_product_param { num_output: 3 bias_term: false } }
layer { name: 'fc3' type: 'InnerProduct' bottom: 'fc1_out' top: 'fc3_out'
  inner_product_param { num_output: 3 bias_term: false } }
"""
        net = make_net(prototxt)
        names = list(net.layer_names())
        # fc1_out → fc2(1) + fc3(1) + loss(1) = 3 consumers → split needed
        assert_split_exists(names, "fc1_out_fc1_0_split")
        assert count_splits(net) == 1  # only fc1 split; data has 1 consumer

    def test_unknown_bottom_raises_error(self):
        """Referencing an undefined bottom blob should raise a runtime error."""
        prototxt = """
name: 'test_bad_ref'
input: 'data'
input_shape { dim: 1 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'nonexistent' top: 'out'
  inner_product_param { num_output: 3 } }
"""
        with pytest.raises((RuntimeError, ValueError), match="Unknown bottom blob|nonexistent"):
            make_net(prototxt)
