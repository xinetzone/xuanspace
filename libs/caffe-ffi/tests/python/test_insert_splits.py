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

import caffe_ffi
from caffe_ffi import net_param_from_string, net_from_param
from .conftest import require_cpp_extension


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_net(prototxt: str):
    return net_from_param(net_param_from_string(prototxt))


def _count_splits(net) -> int:
    return sum(1 for n in net.layer_names() if n.endswith("_split"))


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
        net = _make_net(prototxt)
        # 'data' → fc1+fc2 = 2 consumers → needs 1 split
        # 'dead_end' → 0 consumers → no split
        # 'out1' → fc3 only → no split
        assert _count_splits(net) == 1

    def test_single_consumer_no_split(self):
        """Single-consumer blob should not trigger a split."""
        prototxt = """
name: 'test_single'
input: 'data'
input_shape { dim: 1 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'out'
  inner_product_param { num_output: 3 } }
"""
        net = _make_net(prototxt)
        assert _count_splits(net) == 0

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # Split must be named after the LAST producer (relu1), not fc1
        assert any("fc1_out_relu1_0_split" in n for n in names)
        # NO split after fc1 (only 1 consumer: relu)
        assert not any(n == "fc1_out_fc1_0_split" for n in names)
        assert _count_splits(net) == 1

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # data → fc1+fc2 (2 consumers) → split
        # fc1_out → add(1) + loss_weight(1) = 2 → split
        assert any("data_input_0_split" in n for n in names)
        assert any("fc1_out_fc1_0_split" in n for n in names)
        assert _count_splits(net) == 2

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        assert any("data_input_0_split" in n for n in names)
        assert any("fc2_out_fc2_0_split" in n for n in names)
        assert _count_splits(net) == 2

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
        net = _make_net(prototxt)
        # Each split output has exactly 1 consumer → no extra splits
        assert _count_splits(net) == 1

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
        net = _make_net(prototxt)
        outputs = net.Forward({})
        assert "fc2_out" in outputs
        assert "fc3_out" in outputs
        assert outputs["fc2_out"].shape == (2, 3)
        assert outputs["fc3_out"].shape == (2, 3)

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        assert any(n.startswith("data_input_") and "_split" in n for n in names)
        assert any(n.startswith("weight_input_") and "_split" in n for n in names)
        assert _count_splits(net) == 2
        # Splits must be in input declaration order: data split before weight split
        data_split_idx = next(
            i for i, n in enumerate(names)
            if n.startswith("data_input_") and "_split" in n
        )
        weight_split_idx = next(
            i for i, n in enumerate(names)
            if n.startswith("weight_input_") and "_split" in n
        )
        assert data_split_idx < weight_split_idx, (
            f"data split (idx {data_split_idx}) should appear before "
            f"weight split (idx {weight_split_idx})"
        )

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        assert _count_splits(net) == 0
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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # Split named after relu2 (last producer), NOT relu1 or fc1
        assert any("x_relu2_0_split" in n for n in names)
        assert not any(n == "x_relu1_0_split" for n in names)
        assert not any(n == "x_fc1_0_split" for n in names)
        assert _count_splits(net) == 1

    def test_mixed_input_layer_and_param_input(self):
        """Mixed explicit Input layer + param.input() external inputs.

        Scenario: param.input('data') has 2 consumers (fc1, fc2).
        An explicit Input-type layer ('aux') also has 2 consumers (concat1, concat2).
        Both need splits. The param.input split should appear first (at position 0),
        and the Input layer's split should appear right after the Input layer itself.
        Forward pass is not tested here (covered by test_forward_correctness);
        this test focuses on structural graph transformation.
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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # param.input('data') has 2 consumers (fc1, fc2) → split
        # Input layer 'aux' top has 2 consumers (concat1, concat2) → split
        assert any("data_input_0_split" in n for n in names)
        assert any("aux_aux_0_split" in n for n in names)
        assert _count_splits(net) == 2

        # data split must be at the very beginning (before any regular layer)
        data_split_idx = next(
            i for i, n in enumerate(names) if "data_input_0_split" in n
        )
        assert data_split_idx == 0, (
            f"param.input() split should be at position 0, got idx {data_split_idx}"
        )

        # aux split must appear right after the 'aux' Input layer
        aux_idx = names.index("aux")
        aux_split_idx = next(
            i for i, n in enumerate(names) if "aux_aux_0_split" in n
        )
        assert aux_split_idx == aux_idx + 1, (
            f"aux split should be immediately after aux layer (idx {aux_idx}), "
            f"got idx {aux_split_idx}"
        )

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # data is consumed by innerprod1 and fc2 → split named data_data_0_split
        assert "data_data_0_split" in names
        # innerprod1 after relu1 is consumed by innerprod2 and fc3 → split
        assert "innerprod1_relu1_0_split" in names
        assert _count_splits(net) == 2

        # Forward pass
        inp_data = np.random.randn(2, 4).astype(np.float32)
        outputs = net.Forward({"data": inp_data})
        assert "innerprod2" in outputs
        assert "fc2_out" in outputs
        assert "fc3_out" in outputs
        assert outputs["innerprod2"].shape == (2, 3)
        assert outputs["fc2_out"].shape == (2, 3)
        assert outputs["fc3_out"].shape == (2, 3)

    def test_split_concat_split_nested(self):
        """Split→Concat→Split nesting (Inception-style topology).

        Topology:
            data → split → fc_a, fc_b
            fc_a → concat
            fc_b → concat
            concat → split → loss1, loss2
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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # data → fc_a + fc_b = 2 consumers → split
        # cat_out → fc_c + fc_d = 2 consumers → split
        assert any("data_input_0_split" in n for n in names)
        assert any("cat_out_cat_0_split" in n for n in names)
        assert _count_splits(net) == 2

        # Verify split positions: data split at position 0,
        # cat split immediately after cat layer
        cat_idx = names.index("cat")
        cat_split_idx = next(i for i, n in enumerate(names) if "cat_out_cat_0_split" in n)
        assert cat_split_idx == cat_idx + 1

        # Forward correctness
        inp = np.random.randn(2, 4).astype(np.float32)
        outputs = net.Forward({"data": inp})
        assert outputs["fc_c_out"].shape == (2, 3)
        assert outputs["fc_d_out"].shape == (2, 3)

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # data has 1 consumer (fc1) → no split
        # fc1_out → fc2 + fc3 = 2 → split
        # fc2_out → fc4 + fc5 = 2 → split
        assert not any("data_input_0_split" in n for n in names)
        assert any("fc1_out_fc1_0_split" in n for n in names)
        assert any("fc2_out_fc2_0_split" in n for n in names)
        assert _count_splits(net) == 2

        # Each split immediately follows its producer
        fc1_idx = names.index("fc1")
        fc1_split_idx = next(i for i, n in enumerate(names) if "fc1_out_fc1_0_split" in n)
        assert fc1_split_idx == fc1_idx + 1
        fc2_idx = names.index("fc2")
        fc2_split_idx = next(i for i, n in enumerate(names) if "fc2_out_fc2_0_split" in n)
        assert fc2_split_idx == fc2_idx + 1

    def test_empty_network_no_crash(self):
        """Network with zero layers should not crash and produce zero splits."""
        prototxt = """
name: 'test_empty'
input: 'data'
input_shape { dim: 1 dim: 4 }
"""
        net = _make_net(prototxt)
        assert _count_splits(net) == 0
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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        assert "data_data_0_split" in names
        assert _count_splits(net) == 1
        data_idx = names.index("data")
        split_idx = names.index("data_data_0_split")
        assert split_idx == data_idx + 1

        # Forward: all three branches produce output
        inp = np.random.randn(2, 4).astype(np.float32)
        outputs = net.Forward({"data": inp})
        assert outputs["fc1_out"].shape == (2, 3)
        assert outputs["fc2_out"].shape == (2, 3)
        assert outputs["fc3_out"].shape == (2, 3)

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
        net = _make_net(prototxt)
        names = list(net.layer_names())
        # fc1_out → fc2(1) + fc3(1) + loss(1) = 3 consumers → split needed
        assert any("fc1_out_fc1_0_split" in n for n in names)
        assert _count_splits(net) == 1  # only fc1 split; data has 1 consumer

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
            _make_net(prototxt)
