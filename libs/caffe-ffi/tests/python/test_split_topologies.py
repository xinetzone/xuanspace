"""P2-B: Split layer topology tests.

Tests cover:
- TestSplitTopologies: basic 1->N split, residual connections with explicit Split,
  in-place branches, N=1 passthrough, multi-level splits, performance benchmarking
"""
from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import (
    Net, Blob,
    net_param_from_string, net_from_param,
)
from .conftest import require_cpp_extension, perf_trace


# ─── Prototxt builders ─────────────────────────────────────────────

def _make_basic_split_prototxt(num_top: int = 2, feat_dim: int = 4) -> str:
    """Basic 1->N split: data -> Split{top1,...,topN}."""
    lines = [
        'name: "basic_split"',
        'layer {',
        '  name: "data"',
        '  type: "Input"',
        '  top: "data"',
        f'  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"',
        '  type: "Split"',
        '  bottom: "data"',
    ]
    for i in range(num_top):
        lines.append(f'  top: "split_{i}"')
    lines.append('}')
    return "\n".join(lines)


def _make_split_concat_prototxt(num_branches: int = 3, feat_dim: int = 4) -> str:
    """Split into N branches then Concat back (round-trip test)."""
    lines = [
        'name: "split_concat"',
        'layer {',
        '  name: "data"',
        '  type: "Input"',
        '  top: "data"',
        f'  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"',
        '  type: "Split"',
        '  bottom: "data"',
    ]
    for i in range(num_branches):
        lines.append(f'  top: "branch_{i}"')
    lines.append('}')
    lines.append('layer {')
    lines.append('  name: "concat"')
    lines.append('  type: "Concat"')
    for i in range(num_branches):
        lines.append(f'  bottom: "branch_{i}"')
    lines.append('  top: "concat_out"')
    lines.append('  concat_param { axis: 1 }')
    lines.append('}')
    return "\n".join(lines)


def _make_residual_split_prototxt(feat_dim: int = 8) -> str:
    """True residual connection with explicit Split:
    data -> Split -> identity_path + fc+relu_path -> Eltwise SUM -> output
    """
    return f"""name: "residual_split"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
  top: "identity"
  top: "fc_path"
}}
layer {{
  name: "fc1"
  type: "InnerProduct"
  bottom: "fc_path"
  top: "fc1_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: true }}
}}
layer {{
  name: "relu1"
  type: "ReLU"
  bottom: "fc1_out"
  top: "fc1_out"
}}
layer {{
  name: "add"
  type: "Eltwise"
  bottom: "identity"
  bottom: "fc1_out"
  top: "residual_out"
  eltwise_param {{ operation: SUM }}
}}
layer {{
  name: "prob"
  type: "Softmax"
  bottom: "residual_out"
  top: "prob"
}}
"""


def _make_split_inplace_branch_prototxt(feat_dim: int = 8) -> str:
    """Split with in-place ReLU on one branch (tests that in-place doesn't affect siblings)."""
    return f"""name: "split_inplace"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw_branch"
  top: "relu_branch"
}}
layer {{
  name: "relu_on_branch"
  type: "ReLU"
  bottom: "relu_branch"
  top: "relu_branch"
}}
layer {{
  name: "raw_ip"
  type: "InnerProduct"
  bottom: "raw_branch"
  top: "raw_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: false }}
}}
layer {{
  name: "relu_ip"
  type: "InnerProduct"
  bottom: "relu_branch"
  top: "relu_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: false }}
}}
"""


def _make_n1_split_passthrough_prototxt(feat_dim: int = 4) -> str:
    """N=1 Split: should act as passthrough (copy)."""
    return f"""name: "n1_split"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
  top: "passthrough"
}}
layer {{
  name: "ip"
  type: "InnerProduct"
  bottom: "passthrough"
  top: "out"
  inner_product_param {{ num_output: 2 bias_term: true }}
}}
layer {{
  name: "prob"
  type: "Softmax"
  bottom: "out"
  top: "prob"
}}
"""


def _make_split_mlp_perf_prototxt(batch: int, feat_dim: int, n_branches: int, branch_ip_dim: int) -> str:
    """Split + IP branches + Concat + IP + Softmax for performance testing."""
    lines = [
        'name: "split_perf"',
        'layer {',
        '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: {batch} dim: {feat_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"', '  type: "Split"', '  bottom: "data"',
    ]
    for i in range(n_branches):
        lines.append(f'  top: "b{i}"')
    lines.append('}')
    for i in range(n_branches):
        lines.extend([
            'layer {',
            f'  name: "ip_b{i}"', '  type: "InnerProduct"',
            f'  bottom: "b{i}"', f'  top: "b{i}_out"',
            f'  inner_product_param {{ num_output: {branch_ip_dim} bias_term: false }}',
            '}',
        ])
    lines.append('layer {')
    lines.append('  name: "concat"')
    lines.append('  type: "Concat"')
    for i in range(n_branches):
        lines.append(f'  bottom: "b{i}_out"')
    lines.append('  top: "concat"')
    lines.append('  concat_param { axis: 1 }')
    lines.append('}')
    concat_dim = branch_ip_dim * n_branches
    lines.extend([
        'layer {',
        '  name: "ip_out"', '  type: "InnerProduct"',
        '  bottom: "concat"', '  top: "ip_out"',
        '  inner_product_param { num_output: 10 bias_term: true }',
        '}',
        'layer {',
        '  name: "prob"', '  type: "Softmax"',
        '  bottom: "ip_out"', '  top: "prob"',
        '}',
    ])
    return "\n".join(lines)


# ─── Helper: set weights ──────────────────────────────────────────

def _set_random_weights(net: Net, rng: np.random.RandomState, scale: float = 0.1) -> None:
    """Set small random weights into all InnerProduct layers."""
    for layer in net.layers_array():
        if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
            W = layer.blobs[0]
            w_data = rng.randn(*W.shape).astype(np.float32) * scale
            W.from_numpy(w_data)
            if len(layer.blobs) >= 2:
                b = layer.blobs[1]
                b.from_numpy(np.zeros(b.shape, dtype=np.float32))


# ─── Test Class ────────────────────────────────────────────────────

@require_cpp_extension
class TestSplitTopologies:
    """Tests for explicit Split layer multi-branch topologies."""

    def test_split_1to2_copies_data(self, ptrace):
        """1->2 Split: both top blobs must match bottom exactly."""
        prototxt = _make_basic_split_prototxt(num_top=2, feat_dim=4)
        with ptrace("Net(basic_split_1to2)") as t:
            param = net_param_from_string(prototxt)
            net = net_from_param(param)
            t['layers'] = len(net.layers_array())

        rng = np.random.RandomState(123)
        inp = rng.randn(2, 4).astype(np.float32)
        with ptrace("Forward(basic_split_1to2)") as t:
            net.Forward({"data": inp})
            t['shape'] = str(inp.shape)

        split_0 = net.blob_by_name("split_0")
        split_1 = net.blob_by_name("split_1")
        np.testing.assert_array_equal(split_0.to_numpy(), inp)
        np.testing.assert_array_equal(split_1.to_numpy(), inp)

    def test_split_1to3_concat_roundtrip(self, ptrace):
        """1->3 Split then Concat: concat should contain 3 copies of input along axis=1."""
        feat_dim = 4
        num_branches = 3
        prototxt = _make_split_concat_prototxt(num_branches=num_branches, feat_dim=feat_dim)
        with ptrace("Net(split_concat_3)"):
            param = net_param_from_string(prototxt)
            net = net_from_param(param)

        rng = np.random.RandomState(42)
        inp = rng.randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(split_concat_3)"):
            net.Forward({"data": inp})

        concat = net.blob_by_name("concat_out")
        concat_data = concat.to_numpy()
        assert concat_data.shape == (2, feat_dim * num_branches)
        for i in range(num_branches):
            branch = net.blob_by_name(f"branch_{i}")
            np.testing.assert_array_equal(branch.to_numpy(), inp)
        expected = np.concatenate([inp]*num_branches, axis=1)
        np.testing.assert_array_equal(concat_data, expected)

    def test_residual_with_split(self, ptrace):
        """True residual: data -> Split -> (identity, FC+ReLU) -> Eltwise SUM."""
        feat_dim = 8
        prototxt = _make_residual_split_prototxt(feat_dim=feat_dim)
        with ptrace("Net(residual_split)"):
            param = net_param_from_string(prototxt)
            net = net_from_param(param)

        _set_random_weights(net, np.random.RandomState(7), scale=0.1)
        fc1 = net.layer_by_name("fc1")
        if len(fc1.blobs) >= 2:
            b = np.ones(feat_dim, dtype=np.float32) * 0.01
            fc1.blobs[1].from_numpy(b)

        rng = np.random.RandomState(99)
        inp = rng.randn(2, feat_dim).astype(np.float32) + 1.0
        with ptrace("Forward(residual_split)") as t:
            net.Forward({"data": inp})
            t['shape'] = str(inp.shape)

        identity = net.blob_by_name("identity")
        np.testing.assert_array_equal(identity.to_numpy(), inp)
        residual = net.blob_by_name("residual_out")
        res_data = residual.to_numpy()
        assert not np.any(np.isnan(res_data)), "residual output contains NaN"
        assert res_data.shape == inp.shape
        prob = net.blob_by_name("prob").to_numpy()
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)

    def test_split_inplace_branch_isolation(self, ptrace):
        """Split with in-place ReLU on one branch must NOT affect the other branch."""
        feat_dim = 8
        prototxt = _make_split_inplace_branch_prototxt(feat_dim=feat_dim)
        with ptrace("Net(split_inplace)"):
            param = net_param_from_string(prototxt)
            net = net_from_param(param)
        _set_random_weights(net, np.random.RandomState(55), scale=0.1)

        rng = np.random.RandomState(55)
        inp = rng.randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(split_inplace)") as t:
            net.Forward({"data": inp})
            t['shape'] = str(inp.shape)

        raw = net.blob_by_name("raw_branch").to_numpy()
        np.testing.assert_array_equal(raw, inp,
            err_msg="In-place ReLU on one branch corrupted sibling branch data!")
        relu = net.blob_by_name("relu_branch").to_numpy()
        np.testing.assert_array_equal(relu, np.maximum(inp, 0))

    def test_n1_split_passthrough(self, ptrace):
        """N=1 Split: data passes through correctly (equivalent to identity copy)."""
        feat_dim = 4
        prototxt = _make_n1_split_passthrough_prototxt(feat_dim=feat_dim)
        with ptrace("Net(n1_split)"):
            param = net_param_from_string(prototxt)
            net = net_from_param(param)
        _set_random_weights(net, np.random.RandomState(77), scale=0.1)

        rng = np.random.RandomState(77)
        inp = rng.randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n1_split)") as t:
            net.Forward({"data": inp})
            t['shape'] = str(inp.shape)

        pt_data = net.blob_by_name("passthrough").to_numpy()
        np.testing.assert_array_equal(pt_data, inp)
        prob = net.blob_by_name("prob").to_numpy()
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)

    def test_split_deterministic_repeated_forward(self, ptrace):
        """Same input through Split must produce identical results across multiple forwards."""
        prototxt = _make_basic_split_prototxt(num_top=2, feat_dim=4)
        param = net_param_from_string(prototxt)
        net = net_from_param(param)

        rng = np.random.RandomState(2024)
        inp = rng.randn(2, 4).astype(np.float32)

        results = []
        for i in range(5):
            with ptrace(f"Forward(deterministic_{i})"):
                net.Forward({"data": inp})
            results.append(net.blob_by_name("split_0").to_numpy().copy())

        for r in results[1:]:
            np.testing.assert_array_equal(r, results[0])

    def test_split_perf_scaling(self, ptrace):
        """Performance: measure memcpy scaling with input size and number of tops."""
        configs = [
            # (feat_dim, batch, num_tops)
            (128, 1, 2),
            (128, 1, 4),
            (256, 32, 2),
            (256, 32, 4),
            (512, 32, 2),
            (1024, 32, 2),
            (2048, 16, 2),
        ]
        for feat_dim, batch, num_tops in configs:
            proto = _make_split_mlp_perf_prototxt(
                batch=batch, feat_dim=feat_dim,
                n_branches=num_tops, branch_ip_dim=feat_dim)
            param = net_param_from_string(proto)
            net = net_from_param(param)
            _set_random_weights(net, np.random.RandomState(0), scale=0.01)

            rng = np.random.RandomState(0)
            inp = rng.randn(batch, feat_dim).astype(np.float32)

            label = f"SplitPerf(b={batch},d={feat_dim},n={num_tops})"
            net.Forward({"data": inp})  # warm-up
            with ptrace(label) as t:
                net.Forward({"data": inp})
                nbytes = batch * feat_dim * 4 * num_tops
                t['batch'] = batch
                t['feat_dim'] = feat_dim
                t['num_tops'] = num_tops
                t['total_bytes'] = nbytes
