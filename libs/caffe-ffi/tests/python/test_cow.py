"""P2-B: Copy-on-Write (COW) extension tests.

Tests cover:
- TestBlobCOWApi: Python-side COW API validation (IsDataShared, DataRefCount,
  UnshareData, mutable_data_tensor, cow_snapshot helper)
- TestSplitCOWBehavior: Split-layer COW integration tests (N>=2 data isolation,
  refcount verification, const access no-COW, COW after in-place ReLU)
- TestDropoutCOWBehavior: Dropout inference-mode COW sharing (zero-copy)
"""

from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import Blob, net_param_from_string, net_from_param
from .conftest import require_cpp_extension, perf_trace, cow_snapshot


# ─── Prototxt builders ──────────────────────────────────────────────

def _make_basic_split_prototxt(num_top: int = 2, feat_dim: int = 4) -> str:
    lines = [
        'name: "basic_split"',
        'layer {',
        '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"', '  type: "Split"', '  bottom: "data"',
    ]
    for i in range(num_top):
        lines.append(f'  top: "split_{i}"')
    lines.append('}')
    return "\n".join(lines)


def _make_split_inplace_branch_prototxt(feat_dim: int = 8) -> str:
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


# ═══════════════════════════════════════════════════════════════════════
# Test Class 1: Blob-level COW API (Python side)
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestBlobCOWApi:
    """Python-side validation of COW API methods on Blob objects."""

    def test_IsDataShared_false_for_standalone(self):
        """Standalone Blob: IsDataShared() returns False, DataRefCount() == 1."""
        b = Blob([4, 4])
        assert not b.IsDataShared()
        assert b.DataRefCount() == 1

    def test_IsDataShared_true_after_ShareData(self):
        """After ShareData: IsDataShared() returns True, DataRefCount() > 1."""
        src = Blob([4, 4])
        dst = Blob([4, 4])
        dst.ShareData(src)
        assert dst.IsDataShared()
        assert dst.DataRefCount() >= 2
        # src has unique ownership (dst acquired a reference to src's tensor)
        assert not src.IsDataShared()

    def test_IsDiffShared_false_for_standalone(self):
        """Standalone Blob: IsDiffShared() returns False, DiffRefCount() == 1."""
        b = Blob([3, 3])
        assert not b.IsDiffShared()
        assert b.DiffRefCount() == 1

    def test_IsDiffShared_true_after_ShareDiff(self):
        """After ShareDiff: IsDiffShared() returns True."""
        src = Blob([3, 3])
        dst = Blob([3, 3])
        dst.ShareDiff(src)
        assert dst.IsDiffShared()
        assert dst.DiffRefCount() >= 2

    def test_DataRefCount_zero_for_undefined(self):
        """Empty Blob (no Reshape): DataRefCount() returns 0."""
        b = Blob()  # shape [0], no tensor allocated
        # DataRefCount returns 0 for undefined tensor
        assert b.DataRefCount() == 0

    def test_UnshareData_breaks_sharing(self):
        """UnshareData() breaks sharing: IsDataShared → False, pointer changes."""
        src = Blob([4])
        src_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        src.from_numpy(src_data)

        dst = Blob([4])
        dst.ShareData(src)
        assert dst.IsDataShared()

        old_ptr = dst.data_tensor.ctypes.data
        dst.UnshareData()
        assert not dst.IsDataShared()
        new_ptr = dst.data_tensor.ctypes.data
        assert new_ptr != old_ptr, "UnshareData must clone to a new buffer"

        # Data content preserved after COW
        np.testing.assert_array_equal(dst.to_numpy(), src_data)

    def test_UnshareDiff_breaks_sharing(self):
        """UnshareDiff() breaks diff sharing."""
        src = Blob([3])
        src.diff = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        dst = Blob([3])
        dst.ShareDiff(src)
        assert dst.IsDiffShared()

        old_ptr = dst.diff_tensor.ctypes.data
        dst.UnshareDiff()
        assert not dst.IsDiffShared()
        new_ptr = dst.diff_tensor.ctypes.data
        assert new_ptr != old_ptr

    def test_mutable_data_tensor_triggers_COW(self):
        """mutable_data_tensor() triggers COW: refcount drops to 1, pointer changes."""
        src = Blob([4])
        src.from_numpy(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
        dst = Blob([4])
        dst.ShareData(src)
        assert dst.IsDataShared()
        old_ptr = dst.data_tensor.ctypes.data

        mt = dst.mutable_data_tensor()
        assert not dst.IsDataShared()
        assert dst.DataRefCount() == 1
        new_ptr = dst.data_tensor.ctypes.data
        assert new_ptr != old_ptr

        # Data content preserved
        np.testing.assert_array_equal(
            np.asarray(mt), [10.0, 20.0, 30.0, 40.0])

    def test_mutable_diff_tensor_triggers_COW(self):
        """mutable_diff_tensor() triggers COW for diff tensor."""
        src = Blob([3])
        src.diff = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        dst = Blob([3])
        dst.ShareDiff(src)
        assert dst.IsDiffShared()

        mt = dst.mutable_diff_tensor()
        assert not dst.IsDiffShared()
        assert dst.DiffRefCount() == 1

    def test_cow_snapshot_helper(self):
        """cow_snapshot() returns expected dict structure."""
        src = Blob([4])
        dst = Blob([4])
        dst.ShareData(src)
        dst.ShareDiff(src)

        snap = cow_snapshot(dst)
        assert snap["data_shared"] is True
        assert snap["diff_shared"] is True
        assert snap["data_refcount"] >= 2
        assert snap["diff_refcount"] >= 2

        dst.UnshareData()
        snap2 = cow_snapshot(dst)
        assert snap2["data_shared"] is False
        assert snap2["diff_shared"] is True  # diff still shared

    def test_three_way_share_refcount(self):
        """Three-way ShareData: refcounts reflect correct fan-out."""
        a = Blob([4])
        a.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = Blob([4])
        c = Blob([4])

        b.ShareData(a)
        c.ShareData(a)

        # b and c share with a
        assert b.IsDataShared()
        assert c.IsDataShared()
        assert b.DataRefCount() >= 3  # a + b + c
        assert c.DataRefCount() >= 3

        # b triggers COW → only b breaks sharing
        b.UnshareData()
        assert not b.IsDataShared()
        assert b.DataRefCount() == 1
        # c still shares with a
        assert c.IsDataShared()
        assert c.DataRefCount() >= 2  # a + c

    def test_UnshareData_noop_when_not_shared(self):
        """UnshareData() on standalone blob is a no-op: pointer unchanged."""
        b = Blob([4])
        b.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert not b.IsDataShared()
        old_ptr = b.data_tensor.ctypes.data
        b.UnshareData()
        assert not b.IsDataShared()
        assert b.data_tensor.ctypes.data == old_ptr

    def test_const_data_tensor_does_not_trigger_COW(self):
        """data_tensor() (const) does NOT trigger COW: sharing maintained."""
        src = Blob([4])
        dst = Blob([4])
        dst.ShareData(src)
        assert dst.IsDataShared()

        # Access via const data_tensor() — should not trigger COW
        _ = dst.data_tensor
        assert dst.IsDataShared(), "const data_tensor() must not trigger COW"

        # Access via mutable_data_tensor() — should trigger COW
        _ = dst.mutable_data_tensor()
        assert not dst.IsDataShared(), "mutable_data_tensor() must trigger COW"


# ═══════════════════════════════════════════════════════════════════════
# Test Class 2: Split-layer COW Integration Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSplitCOWBehavior:
    """COW behavior verification in Split layer topologies."""

    def test_n1_split_zero_copy_data_shared(self, ptrace):
        """N=1 Split: top blob shares data pointer with bottom (Phase 1 zero-copy)."""
        feat_dim = 4
        net = net_from_param(net_param_from_string(
            _make_n1_split_passthrough_prototxt(feat_dim)))
        inp = np.random.RandomState(77).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n1_split_cow)"):
            net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        pass_blob = net.blob_by_name("passthrough")

        # N=1 Split uses ShareData → zero-copy sharing
        assert pass_blob.IsDataShared(), \
            "N=1 Split: passthrough blob must share data tensor"
        assert pass_blob.SharesDataWith(data_blob), \
            "N=1 Split: passthrough must share data with bottom"

        # Pointers must be equal (zero-copy)
        data_ptr = data_blob.data_tensor.ctypes.data
        pass_ptr = pass_blob.data_tensor.ctypes.data
        assert data_ptr == pass_ptr, \
            f"N=1 zero-copy broken: data=0x{data_ptr:x}, pass=0x{pass_ptr:x}"

    def test_n2_split_data_shared_before_write(self, ptrace):
        """N=2 Split: both top blobs share data with bottom BEFORE any write."""
        feat_dim = 4
        net = net_from_param(net_param_from_string(
            _make_basic_split_prototxt(num_top=2, feat_dim=feat_dim)))
        inp = np.random.RandomState(42).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n2_split_cow)"):
            net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        split_0 = net.blob_by_name("split_0")
        split_1 = net.blob_by_name("split_1")

        # Phase 2: N>=2 Split uses ShareData → zero-copy sharing
        assert split_0.IsDataShared(), \
            "N=2 Split: split_0 must share data tensor (Phase 2 COW)"
        assert split_1.IsDataShared(), \
            "N=2 Split: split_1 must share data tensor (Phase 2 COW)"
        assert split_0.SharesDataWith(data_blob)
        assert split_1.SharesDataWith(data_blob)
        assert split_0.SharesDataWith(split_1)

        # All three share the same physical memory
        assert split_0.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data
        assert split_1.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data

    def test_n2_split_cow_isolation_after_write(self, ptrace):
        """N=2 Split: writing to split_0 via mutable_data_tensor triggers COW,
        isolating it from split_1 and data."""
        feat_dim = 4
        net = net_from_param(net_param_from_string(
            _make_basic_split_prototxt(num_top=2, feat_dim=feat_dim)))
        inp = np.random.RandomState(99).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n2_split_cow_iso)"):
            net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        split_0 = net.blob_by_name("split_0")
        split_1 = net.blob_by_name("split_1")

        # Before write: all shared
        assert split_0.IsDataShared()
        assert split_1.IsDataShared()

        # Write to split_0 via mutable_data_tensor → COW triggers
        mt0 = split_0.mutable_data_tensor()
        mt0[0, 0] = 999.0

        # split_0 must break sharing after COW
        assert not split_0.IsDataShared(), \
            "split_0 must break sharing after COW-triggered write"
        assert split_0.DataRefCount() == 1

        # split_1 and data must still share (unaffected)
        assert split_1.IsDataShared(), \
            "split_1 must remain shared after sibling COW"
        assert split_1.SharesDataWith(data_blob)

        # Data isolation: split_1 still has original values
        split_1_data = split_1.to_numpy()
        np.testing.assert_array_equal(split_1_data, inp,
            err_msg="COW on split_0 must not affect split_1 data")

        # split_0 has the modified value
        split_0_data = split_0.to_numpy()
        assert split_0_data[0, 0] == 999.0, \
            "split_0 must have the modified value after COW"

    def test_n4_split_cow_isolation_after_write(self, ptrace):
        """N=4 Split: writing to one branch triggers COW for that branch only."""
        feat_dim = 6
        num_tops = 4
        net = net_from_param(net_param_from_string(
            _make_basic_split_prototxt(num_top=num_tops, feat_dim=feat_dim)))
        inp = np.random.RandomState(55).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n4_split_cow)"):
            net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        splits = [net.blob_by_name(f"split_{i}") for i in range(num_tops)]

        # Before write: all shared
        for i, s in enumerate(splits):
            assert s.IsDataShared(), f"split_{i} must be shared before write"
            assert s.SharesDataWith(data_blob)

        # Write to split_2 → COW triggers for split_2 only
        mt2 = splits[2].mutable_data_tensor()
        mt2[0, 0] = 777.0

        # split_2: no longer shared
        assert not splits[2].IsDataShared()
        assert splits[2].DataRefCount() == 1

        # Other splits: still shared
        for i in [0, 1, 3]:
            assert splits[i].IsDataShared(), \
                f"split_{i} must remain shared after sibling COW"
            assert splits[i].SharesDataWith(data_blob)
            np.testing.assert_array_equal(splits[i].to_numpy(), inp,
                err_msg=f"COW on split_2 must not affect split_{i}")

        # split_2 has modified value
        assert splits[2].to_numpy()[0, 0] == 777.0

    def test_n2_split_const_access_no_cow(self, ptrace):
        """N=2 Split: const data_tensor access does NOT trigger COW."""
        feat_dim = 4
        net = net_from_param(net_param_from_string(
            _make_basic_split_prototxt(num_top=2, feat_dim=feat_dim)))
        inp = np.random.RandomState(33).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n2_const_no_cow)"):
            net.Forward({"data": inp})

        split_0 = net.blob_by_name("split_0")
        split_1 = net.blob_by_name("split_1")

        # Const access via data_tensor (read-only)
        _ = split_0.data_tensor
        _ = split_1.data_tensor

        # Both must still be shared (const access = no COW)
        assert split_0.IsDataShared(), "const data_tensor must not trigger COW"
        assert split_1.IsDataShared(), "const data_tensor must not trigger COW"
        assert split_0.SharesDataWith(split_1)

        # Const access via to_numpy (read-only copy)
        _ = split_0.to_numpy()
        assert split_0.IsDataShared(), "to_numpy must not trigger COW"

    def test_n2_split_cow_after_inplace_relu(self, ptrace):
        """N=2 Split with in-place ReLU on one branch: COW isolates sibling."""
        feat_dim = 8
        net = net_from_param(net_param_from_string(
            _make_split_inplace_branch_prototxt(feat_dim)))

        # Set random weights for InnerProduct layers
        rng = np.random.RandomState(55)
        for layer in net.layers_array():
            if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
                W = layer.blobs[0]
                w_data = rng.randn(*W.shape).astype(np.float32) * 0.1
                W.from_numpy(w_data)
                if len(layer.blobs) >= 2:
                    b = layer.blobs[1]
                    b.from_numpy(np.zeros(b.shape, dtype=np.float32))

        inp = np.random.RandomState(55).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(split_inplace_cow)"):
            net.Forward({"data": inp})

        raw_branch = net.blob_by_name("raw_branch")
        relu_branch = net.blob_by_name("relu_branch")

        # After in-place ReLU, relu_branch should have triggered COW
        # (ReLU writes to relu_branch in-place, which triggers COW)
        assert not relu_branch.IsDataShared(), \
            "In-place ReLU must trigger COW, breaking sharing"

        # raw_branch must still contain original data (isolated by COW)
        raw_data = raw_branch.to_numpy()
        np.testing.assert_array_equal(raw_data, inp,
            err_msg="In-place ReLU on sibling must not corrupt raw_branch (COW isolation)")

        # relu_branch must have ReLU'd values
        relu_data = relu_branch.to_numpy()
        np.testing.assert_array_equal(relu_data, np.maximum(inp, 0))

    def test_n2_split_cow_refcount_after_multiple_writes(self, ptrace):
        """N=2 Split: multiple writes to same branch trigger COW only once."""
        feat_dim = 4
        net = net_from_param(net_param_from_string(
            _make_basic_split_prototxt(num_top=2, feat_dim=feat_dim)))
        inp = np.random.RandomState(11).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n2_multi_write)"):
            net.Forward({"data": inp})

        split_0 = net.blob_by_name("split_0")

        # First write: COW triggers
        mt0 = split_0.mutable_data_tensor()
        mt0[0, 0] = 100.0
        assert not split_0.IsDataShared()
        assert split_0.DataRefCount() == 1

        # Second write: already private, no additional COW
        mt0_2 = split_0.mutable_data_tensor()
        mt0_2[0, 1] = 200.0
        assert not split_0.IsDataShared()
        assert split_0.DataRefCount() == 1

        # Verify both modifications are present
        split_0_data = split_0.to_numpy()
        assert split_0_data[0, 0] == 100.0
        assert split_0_data[0, 1] == 200.0

    def test_cow_snapshot_before_after_forward(self, ptrace):
        """cow_snapshot() before and after Split forward shows correct state."""
        feat_dim = 4
        net = net_from_param(net_param_from_string(
            _make_basic_split_prototxt(num_top=2, feat_dim=feat_dim)))
        inp = np.random.RandomState(22).randn(2, feat_dim).astype(np.float32)

        with ptrace("Forward(cow_snapshot)"):
            net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        split_0 = net.blob_by_name("split_0")
        split_1 = net.blob_by_name("split_1")

        # Snapshot before any write
        snap0_before = cow_snapshot(split_0)
        assert snap0_before["data_shared"] is True
        assert snap0_before["data_refcount"] >= 3  # data + split_0 + split_1

        # Trigger COW on split_0
        _ = split_0.mutable_data_tensor()

        snap0_after = cow_snapshot(split_0)
        assert snap0_after["data_shared"] is False
        assert snap0_after["data_refcount"] == 1

        # split_1 still shared
        snap1_after = cow_snapshot(split_1)
        assert snap1_after["data_shared"] is True
        assert snap1_after["data_refcount"] >= 2  # data + split_1


# ═══════════════════════════════════════════════════════════════════════
# Test Class 3: Dropout inference-mode COW sharing (S5 optimization)
# ═══════════════════════════════════════════════════════════════════════

def _make_dropout_prototxt(input_dims, dropout_ratio=0.5):
    """Build Input -> Dropout prototxt (non-inplace, same shape)."""
    dims = " ".join(f"input_dim: {d}" for d in input_dims)
    return f"""name: "dropout_cow"
input: "data"
{dims}
layer {{
  name: "drop"
  type: "Dropout"
  bottom: "data"
  top: "drop"
  dropout_param {{ dropout_ratio: {dropout_ratio} }}
}}
"""


def _make_dropout_relu_prototxt(input_dims, dropout_ratio=0.5):
    """Input -> Dropout -> in-place ReLU (exercises COW-on-write downstream)."""
    dims = " ".join(f"input_dim: {d}" for d in input_dims)
    return f"""name: "dropout_relu_cow"
input: "data"
{dims}
layer {{
  name: "drop"
  type: "Dropout"
  bottom: "data"
  top: "drop"
  dropout_param {{ dropout_ratio: {dropout_ratio} }}
}}
layer {{
  name: "relu"
  type: "ReLU"
  bottom: "drop"
  top: "drop"
}}
"""


@require_cpp_extension
class TestDropoutCOWBehavior:
    """Dropout inference-mode COW zero-copy sharing.

    In inference mode Dropout is identity (y = x), so the non-inplace forward
    now shares bottom's data tensor via ShareData (O(1)) instead of memcpy
    (O(n)). The backward shares bottom's diff via ShareDiff.
    """

    def test_inference_forward_zerocopy_data_shared(self, ptrace):
        """Inference non-inplace forward: top shares bottom's data pointer."""
        net = net_from_param(net_param_from_string(
            _make_dropout_prototxt((2, 8))))
        inp = np.random.RandomState(7).randn(2, 8).astype(np.float32)
        with ptrace("Forward(dropout_cow_zerocopy)"):
            net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        drop_blob = net.blob_by_name("drop")

        # COW zero-copy: top shares bottom's data tensor.
        assert drop_blob.IsDataShared(), \
            "Inference Dropout: top must share data tensor (zero-copy)"
        assert drop_blob.SharesDataWith(data_blob), \
            "Inference Dropout: top must share data with bottom"
        assert drop_blob.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data, \
            "Inference Dropout: pointers must be equal (zero-copy)"

        # Identity output preserved.
        np.testing.assert_array_equal(drop_blob.to_numpy(), inp)

    def test_inference_forward_identity_preserved(self, ptrace):
        """Identity forward still holds after COW sharing (y == x)."""
        dims = (3, 4, 5)
        net = net_from_param(net_param_from_string(
            _make_dropout_prototxt(dims, dropout_ratio=0.3)))
        inp = np.random.RandomState(21).randn(*dims).astype(np.float32)
        with ptrace("Forward(dropout_cow_identity)"):
            net.Forward({"data": inp})
        drop_blob = net.blob_by_name("drop")
        # .data returns a copy; const access must not trigger COW.
        np.testing.assert_array_equal(drop_blob.data, inp)
        assert drop_blob.IsDataShared(), \
            "const .data access must not trigger COW"

    def test_inference_downstream_inplace_relu_cow_isolation(self, ptrace):
        """In-place ReLU downstream triggers COW; bottom data stays intact."""
        dims = (2, 8)
        net = net_from_param(net_param_from_string(
            _make_dropout_relu_prototxt(dims, dropout_ratio=0.5)))
        inp = np.random.RandomState(33).randn(*dims).astype(np.float32)
        with ptrace("Forward(dropout_cow_relu)"):
            net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        drop_blob = net.blob_by_name("drop")

        # In-place ReLU wrote to drop -> COW triggered, drop isolated.
        assert not drop_blob.IsDataShared(), \
            "In-place ReLU on drop must trigger COW, breaking sharing"
        assert drop_blob.data_tensor.ctypes.data != data_blob.data_tensor.ctypes.data
        # bottom (data) unchanged.
        np.testing.assert_array_equal(data_blob.to_numpy(), inp)
        # drop holds ReLU'd values.
        np.testing.assert_array_equal(drop_blob.to_numpy(), np.maximum(inp, 0))

    def test_inference_backward_zerocopy_diff_shared(self, ptrace):
        """Inference backward (non-inplace): bottom diff shares top diff."""
        net = net_from_param(net_param_from_string(
            _make_dropout_prototxt((2, 8))))
        inp = np.random.RandomState(5).randn(2, 8).astype(np.float32)
        dy = np.random.RandomState(6).randn(2, 8).astype(np.float32)
        with ptrace("Forward(dropout_cow_bw)"):
            net.Forward({"data": inp})
            net.backward({"drop": dy})

        data_blob = net.blob_by_name("data")
        drop_blob = net.blob_by_name("drop")

        assert data_blob.IsDiffShared(), \
            "Inference Dropout backward: bottom diff must be shared"
        assert data_blob.SharesDiffWith(drop_blob), \
            "Inference Dropout backward: bottom diff shares top diff"
        # Identity dx == dy.
        np.testing.assert_array_equal(data_blob.diff, dy)

    def test_training_forward_no_cow_share(self):
        """Training forward must NOT share (masked copy), default behavior kept."""
        net = net_from_param(net_param_from_string(
            _make_dropout_prototxt((4, 16))))
        layer = net.layer_by_name("drop")
        layer.set_train_mode(True)
        x = np.full((4, 16), 1.0, dtype=np.float32)
        net.Forward({"data": x})
        drop_blob = net.blob_by_name("drop")
        # Training performs a real masked copy -> no COW sharing with bottom.
        assert not drop_blob.IsDataShared(), \
            "Training Dropout must not share data tensor with bottom"

    def test_inplace_dropout_no_share(self):
        """Inplace Dropout (top == bottom) needs no COW sharing."""
        proto = """name: "dropout_inplace_cow"
input: "data"
input_dim: 1
input_dim: 4
layer {
  name: "drop"
  type: "Dropout"
  bottom: "data"
  top: "data"
  dropout_param { dropout_ratio: 0.5 }
}
"""
        net = net_from_param(net_param_from_string(proto))
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        net.Forward({"data": x})
        data_blob = net.blob_by_name("data")
        # Inplace: same blob, no shared-state flag (bottom==top).
        np.testing.assert_array_equal(data_blob.to_numpy(), x)


# ─── Prototxt builders for TS31-B4 identity-layer COW promotion ──────────

def _make_scale_identity_prototxt(dims, bias_term=True) -> str:
    """Input -> Scale with scale=1.0 and bias=0.0 (identity)."""
    bias_str = "true" if bias_term else "false"
    bias_filler = "bias_filler { type: \"constant\" value: 0.0 }" if bias_term else ""
    return f"""name: "scale_cow"
input: "data"
input_dim: {dims[0]}
input_dim: {dims[1]}
layer {{
  name: "scale"
  type: "Scale"
  bottom: "data"
  top: "scale_out"
  scale_param {{
    axis: 1
    num_axes: 1
    bias_term: {bias_str}
    filler {{ type: "constant" value: 1.0 }}
    {bias_filler}
  }}
}}
"""


def _make_scale_nonidentity_prototxt(dims) -> str:
    """Input -> Scale with scale=2.0 (non-identity)."""
    return f"""name: "scale_cow_nonid"
input: "data"
input_dim: {dims[0]}
input_dim: {dims[1]}
layer {{
  name: "scale"
  type: "Scale"
  bottom: "data"
  top: "scale_out"
  scale_param {{
    axis: 1
    num_axes: 1
    bias_term: false
    filler {{ type: "constant" value: 2.0 }}
  }}
}}
"""


def _make_bias_identity_prototxt(dims) -> str:
    """Input -> Bias with bias=0.0 (identity)."""
    return f"""name: "bias_cow"
input: "data"
input_dim: {dims[0]}
input_dim: {dims[1]}
layer {{
  name: "bias"
  type: "Bias"
  bottom: "data"
  top: "bias_out"
  bias_param {{
    axis: 1
    num_axes: 1
    filler {{ type: "constant" value: 0.0 }}
  }}
}}
"""


def _make_bias_nonidentity_prototxt(dims) -> str:
    """Input -> Bias with bias=2.0 (non-identity)."""
    return f"""name: "bias_cow_nonid"
input: "data"
input_dim: {dims[0]}
input_dim: {dims[1]}
layer {{
  name: "bias"
  type: "Bias"
  bottom: "data"
  top: "bias_out"
  bias_param {{
    axis: 1
    num_axes: 1
    filler {{ type: "constant" value: 2.0 }}
  }}
}}
"""


def _make_eltwise_identity_prototxt(dims, coeff=1.0) -> str:
    """Input -> Eltwise (single bottom, coeff)."""
    return f"""name: "eltwise_cow"
input: "data"
input_dim: {dims[0]}
input_dim: {dims[1]}
layer {{
  name: "eltwise"
  type: "Eltwise"
  bottom: "data"
  top: "eltwise_out"
  eltwise_param {{
    operation: SUM
    coeff: {coeff}
  }}
}}
"""


def _make_scale_relu_prototxt(dims) -> str:
    """Input -> Scale(identity) -> in-place ReLU on scale_out."""
    return f"""name: "scale_cow_relu"
input: "data"
input_dim: {dims[0]}
input_dim: {dims[1]}
layer {{
  name: "scale"
  type: "Scale"
  bottom: "data"
  top: "scale_out"
  scale_param {{
    axis: 1
    num_axes: 1
    bias_term: false
    filler {{ type: "constant" value: 1.0 }}
  }}
}}
layer {{
  name: "relu"
  type: "ReLU"
  bottom: "scale_out"
  top: "scale_out"
}}
"""


@require_cpp_extension
class TestIdentityLayerCOWBehavior:
    """TS31-B4: COW zero-copy sharing promoted to identity layers.

    Scale (scale=1, bias=0), Bias (bias=0), and single-bottom Eltwise
    (coeff=1) all degenerate to identity (y = x). Their forward now shares
    bottom's data via ShareData (O(1)) instead of an O(n) memcpy, and their
    backward shares bottom's diff via ShareDiff. Non-identity parameters must
    keep the original memcpy path (no COW sharing).
    """

    def test_scale_identity_forward_zerocopy(self, ptrace):
        """Scale(scale=1, no bias) forward: top shares bottom's data pointer."""
        net = net_from_param(net_param_from_string(
            _make_scale_identity_prototxt((2, 8), bias_term=False)))
        inp = np.random.RandomState(11).randn(2, 8).astype(np.float32)
        with ptrace("Forward(scale_cow_identity)"):
            net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        scale_blob = net.blob_by_name("scale_out")
        assert scale_blob.IsDataShared(), \
            "Identity Scale: top must share data tensor (zero-copy)"
        assert scale_blob.SharesDataWith(data_blob), \
            "Identity Scale: top must share data with bottom"
        assert scale_blob.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data, \
            "Identity Scale: pointers must be equal (zero-copy)"
        np.testing.assert_array_equal(scale_blob.to_numpy(), inp)

    def test_scale_identity_with_bias_zero_zerocopy(self, ptrace):
        """Scale(scale=1, bias=0) forward: also zero-copy."""
        net = net_from_param(net_param_from_string(
            _make_scale_identity_prototxt((2, 8), bias_term=True)))
        inp = np.random.RandomState(12).randn(2, 8).astype(np.float32)
        with ptrace("Forward(scale_cow_bias_zero)"):
            net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        scale_blob = net.blob_by_name("scale_out")
        assert scale_blob.IsDataShared()
        assert scale_blob.SharesDataWith(data_blob)
        np.testing.assert_array_equal(scale_blob.to_numpy(), inp)

    def test_scale_nonidentity_no_cow(self):
        """Scale(scale=2.0) forward: real copy, no COW sharing."""
        net = net_from_param(net_param_from_string(
            _make_scale_nonidentity_prototxt((2, 8))))
        inp = np.random.RandomState(13).randn(2, 8).astype(np.float32)
        net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        scale_blob = net.blob_by_name("scale_out")
        assert not scale_blob.IsDataShared(), \
            "Non-identity Scale must not share data tensor"
        assert not scale_blob.SharesDataWith(data_blob)
        np.testing.assert_allclose(scale_blob.to_numpy(), 2.0 * inp, rtol=1e-6)

    def test_scale_identity_backward_zerocopy(self, ptrace):
        """Identity Scale backward: bottom diff shares top diff."""
        net = net_from_param(net_param_from_string(
            _make_scale_identity_prototxt((2, 8), bias_term=False)))
        inp = np.random.RandomState(14).randn(2, 8).astype(np.float32)
        dy = np.random.RandomState(15).randn(2, 8).astype(np.float32)
        with ptrace("FwdBwd(scale_cow_bw)"):
            net.Forward({"data": inp})
            net.backward({"scale_out": dy})
        data_blob = net.blob_by_name("data")
        scale_blob = net.blob_by_name("scale_out")
        assert data_blob.IsDiffShared(), \
            "Identity Scale backward: bottom diff must be shared"
        assert data_blob.SharesDiffWith(scale_blob), \
            "Identity Scale backward: bottom diff shares top diff"
        np.testing.assert_array_equal(data_blob.diff, dy)

    def test_scale_downstream_inplace_relu_cow_isolation(self, ptrace):
        """In-place ReLU downstream triggers COW; bottom data stays intact."""
        dims = (2, 8)
        net = net_from_param(net_param_from_string(
            _make_scale_relu_prototxt(dims)))
        inp = np.random.RandomState(16).randn(*dims).astype(np.float32)
        with ptrace("Forward(scale_cow_relu)"):
            net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        scale_blob = net.blob_by_name("scale_out")
        assert not scale_blob.IsDataShared(), \
            "In-place ReLU on scale_out must trigger COW, breaking sharing"
        assert scale_blob.data_tensor.ctypes.data != data_blob.data_tensor.ctypes.data
        np.testing.assert_array_equal(data_blob.to_numpy(), inp)
        np.testing.assert_array_equal(scale_blob.to_numpy(), np.maximum(inp, 0))

    def test_bias_identity_forward_zerocopy(self, ptrace):
        """Bias(bias=0) forward: top shares bottom's data pointer."""
        net = net_from_param(net_param_from_string(
            _make_bias_identity_prototxt((2, 8))))
        inp = np.random.RandomState(17).randn(2, 8).astype(np.float32)
        with ptrace("Forward(bias_cow_identity)"):
            net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        bias_blob = net.blob_by_name("bias_out")
        assert bias_blob.IsDataShared()
        assert bias_blob.SharesDataWith(data_blob)
        assert bias_blob.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data
        np.testing.assert_array_equal(bias_blob.to_numpy(), inp)

    def test_bias_nonidentity_no_cow(self):
        """Bias(bias=2.0) forward: real copy, no COW sharing."""
        net = net_from_param(net_param_from_string(
            _make_bias_nonidentity_prototxt((2, 8))))
        inp = np.random.RandomState(18).randn(2, 8).astype(np.float32)
        net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        bias_blob = net.blob_by_name("bias_out")
        assert not bias_blob.IsDataShared()
        assert not bias_blob.SharesDataWith(data_blob)
        np.testing.assert_allclose(bias_blob.to_numpy(), inp + 2.0, rtol=1e-6)

    def test_bias_identity_backward_zerocopy(self, ptrace):
        """Identity Bias backward: bottom diff shares top diff."""
        net = net_from_param(net_param_from_string(
            _make_bias_identity_prototxt((2, 8))))
        inp = np.random.RandomState(19).randn(2, 8).astype(np.float32)
        dy = np.random.RandomState(20).randn(2, 8).astype(np.float32)
        with ptrace("FwdBwd(bias_cow_bw)"):
            net.Forward({"data": inp})
            net.backward({"bias_out": dy})
        data_blob = net.blob_by_name("data")
        bias_blob = net.blob_by_name("bias_out")
        assert data_blob.IsDiffShared()
        assert data_blob.SharesDiffWith(bias_blob)
        np.testing.assert_array_equal(data_blob.diff, dy)

    def test_eltwise_identity_forward_zerocopy(self, ptrace):
        """Single-bottom Eltwise(coeff=1) forward: top shares bottom's data."""
        net = net_from_param(net_param_from_string(
            _make_eltwise_identity_prototxt((2, 8), coeff=1.0)))
        inp = np.random.RandomState(21).randn(2, 8).astype(np.float32)
        with ptrace("Forward(eltwise_cow_identity)"):
            net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        elt_blob = net.blob_by_name("eltwise_out")
        assert elt_blob.IsDataShared()
        assert elt_blob.SharesDataWith(data_blob)
        assert elt_blob.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data
        np.testing.assert_array_equal(elt_blob.to_numpy(), inp)

    def test_eltwise_nonidentity_no_cow(self):
        """Single-bottom Eltwise(coeff=2.0) forward: real copy, no COW sharing."""
        net = net_from_param(net_param_from_string(
            _make_eltwise_identity_prototxt((2, 8), coeff=2.0)))
        inp = np.random.RandomState(22).randn(2, 8).astype(np.float32)
        net.Forward({"data": inp})
        data_blob = net.blob_by_name("data")
        elt_blob = net.blob_by_name("eltwise_out")
        assert not elt_blob.IsDataShared()
        assert not elt_blob.SharesDataWith(data_blob)
        np.testing.assert_allclose(elt_blob.to_numpy(), 2.0 * inp, rtol=1e-6)

    def test_eltwise_identity_backward_zerocopy(self, ptrace):
        """Identity Eltwise backward: bottom diff shares top diff."""
        net = net_from_param(net_param_from_string(
            _make_eltwise_identity_prototxt((2, 8), coeff=1.0)))
        inp = np.random.RandomState(23).randn(2, 8).astype(np.float32)
        dy = np.random.RandomState(24).randn(2, 8).astype(np.float32)
        with ptrace("FwdBwd(eltwise_cow_bw)"):
            net.Forward({"data": inp})
            net.backward({"eltwise_out": dy})
        data_blob = net.blob_by_name("data")
        elt_blob = net.blob_by_name("eltwise_out")
        assert data_blob.IsDiffShared()
        assert data_blob.SharesDiffWith(elt_blob)
        np.testing.assert_array_equal(data_blob.diff, dy)