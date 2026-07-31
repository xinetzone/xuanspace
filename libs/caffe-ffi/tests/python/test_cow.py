"""P2-B: Copy-on-Write (COW) extension tests.

Tests cover:
- TestBlobCOWApi: Python-side COW API validation (IsDataShared, DataRefCount,
  UnshareData, mutable_data_tensor, cow_snapshot helper)
- TestSplitCOWBehavior: Split-layer COW integration tests (N>=2 data isolation,
  refcount verification, const access no-COW, COW after in-place ReLU)
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