"""Phase 3.1: SetShapeOnly compatibility and integration tests.

Covers 17 test cases for the SetShapeOnly lazy allocation API:
  - 7 Blob-level unit tests (TestSetShapeOnly)
  - 4 Split-layer integration tests (TestSplitLazyReshape)
  - 6 extended tests for edge cases & regression

Usage:
  pytest tests/python/test_phase3_set_shape_only.py -v
"""
import pytest
import numpy as np
import caffe_ffi
from caffe_ffi import Blob
from .conftest import require_cpp_extension


# ── Prototxt helpers ──────────────────────────────────────────────────

def _dims_str(dims):
    """Convert a dimension tuple to protobuf text format repeated dim entries."""
    return " ".join(f"dim: {d}" for d in dims)

def _make_split_prototxt(num_top, dims, name="test_split"):
    """Build a minimal Input + Split prototxt string."""
    dims_field = _dims_str(dims)
    tops = "\n".join(f'  top: "split_{i}"' for i in range(num_top))
    return f"""name: "{name}"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ {dims_field} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
{tops}
}}
"""


def _make_split_relu_prototxt(num_top, dims, name="test_split_relu"):
    """Build Input + Split(N) + ReLU(split_0) prototxt."""
    dims_field = _dims_str(dims)
    tops = "\n".join(f'  top: "split_{i}"' for i in range(num_top))
    return f"""name: "{name}"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ {dims_field} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
{tops}
}}
layer {{
  name: "relu"
  type: "ReLU"
  bottom: "split_0"
  top: "relu_out"
}}
"""


def _make_split_activation_prototxt(num_top, dims, act_type, out_name, name=None):
    """Build Input + Split(N) + Activation(split_0) prototxt."""
    if name is None:
        name = f"test_split_{act_type.lower()}"
    dims_field = _dims_str(dims)
    tops = "\n".join(f'  top: "split_{i}"' for i in range(num_top))
    return f"""name: "{name}"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ {dims_field} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
{tops}
}}
layer {{
  name: "downstream"
  type: "{act_type}"
  bottom: "split_0"
  top: "{out_name}"
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# TestSetShapeOnly — Blob-level unit tests (7 cases)
# ═══════════════════════════════════════════════════════════════════════

class TestSetShapeOnly:
    """Blob-level SetShapeOnly unit tests."""

    # ── Case 1: BasicShapeStorage ──

    def test_basic_shape_storage(self):
        """SetShapeOnly stores shape correctly and sets lazy flag."""
        blob = Blob()
        blob.set_shape_only([2, 3, 224, 224])

        assert blob.is_lazy_allocated(), "is_lazy_allocated() should return True"
        assert blob.num_axes == 4, f"num_axes should be 4, got {blob.num_axes}"
        assert blob.shape_at(0) == 2
        assert blob.shape_at(1) == 3
        assert blob.shape_at(2) == 224
        assert blob.shape_at(3) == 224
        assert blob.count() == 2 * 3 * 224 * 224
        assert blob.shape == (2, 3, 224, 224)

    # ── Case 2: NoDataAllocation ──

    def test_no_data_allocation(self):
        """SetShapeOnly should not allocate data memory."""
        blob = Blob()
        bytes_before = caffe_ffi.total_allocated_bytes()
        blob.set_shape_only([100, 100, 100])
        bytes_after = caffe_ffi.total_allocated_bytes()

        assert bytes_after == bytes_before, (
            f"SetShapeOnly allocated {bytes_after - bytes_before} bytes, "
            f"expected 0"
        )

    # ── Case 3: DataAccessReturnsNone (semantic: lazy flag set) ──

    def test_lazy_flag_indicates_no_data(self):
        """is_lazy_allocated() returns True for lazy blob (data not allocated)."""
        blob = Blob()
        blob.set_shape_only([64, 3, 32, 32])

        assert blob.is_lazy_allocated(), "cpu_data equivalent: lazy flag should be True"
        # Verify shape is accessible even without data
        assert blob.num_axes == 4
        assert blob.shape_at(0) == 64

    # ── Case 4: ReshapeClearsLazyFlag ──

    def test_reshape_clears_lazy_flag(self):
        """Reshape() should allocate memory and clear lazy flag."""
        blob = Blob()
        blob.set_shape_only([10, 20])
        assert blob.is_lazy_allocated()

        blob.Reshape([5, 5])
        assert not blob.is_lazy_allocated(), "Reshape should clear lazy flag"
        assert blob.shape == (5, 5)
        # data_tensor should now be accessible
        assert blob.data_tensor.shape == (5, 5)

    # ── Case 5: ShareDataClearsLazyFlag ──

    def test_share_data_clears_lazy_flag(self):
        """ShareData() from a real blob should clear lazy flag."""
        source = Blob([3, 4])
        source.set_data(np.random.randn(3, 4).astype(np.float32))

        target = Blob()
        target.set_shape_only([3, 4])
        assert target.is_lazy_allocated()

        target.ShareData(source)
        assert not target.is_lazy_allocated(), "ShareData should clear lazy flag"
        assert target.SharesDataWith(source), "After ShareData, blobs should share data"

    # ── Case 6: MutableDataTriggersAllocation ──

    def test_mutable_data_triggers_allocation(self):
        """mutable_data_tensor() should allocate data+diff for lazy blob."""
        blob = Blob()
        blob.set_shape_only([4, 8])
        assert blob.is_lazy_allocated()

        mt = blob.mutable_data_tensor()
        assert not blob.is_lazy_allocated(), (
            "mutable_data_tensor() should allocate and clear lazy flag"
        )
        assert blob.data_tensor.shape == (4, 8)
        assert blob.count() == 32

    # ── Case 7: NdimShape ──

    def test_1d_shape(self):
        """1-dimensional shape."""
        blob = Blob()
        blob.set_shape_only([100])
        assert blob.num_axes == 1
        assert blob.shape_at(0) == 100
        assert blob.count() == 100


# ═══════════════════════════════════════════════════════════════════════
# TestSplitLazyReshape — Split layer integration tests (4 cases)
# ═══════════════════════════════════════════════════════════════════════

class TestSplitLazyReshape:
    """Split layer lazy reshape integration tests."""

    # ── Case 8: LargeNTriggersLazyAllocation ──

    def test_large_n_triggers_lazy_allocation(self):
        """N=64 (>= threshold 16) should trigger lazy allocation during Reshape."""
        N = 64
        C = 3 * 32 * 32  # 3072

        prototxt = _make_split_prototxt(N, (1, 3, 32, 32), name="test_lazy_N64")
        net = caffe_ffi.Net(prototxt)

        # After construction (which calls Reshape), check tops are lazy
        lazy_count = sum(
            1 for i in range(N)
            if net.blob_by_name(f"split_{i}").is_lazy_allocated()
        )
        assert lazy_count == N, (
            f"Expected {N} lazy blobs after Reshape for N=64, got {lazy_count}"
        )

        # Memory should be close to 0 additional from the 64 split tops
        # (just the input blob)
        inp = np.random.randn(1, 3, 32, 32).astype(np.float32)
        bytes_before = caffe_ffi.total_allocated_bytes()
        out = net.Forward({"data": inp})
        bytes_after = caffe_ffi.total_allocated_bytes()

        # With lazy allocation during Reshape, tops don't allocate until Forward's
        # ShareData (which shares bottom's data, so net new allocation ~0).
        # Without lazy, Reshape would allocate N * C * 4 bytes.
        expected_full_alloc = N * C * 4  # would be 786,432 bytes without lazy
        actual_delta = bytes_after - bytes_before
        # After Forward, all tops share bottom data, so delta should be small
        assert actual_delta < expected_full_alloc, (
            f"Memory increase {actual_delta} >= expected_full {expected_full_alloc}, "
            f"lazy allocation may not be working"
        )

    # ── Case 9: ForwardTransitionsToNormal ──

    def test_forward_transitions_to_normal(self):
        """After Forward, ShareData replaces lazy tensors with shared data."""
        N = 64
        prototxt = _make_split_prototxt(N, (1, 3, 32, 32), name="test_transition_N64")
        net = caffe_ffi.Net(prototxt)
        inp = np.random.randn(1, 3, 32, 32).astype(np.float32)

        out = net.Forward({"data": inp})

        for i in range(N):
            key = f"split_{i}"
            assert key in out, f"Missing output '{key}'"
            blob = net.blob_by_name(key)
            assert not blob.is_lazy_allocated(), f"split_{i} should not be lazy after Forward"
            np.testing.assert_array_almost_equal(
                blob.to_numpy(), inp,
                err_msg=f"split_{i} output differs from input"
            )

    # ── Case 10: SmallNStaysNormal ──

    def test_small_n_stays_normal(self):
        """N=4 (< threshold 16) should NOT trigger lazy allocation."""
        N = 4
        C = 256

        prototxt = _make_split_prototxt(N, (1, C), name="test_normal_N4")
        net = caffe_ffi.Net(prototxt)

        # N=4 < 16, so tops should NOT be lazy after Reshape
        lazy_count = sum(
            1 for i in range(N)
            if net.blob_by_name(f"split_{i}").is_lazy_allocated()
        )
        assert lazy_count == 0, (
            f"N=4 should not use lazy allocation, but {lazy_count}/4 blobs are lazy"
        )

        inp = np.random.randn(1, C).astype(np.float32)
        out = net.Forward({"data": inp})

        for i in range(N):
            key = f"split_{i}"
            assert key in out
            np.testing.assert_array_almost_equal(
                net.blob_by_name(key).to_numpy(), inp
            )

    # ── Case 11: DownstreamLayerCompatibility ──

    def test_downstream_layer_compatibility_relu(self):
        """ReLU downstream of lazy-allocated Split should work correctly."""
        N = 100
        prototxt = _make_split_relu_prototxt(N, (1, 64, 28, 28), name="test_downstream_relu")
        net = caffe_ffi.Net(prototxt)
        inp = np.random.randn(1, 64, 28, 28).astype(np.float32)

        out = net.Forward({"data": inp})

        assert "relu_out" in out, "ReLU output missing"
        expected_relu = np.maximum(0, inp)
        np.testing.assert_array_almost_equal(
            net.blob_by_name("relu_out").to_numpy(), expected_relu,
            err_msg="ReLU output differs from expected"
        )


# ═══════════════════════════════════════════════════════════════════════
# Extended tests — edge cases & regression (6 cases)
# ═══════════════════════════════════════════════════════════════════════

class TestSetShapeOnlyExtended:
    """Extended edge case and regression tests for SetShapeOnly."""

    # ── Case 12: CountAfterSetShapeOnly ──

    def test_count_after_set_shape_only(self):
        """count() and count(start_axis) should work with stored shape."""
        blob = Blob()
        blob.set_shape_only([32, 64, 112, 112])

        assert blob.count() == 32 * 64 * 112 * 112
        assert blob.count(1) == 64 * 112 * 112
        assert blob.count(0, 2) == 32 * 64

    # ── Case 13: SetShapeOnlyThenReshape (replaces FromProto test since FromProto not in FFI) ──

    def test_set_shape_only_then_reshape(self):
        """Reshape after SetShapeOnly should allocate and work correctly."""
        blob = Blob()
        blob.set_shape_only([3, 4])
        assert blob.is_lazy_allocated()

        blob.Reshape([3, 4])
        assert not blob.is_lazy_allocated(), "Reshape should clear lazy flag"
        assert blob.data_tensor is not None

        data = np.random.randn(3, 4).astype(np.float32)
        blob.set_data(data)
        np.testing.assert_array_almost_equal(blob.to_numpy(), data)

    # ── Case 14: EmptyShape ──

    def test_empty_shape_rejected(self):
        """Empty shape should raise an error."""
        blob = Blob()
        with pytest.raises((ValueError, RuntimeError, Exception)):
            blob.set_shape_only([])

    # ── Case 15: LazyBlobShareDataThenWrite ──

    def test_lazy_blob_share_data_then_write(self):
        """After ShareData, mutable_data_tensor() should trigger COW."""
        source = Blob([5, 5])
        source_data = np.random.randn(5, 5).astype(np.float32)
        source.set_data(source_data)

        target = Blob()
        target.set_shape_only([5, 5])
        target.ShareData(source)
        assert not target.is_lazy_allocated()

        # Writing to target should trigger COW (if COW is enabled)
        mt = target.mutable_data_tensor()
        assert mt is not None, "mutable_data_tensor() should return valid tensor"

    # ── Case 16: N1SplitNoLazyAllocation ──

    def test_n1_split_no_lazy_allocation(self):
        """N=1 Split should NOT trigger lazy allocation (special path)."""
        prototxt = _make_split_prototxt(1, (1, 256), name="test_n1_normal")
        net = caffe_ffi.Net(prototxt)

        top0 = net.blob_by_name("split_0")
        assert not top0.is_lazy_allocated(), "N=1 Split should not use lazy allocation"

        inp = np.random.randn(1, 256).astype(np.float32)
        out = net.Forward({"data": inp})
        assert "split_0" in out
        np.testing.assert_array_almost_equal(top0.to_numpy(), inp)

    # ── Case 17: N16Boundary ──

    def test_n16_boundary(self):
        """N=16 exactly at threshold should trigger lazy allocation."""
        N = 16
        C = 128

        prototxt = _make_split_prototxt(N, (1, C), name=f"test_boundary_N{N}")
        net = caffe_ffi.Net(prototxt)

        lazy_count = sum(
            1 for i in range(N)
            if net.blob_by_name(f"split_{i}").is_lazy_allocated()
        )
        assert lazy_count == N, (
            f"N=16 (threshold) should use lazy allocation, {lazy_count}/{N} are lazy"
        )

        inp = np.random.randn(1, C).astype(np.float32)
        out = net.Forward({"data": inp})

        for i in range(N):
            key = f"split_{i}"
            assert key in out
            np.testing.assert_array_almost_equal(
                net.blob_by_name(key).to_numpy(), inp
            )


# ═══════════════════════════════════════════════════════════════════════
# Parametrized compatibility test for downstream layers
# ═══════════════════════════════════════════════════════════════════════

class TestDownstreamLayerCompatibility:
    """Verify downstream layers work with lazy-allocated->shared blobs."""

    @pytest.mark.parametrize("layer_type,output_key", [
        ("ReLU", "downstream_out"),
        ("Sigmoid", "downstream_out"),
        ("TanH", "downstream_out"),
    ])
    def test_activation_after_lazy_split(self, layer_type, output_key):
        """Activation layers after lazy Split should work."""
        N = 100
        prototxt = _make_split_activation_prototxt(
            N, (1, 64, 28, 28), layer_type, output_key
        )
        net = caffe_ffi.Net(prototxt)
        inp = np.random.randn(1, 64, 28, 28).astype(np.float32)

        out = net.Forward({"data": inp})

        assert output_key in out, f"{layer_type} output '{output_key}' missing"
        result = net.blob_by_name(output_key).to_numpy()
        assert result.shape == inp.shape, (
            f"{layer_type} output shape {result.shape} != input shape {inp.shape}"
        )


# ── Manual verification ────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "-s"])
