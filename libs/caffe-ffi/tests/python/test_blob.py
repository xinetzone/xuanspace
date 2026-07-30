from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import Blob
from .conftest import require_cpp_extension


class TestBlobReshape:
    def test_reshape_1d(self):
        b = Blob()
        b.Reshape([5])
        assert b.shape == (5,)
        assert b.ndim == 1
        assert b.size == 5

    def test_reshape_2d(self):
        b = Blob()
        b.Reshape([2, 3])
        assert b.shape == (2, 3)
        assert b.ndim == 2
        assert b.size == 6

    def test_reshape_4d(self):
        b = Blob()
        b.Reshape([1, 2, 3, 4])
        assert b.shape == (1, 2, 3, 4)
        assert b.ndim == 4
        assert b.size == 24

    def test_reshape_changes_size(self):
        b = Blob([2, 3])
        assert b.size == 6
        b.Reshape([4, 5])
        assert b.shape == (4, 5)
        assert b.size == 20


class TestBlobNumpy:
    def test_from_numpy_to_numpy(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        b = Blob()
        b.from_numpy(arr)
        assert b.shape == (2, 3)
        result = b.to_numpy()
        np.testing.assert_array_equal(result, arr)

    def test_from_numpy_creates_copy(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        b = Blob()
        b.from_numpy(arr)
        arr[0] = 999
        assert b.to_numpy()[0] != 999

    def test_to_numpy_creates_copy(self):
        b = Blob()
        b.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        result = b.to_numpy()
        result[0] = 999
        assert b.to_numpy()[0] != 999

    def test_data_property(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = Blob()
        b.data = arr
        np.testing.assert_array_equal(b.data, arr)

    def test_data_setter_reshape(self):
        b = Blob([5])
        new_data = np.ones((2, 3), dtype=np.float32)
        b.data = new_data
        assert b.shape == (2, 3)
        np.testing.assert_array_equal(b.data, new_data)

    def test_diff_property(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        b = Blob()
        b.diff = arr
        np.testing.assert_array_equal(b.diff, arr)


class TestBlobFill:
    def test_fill(self):
        b = Blob([2, 3])
        b.fill(3.14)
        assert np.all(b.data == 3.14)

    def test_zero(self):
        b = Blob([2, 3])
        b.fill(1.0)
        b.zero()
        assert np.all(b.data == 0.0)

    def test_fill_zeros_diff(self):
        """fill() must zero the diff tensor."""
        b = Blob([2, 3])
        b.diff_tensor[:] = 9.99
        b.fill(1.0)
        assert np.all(b.diff_tensor == 0.0)

    def test_fill_returns_self(self):
        """fill() returns self for chaining."""
        b = Blob([2, 3])
        result = b.fill(5.0)
        assert result is b

    def test_fill_negative_value(self):
        b = Blob([3])
        b.fill(-1.5)
        np.testing.assert_allclose(b.data, [-1.5, -1.5, -1.5], rtol=1e-6)

    def test_fill_zero_value(self):
        b = Blob([2, 2])
        b.fill(0.0)
        assert np.all(b.data == 0.0)

    def test_fill_large_value(self):
        b = Blob([1])
        b.fill(1e6)
        assert abs(b.data[0] - 1e6) / 1e6 < 1e-5

    def test_fill_data_tensor_reflects_value(self):
        """fill writes through to data_tensor (zero-copy view)."""
        b = Blob([2, 3])
        b.fill(7.77)
        dt = b.data_tensor
        assert np.all(dt == np.float32(7.77))

    def test_fill_then_overwrite(self):
        b = Blob([3])
        b.fill(1.0)
        b.fill(2.0)
        assert np.all(b.data == 2.0)

    def test_fill_int_coerced_to_float32(self):
        b = Blob([2])
        b.fill(42)
        assert b.data_tensor.dtype == np.float32
        assert np.all(b.data == 42.0)


class TestBlobFromNumpyComprehensive:
    """Comprehensive tests for Blob.from_numpy() covering data conversion correctness."""

    def test_from_numpy_1d(self):
        arr = np.arange(10, dtype=np.float32)
        b = Blob()
        b.from_numpy(arr)
        assert b.shape == (10,)
        np.testing.assert_array_equal(b.to_numpy(), arr)

    def test_from_numpy_2d(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        b = Blob()
        b.from_numpy(arr)
        assert b.shape == (2, 3)
        np.testing.assert_array_equal(b.to_numpy(), arr)

    def test_from_numpy_4d(self):
        arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        b = Blob()
        b.from_numpy(arr)
        assert b.shape == (2, 3, 4, 5)
        np.testing.assert_allclose(b.to_numpy(), arr, rtol=1e-6)

    def test_from_numpy_int_converts_to_float32(self):
        """Integer numpy arrays must be converted to float32."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        b = Blob()
        b.from_numpy(arr)
        assert b.data_tensor.dtype == np.float32
        np.testing.assert_array_equal(b.to_numpy(), [1.0, 2.0, 3.0])

    def test_from_numpy_float64_converts_to_float32(self):
        """float64 arrays must be downcast to float32."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        b = Blob()
        b.from_numpy(arr)
        assert b.data_tensor.dtype == np.float32
        np.testing.assert_allclose(b.to_numpy(), [1.5, 2.5, 3.5], rtol=1e-6)

    def test_from_numpy_list_input(self):
        """from_numpy should accept Python lists and convert them."""
        b = Blob()
        b.from_numpy([1.0, 2.0, 3.0])
        assert b.shape == (3,)
        np.testing.assert_allclose(b.to_numpy(), [1.0, 2.0, 3.0], rtol=1e-6)

    def test_from_numpy_set_diff_true(self):
        """set_diff=True sets diff tensor instead of data tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = Blob()
        b.from_numpy(arr, set_diff=True)
        assert b.shape == (3,)
        np.testing.assert_array_equal(b.diff, arr)
        assert np.all(b.data_tensor == 0.0), "data should remain zero when set_diff=True"

    def test_from_numpy_set_diff_false_default(self):
        """Default (set_diff=False) sets data tensor."""
        arr = np.array([10.0, 20.0], dtype=np.float32)
        b = Blob()
        b.from_numpy(arr)
        np.testing.assert_array_equal(b.data, arr)
        assert np.all(b.diff_tensor == 0.0), "diff should be zero for default from_numpy"

    def test_from_numpy_reshapes_existing_blob(self):
        """from_numpy must reshape the blob to match the input array."""
        b = Blob([5, 5])
        new_arr = np.zeros((2, 3), dtype=np.float32)
        b.from_numpy(new_arr)
        assert b.shape == (2, 3)

    def test_from_numpy_returns_self(self):
        """from_numpy returns self for chaining."""
        b = Blob()
        result = b.from_numpy(np.array([1.0], dtype=np.float32))
        assert result is b

    def test_from_numpy_chain_fill(self):
        """from_numpy followed by fill should work as chain."""
        b = Blob()
        b.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).fill(0.0)
        assert np.all(b.data == 0.0)
        assert b.shape == (2,)

    def test_from_numpy_preserves_values_after_reshape(self):
        """Data written via from_numpy should be readable back correctly."""
        arr = np.random.randn(4, 5).astype(np.float32)
        b = Blob()
        b.from_numpy(arr)
        result = b.to_numpy()
        np.testing.assert_allclose(result, arr, rtol=1e-6)

    def test_from_numpy_scalar_shape(self):
        """from_numpy with a single-element array."""
        b = Blob()
        b.from_numpy(np.array([42.0], dtype=np.float32))
        assert b.shape == (1,)
        assert abs(b.to_numpy()[0] - 42.0) < 1e-6

    def test_data_setter_dtype_conversion(self):
        """Setting data with non-float32 dtype should convert."""
        b = Blob([3])
        b.data = np.array([1, 2, 3], dtype=np.int64)
        assert b.data_tensor.dtype == np.float32
        np.testing.assert_array_equal(b.data, [1.0, 2.0, 3.0])


class TestBlobCopy:
    def test_copy_from(self):
        b1 = Blob([2, 3])
        b1.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        b2 = Blob()
        b2.copy_from(b1)
        assert b2.shape == b1.shape
        np.testing.assert_array_equal(b2.data, b1.data)

    def test_copy_from_is_independent(self):
        b1 = Blob([2, 3])
        b1.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        b2 = Blob()
        b2.copy_from(b1)
        b1.fill(0)
        assert not np.all(b2.data == 0)


class TestBlobProperties:
    def test_shape(self):
        b = Blob([2, 3, 4])
        assert b.shape == (2, 3, 4)

    def test_ndim(self):
        b = Blob([2, 3, 4])
        assert b.ndim == 3

    def test_size(self):
        b = Blob([2, 3, 4])
        assert b.size == 24

    def test_num_axes(self):
        b = Blob([2, 3, 4])
        assert b.num_axes == 3


class TestBlobRepr:
    def test_repr(self):
        b = Blob([2, 3])
        r = repr(b)
        assert "Blob" in r
        assert "(2, 3)" in r


@require_cpp_extension
class TestBlobMemoryCounters:
    """Tests for memory counter correctness after AllocData/FreeData migration.

    These tests specifically guard against the global_before ordering bug
    where Reshape() read the counter after subtracting old bytes, causing
    global_before to show 0B instead of the correct pre-reshape value.
    The fix moved counter management into AllocData/FreeData primitives
    so ordering is guaranteed by construction.
    """

    @staticmethod
    def _expected_nbytes(shape):
        count = 1
        for d in shape:
            count *= d
        return count * 4 * 2

    def test_initial_alloc_counter(self):
        import gc
        gc.collect(); gc.collect(); gc.collect()
        mem_before = caffe_ffi.total_allocated_bytes()
        live_before = caffe_ffi.live_blob_count()
        b = Blob([3, 4])
        mem_after = caffe_ffi.total_allocated_bytes()
        live_after = caffe_ffi.live_blob_count()
        expected = self._expected_nbytes([3, 4])
        assert mem_after - mem_before == expected, \
            f"Expected +{expected}B after Blob([3,4]), got +{mem_after - mem_before}B"
        assert live_after == live_before + 1, \
            f"Expected +1 live blob, got +{live_after - live_before}"
        del b
        gc.collect(); gc.collect(); gc.collect()
        assert caffe_ffi.total_allocated_bytes() == mem_before
        assert caffe_ffi.live_blob_count() == live_before

    def test_reshape_counter_delta(self):
        b = Blob([3, 4])
        mem_before_reshape = caffe_ffi.total_allocated_bytes()
        old_expected = self._expected_nbytes([3, 4])
        assert mem_before_reshape >= old_expected
        b.Reshape([5, 6])
        mem_after_reshape = caffe_ffi.total_allocated_bytes()
        new_expected = self._expected_nbytes([5, 6])
        delta = mem_after_reshape - mem_before_reshape
        expected_delta = new_expected - old_expected
        assert delta == expected_delta, \
            f"Reshape delta should be {expected_delta}B ({old_expected}->{new_expected}), got {delta}B"

    def test_reshape_to_zero_frees_memory(self):
        b = Blob([2, 3])
        mem_with_blob = caffe_ffi.total_allocated_bytes()
        assert mem_with_blob > 0
        b.Reshape([0])
        mem_after_zero = caffe_ffi.total_allocated_bytes()
        assert mem_after_zero == mem_with_blob - self._expected_nbytes([2, 3]), \
            "Reshape to [0] should free all tensor memory"

    def test_reshape_same_shape_no_delta(self):
        b = Blob([4, 5])
        mem_before = caffe_ffi.total_allocated_bytes()
        b.Reshape([4, 5])
        mem_after = caffe_ffi.total_allocated_bytes()
        assert mem_after == mem_before, \
            f"Reshape to same shape should not change counter: before={mem_before}, after={mem_after}"

    def test_destructor_frees_memory(self):
        mem_before = caffe_ffi.total_allocated_bytes()
        live_before = caffe_ffi.live_blob_count()
        b = Blob([10, 10])
        mem_after_create = caffe_ffi.total_allocated_bytes()
        expected = self._expected_nbytes([10, 10])
        assert mem_after_create - mem_before == expected
        del b
        mem_after_delete = caffe_ffi.total_allocated_bytes()
        live_after = caffe_ffi.live_blob_count()
        assert mem_after_delete == mem_before, \
            f"After del Blob, counter should return to {mem_before}, got {mem_after_delete}"
        assert live_after == live_before, \
            f"After del Blob, live count should return to {live_before}, got {live_after}"

    def test_multiple_blobs_additive(self):
        mem_before = caffe_ffi.total_allocated_bytes()
        b1 = Blob([2, 2])
        b2 = Blob([3, 3])
        b3 = Blob([4, 4])
        expected = (self._expected_nbytes([2, 2]) +
                    self._expected_nbytes([3, 3]) +
                    self._expected_nbytes([4, 4]))
        mem_after = caffe_ffi.total_allocated_bytes()
        assert mem_after - mem_before == expected, \
            f"3 Blobs should allocate {expected}B, got {mem_after - mem_before}B"
        del b2
        mem_after_del = caffe_ffi.total_allocated_bytes()
        expected_after_del = expected - self._expected_nbytes([3, 3])
        assert mem_after_del - mem_before == expected_after_del, \
            f"After del b2, should have {expected_after_del}B, got {mem_after_del - mem_before}B"
        del b1
        del b3

    def test_memory_info_dict(self):
        info = caffe_ffi.memory_info()
        assert "total_allocated_bytes" in info
        assert "live_blob_count" in info
        assert isinstance(info["total_allocated_bytes"], int)
        assert isinstance(info["live_blob_count"], int)

    def test_reshape_grow_shrink_cycle(self):
        """Test that repeated grow/shrink cycles maintain accurate counters.

        This is the core regression test for the global_before ordering bug:
        if global_before is read after FreeData subtracts old bytes (the bug),
        the counter would show incorrect intermediate values during Reshape.
        With AllocData/FreeData managing the counter automatically, all
        intermediate states are correct by construction.
        """
        b = Blob([1])
        shapes = [(2, 3), (5, 5), (10, 10), (3, 3), (1,)]
        expected_offsets = []
        for shape in shapes:
            b.Reshape(list(shape))
            expected = self._expected_nbytes(list(shape))
            actual = caffe_ffi.total_allocated_bytes()
            assert actual >= expected, \
                f"After Reshape({shape}), counter ({actual}) should be >= blob size ({expected})"
            expected_offsets.append(actual)
        del b

    def test_live_blob_count(self):
        live_before = caffe_ffi.live_blob_count()
        b1 = Blob([1])
        assert caffe_ffi.live_blob_count() == live_before + 1
        b2 = Blob([2])
        assert caffe_ffi.live_blob_count() == live_before + 2
        b3 = Blob([3])
        assert caffe_ffi.live_blob_count() == live_before + 3
        del b2
        assert caffe_ffi.live_blob_count() == live_before + 2
        del b1
        assert caffe_ffi.live_blob_count() == live_before + 1
        del b3
        assert caffe_ffi.live_blob_count() == live_before


@require_cpp_extension
class TestBlobLifecycle:
    """Integration test simulating a complete Blob lifecycle:
    create → numpy load → modification → gradient update → copy → resize → serialize → release.

    This test exercises every major Blob API in sequence, verifying data
    integrity at each stage and memory correctness at creation/destruction.
    It mimics a realistic mini-batch training step: load weights → forward
    → set gradients → SGD update → copy snapshot → reshape for next batch.
    """

    @staticmethod
    def _blob_nbytes(shape):
        """Expected bytes for a Blob with given shape (float32 data + float32 diff)."""
        count = 1
        for d in shape:
            count *= d
        return count * 4 * 2

    def test_full_lifecycle_training_step(self):
        import gc

        def _gc():
            """Force full garbage collection (3 passes, matching conftest._current_mem_state)."""
            gc.collect()
            gc.collect()
            gc.collect()

        def _mem():
            return caffe_ffi.total_allocated_bytes()

        def _live():
            return caffe_ffi.live_blob_count()

        # ── Phase 0: Baseline (must gc first to flush any previous test residue)
        _gc()
        mem_baseline = _mem()
        live_baseline = _live()

        # ── Phase 1: Creation (empty constructor → Reshape[0] → 0 data bytes) ──
        blob = Blob()
        assert blob.shape == (0,), "Default Blob should have shape (0,)"
        assert blob.size == 0
        assert _live() == live_baseline + 1
        # Default Blob (shape [0]) allocates 0 data bytes; only C++ object overhead
        # which is NOT counted in total_allocated_bytes (tracks tensor data only).
        assert _mem() == mem_baseline, \
            f"Empty Blob([0]) should allocate 0 tensor bytes, got +{_mem() - mem_baseline}"

        # ── Phase 2: from_numpy loads 4×3 weights (96 bytes data+diff) ─────
        weights = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ], dtype=np.float32)
        blob.from_numpy(weights)
        assert blob.shape == (4, 3)
        assert blob.size == 12
        assert blob.ndim == 2
        assert blob.data_tensor.dtype == np.float32
        np.testing.assert_array_equal(blob.to_numpy(), weights)

        expected_4x3 = self._blob_nbytes([4, 3])  # 12*4*2 = 96
        assert _mem() == mem_baseline + expected_4x3, \
            f"After from_numpy(4x3), expected +{expected_4x3}B, got +{_mem() - mem_baseline}B"

        # ── Phase 3: Forward-pass modifications (same shape → no realloc) ────
        blob.data_tensor[:] += np.float32(0.01)
        expected_after_bias = weights + 0.01
        np.testing.assert_allclose(blob.data_tensor, expected_after_bias, rtol=1e-6)

        blob.fill(0.5)
        assert np.all(blob.data_tensor == np.float32(0.5))
        assert np.all(blob.diff_tensor == 0.0), "fill() must zero diff"
        assert _mem() == mem_baseline + expected_4x3, \
            "fill() must not change allocation size"

        blob.from_numpy(weights)
        np.testing.assert_array_equal(blob.to_numpy(), weights)
        assert _mem() == mem_baseline + expected_4x3, \
            "from_numpy (same shape) must not change allocation size"

        # ── Phase 4: Set gradients via set_diff=True (same shape → no realloc)
        grads = np.random.RandomState(42).randn(4, 3).astype(np.float32) * 0.01
        blob.from_numpy(grads, set_diff=True)
        np.testing.assert_allclose(blob.diff, grads, rtol=1e-6)
        np.testing.assert_array_equal(blob.data, weights)
        assert blob.diff_tensor.shape == (4, 3)
        assert blob.diff_tensor.dtype == np.float32
        assert _mem() == mem_baseline + expected_4x3, \
            "set_diff=True (same shape) must not change allocation size"

        # ── Phase 5: SGD update (data -= lr * diff), same shape ─────────────
        lr = np.float32(0.1)
        expected_after_update = weights - lr * grads
        blob.diff_tensor[:] *= lr
        blob.Update()
        np.testing.assert_allclose(
            blob.to_numpy(), expected_after_update, rtol=1e-5,
            err_msg="SGD update (data -= lr*grad) produced incorrect values"
        )
        np.testing.assert_allclose(blob.diff, lr * grads, rtol=1e-6)
        assert _mem() == mem_baseline + expected_4x3, \
            "Update() must not change allocation size"

        # ── Phase 6: Copy snapshot (adds another 4×3 blob = +96 bytes) ──────
        snapshot = Blob()
        assert _live() == live_baseline + 2
        snapshot.copy_from(blob)
        assert snapshot.shape == blob.shape
        np.testing.assert_array_equal(snapshot.data, blob.data)
        # snapshot was shape [0] (0B), copy_from reshapes to [4,3] (+96B)
        assert _mem() == mem_baseline + expected_4x3 * 2, \
            f"After copy snapshot, expected +{expected_4x3*2}B, got +{_mem() - mem_baseline}B"
        assert _live() == live_baseline + 2

        # Verify copy independence (mutating original must not affect snapshot)
        blob.fill(999.0)
        assert not np.all(snapshot.data == 999.0), \
            "copy_from must produce an independent copy"
        np.testing.assert_allclose(snapshot.data, expected_after_update, rtol=1e-5)
        assert _mem() == mem_baseline + expected_4x3 * 2, \
            "fill() after copy must not change allocation size"

        blob.copy_from(snapshot)
        np.testing.assert_allclose(blob.data, expected_after_update, rtol=1e-5)

        # ── Phase 7a: Resize grow → 5×4 (blob reallocs: 96→160, delta=+64) ─
        expected_5x4 = self._blob_nbytes([5, 4])  # 20*4*2 = 160
        blob.Reshape([5, 4])
        assert blob.shape == (5, 4)
        assert blob.size == 20
        mem_after_grow = _mem()
        assert mem_after_grow == mem_baseline + expected_5x4 + expected_4x3, \
            f"After Reshape(5,4), expected +{expected_5x4+expected_4x3}B (blob={expected_5x4}+snapshot={expected_4x3}), got +{mem_after_grow - mem_baseline}B"

        new_data = np.arange(20, dtype=np.float32).reshape(5, 4)
        blob.from_numpy(new_data)
        np.testing.assert_array_equal(blob.to_numpy(), new_data)

        # ── Phase 7b: Resize shrink → 2×2 (blob reallocs: 160→32, delta=-128)
        expected_2x2 = self._blob_nbytes([2, 2])  # 4*4*2 = 32
        blob.Reshape([2, 2])
        assert blob.shape == (2, 2)
        mem_after_shrink = _mem()
        assert mem_after_shrink == mem_baseline + expected_2x2 + expected_4x3, \
            f"After Reshape(2,2), expected +{expected_2x2+expected_4x3}B, got +{mem_after_shrink - mem_baseline}B"

        small_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        blob.from_numpy(small_data)
        np.testing.assert_array_equal(blob.to_numpy(), small_data)

        # ── Phase 7c: Reshape to [0] → frees blob's tensor data (-32 bytes) ─
        mem_before_zero = _mem()
        blob.Reshape([0])
        _gc()
        mem_after_zero = _mem()
        assert mem_after_zero == mem_before_zero - expected_2x2, \
            f"Reshape([0]) should free exactly {expected_2x2}B (2x2 data+diff), " \
            f"freed {mem_before_zero - mem_after_zero}B"

        # Restore to non-zero for remaining phases (+32 bytes back)
        blob.from_numpy(small_data)
        assert _mem() == mem_after_zero + expected_2x2, \
            "from_numpy after [0] should restore allocation"

        # ── Phase 8: Serialization roundtrip (restored blob allocates +32B,
        #             then freed → net 0 when properly cleaned up) ───────────
        mem_before_restore = _mem()
        live_before_restore = _live()

        exported = blob.to_numpy()
        assert isinstance(exported, np.ndarray)
        assert exported.dtype == np.float32
        assert exported.shape == (2, 2)

        restored = Blob()
        restored.from_numpy(exported)
        np.testing.assert_array_equal(restored.to_numpy(), small_data)
        assert _live() == live_before_restore + 1
        assert _mem() == mem_before_restore + expected_2x2, \
            f"restored blob should add {expected_2x2}B"

        del restored
        _gc()
        assert _live() == live_before_restore, \
            "restored blob must be freed after del"
        assert _mem() == mem_before_restore, \
            f"restored blob memory must be fully freed (leaked {_mem() - mem_before_restore}B)"

        # ── Phase 9: Cleanup & memory verification ──────────────────────────
        # At this point: blob=[2,2](32B) + snapshot=[4,3](96B) = baseline+128B
        mem_with_both = _mem()
        live_with_both = _live()
        expected_with_both = expected_2x2 + expected_4x3
        assert mem_with_both == mem_baseline + expected_with_both, \
            f"Before cleanup, expected +{expected_with_both}B, got +{mem_with_both - mem_baseline}B"
        assert live_with_both == live_baseline + 2, \
            f"Before cleanup, expected {live_baseline+2} live blobs, got {live_with_both}"

        # Delete snapshot first: frees 96B, live count -1
        del snapshot
        _gc()
        assert _live() == live_with_both - 1, \
            "snapshot must be freed after del"
        assert _mem() == mem_with_both - expected_4x3, \
            f"snapshot should free {expected_4x3}B, freed {mem_with_both - _mem()}B"

        # Delete main blob: frees 32B, live count returns to baseline
        del blob
        _gc()

        live_final = _live()
        mem_final = _mem()
        assert live_final == live_baseline, \
            f"Blob leak: live count {live_final} != baseline {live_baseline}"
        assert mem_final == mem_baseline, \
            f"Memory leak: {mem_final - mem_baseline} bytes not freed after lifecycle " \
            f"(baseline={mem_baseline}, final={mem_final})"

    def test_lifecycle_dtype_conversion_chain(self):
        """Test a lifecycle where data flows through multiple dtype conversions:
        Python list → int64 numpy → float64 numpy → Blob (float32) → verify precision.
        """
        # Start with Python list (inference-style input).
        input_list = [1, 2, 3, 4, 5]

        # Convert through int64 and float64 before reaching Blob.
        arr_int = np.array(input_list, dtype=np.int64)
        arr_f64 = arr_int.astype(np.float64) + 0.5  # [1.5, 2.5, 3.5, 4.5, 5.5]

        blob = Blob()
        blob.from_numpy(arr_f64)
        assert blob.data_tensor.dtype == np.float32
        np.testing.assert_allclose(
            blob.to_numpy(), [1.5, 2.5, 3.5, 4.5, 5.5], rtol=1e-6
        )

        # Modify via zero-copy and verify.
        blob.data_tensor[0] = 99.0
        assert abs(blob.to_numpy()[0] - 99.0) < 0.01

        # set_diff with int array should also convert to float32.
        diff_int = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        blob.from_numpy(diff_int, set_diff=True)
        assert blob.diff_tensor.dtype == np.float32
        np.testing.assert_array_equal(blob.diff, [10.0, 20.0, 30.0, 40.0, 50.0])

        # Chained from_numpy → fill → zero → verify state.
        blob.from_numpy(np.ones(3, dtype=np.float32)).fill(42.0).zero()
        assert np.all(blob.data == 0.0)
        assert blob.shape == (3,)
        assert np.all(blob.diff_tensor == 0.0)


class TestBlobZeroCopy:
    def test_data_tensor_returns_ndarray(self):
        b = Blob([2, 3, 4, 5])
        dt = b.data_tensor
        assert isinstance(dt, np.ndarray)
        assert dt.shape == (2, 3, 4, 5)
        assert dt.dtype == np.float32

    def test_data_tensor_zero_copy_write(self):
        b = Blob([2, 3, 4, 5])
        b.data_tensor[0, 0, 0, 0] = 42.0
        assert abs(b.data_tensor[0, 0, 0, 0] - 42.0) < 0.01

    def test_data_property_returns_copy(self):
        b = Blob([2, 3])
        b.data_tensor[0, 0] = 1.0
        d = b.data
        assert abs(d[0, 0] - 1.0) < 0.01
        d[0, 0] = 99.0
        assert abs(b.data_tensor[0, 0] - 1.0) < 0.01

    def test_diff_tensor_returns_ndarray(self):
        b = Blob([2, 3])
        dt = b.diff_tensor
        assert isinstance(dt, np.ndarray)
        assert dt.shape == (2, 3)
        assert dt.dtype == np.float32

    def test_diff_tensor_zero_copy_write(self):
        b = Blob([3])
        b.diff_tensor[0] = 7.77
        assert abs(b.diff_tensor[0] - 7.77) < 0.01

    def test_diff_property_returns_copy(self):
        b = Blob([3])
        b.diff_tensor[0] = 2.0
        d = b.diff
        assert abs(d[0] - 2.0) < 0.01
        d[0] = 99.0
        assert abs(b.diff_tensor[0] - 2.0) < 0.01

    def test_update_subtracts_diff_from_data(self):
        b = Blob([3])
        b.data_tensor[:] = [10.0, 20.0, 30.0]
        b.diff_tensor[:] = [1.0, 2.0, 3.0]
        b.Update()
        np.testing.assert_allclose(b.data_tensor, [9.0, 18.0, 27.0], rtol=1e-5)

    def test_data_tensor_persists_across_calls(self):
        b = Blob([2, 2])
        t1 = b.data_tensor
        t1[0, 0] = 5.0
        t2 = b.data_tensor
        assert abs(t2[0, 0] - 5.0) < 0.01


@require_cpp_extension
class TestBlobMemoryStress:
    """Stress tests that loop many create/destroy/Reshape cycles to catch
    tiny per-operation leaks that would be invisible in a single lifecycle.

    A leak of even 4 bytes per operation becomes 4000 bytes after 1000
    iterations, which is easily detectable.
    """

    @staticmethod
    def _gc():
        import gc
        gc.collect(); gc.collect(); gc.collect()

    def test_create_destroy_loop_no_leak(self):
        """Create and destroy 500 blobs; net memory change must be 0."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()
        live0 = caffe_ffi.live_blob_count()

        for i in range(500):
            b = Blob([4, 5, 6])  # 120 elements * 8 = 960 bytes each
            b.fill(float(i))
            del b

        self._gc()
        mem1 = caffe_ffi.total_allocated_bytes()
        live1 = caffe_ffi.live_blob_count()
        assert live1 == live0, f"Blob leak: +{live1 - live0} after 500 create/destroy cycles"
        assert mem1 == mem0, f"Memory leak: +{mem1 - mem0} bytes after 500 create/destroy cycles"

    def test_reshape_loop_no_leak(self):
        """Repeatedly grow and shrink a single blob; net change must be 0."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()

        b = Blob([1])
        shapes = [(2, 3), (10, 10), (50, 50), (100, 100), (1, 1), (0,)]
        for _ in range(200):
            for shape in shapes:
                b.Reshape(list(shape))
                b.fill(1.0)

        expected_small = 1 * 4 * 2  # (1,) = 8 bytes
        b.Reshape([1])
        assert caffe_ffi.total_allocated_bytes() - mem0 == expected_small

        del b
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0, \
            f"Memory leak after reshape loop: +{caffe_ffi.total_allocated_bytes() - mem0} bytes"

    def test_copy_from_loop_no_leak(self):
        """Repeatedly copy_from between two blobs; net must stay constant."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()

        src = Blob([8, 8])
        src.from_numpy(np.random.randn(8, 8).astype(np.float32))
        dst = Blob()

        expected_total = 8 * 8 * 4 * 2  # only src allocates initially (dst=[0])
        assert caffe_ffi.total_allocated_bytes() - mem0 == expected_total

        for _ in range(300):
            dst.copy_from(src)  # dst reshapes to [8,8] on first call
            src.copy_from(dst)

        # After first copy, both are [8,8]; subsequent copies should not change allocation
        expected_both = expected_total * 2
        assert caffe_ffi.total_allocated_bytes() - mem0 == expected_both

        del src
        del dst
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0, \
            f"Memory leak after copy loop: +{caffe_ffi.total_allocated_bytes() - mem0} bytes"

    def test_from_numpy_to_numpy_loop_no_leak(self):
        """Repeated from_numpy/to_numpy roundtrips must not leak."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()

        b = Blob()
        for i in range(400):
            arr = np.full((4, 4), float(i), dtype=np.float32)
            b.from_numpy(arr)
            result = b.to_numpy()
            np.testing.assert_array_equal(result, arr)

        expected = 4 * 4 * 4 * 2  # 128 bytes
        assert caffe_ffi.total_allocated_bytes() - mem0 == expected

        del b
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0, \
            f"Memory leak after numpy roundtrip loop: +{caffe_ffi.total_allocated_bytes() - mem0} bytes"

    def test_serialization_roundtrip_loop_no_leak(self):
        """Repeated to_numpy → new Blob → from_numpy → del must net 0."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()

        original = Blob([3, 5])
        original.from_numpy(np.arange(15, dtype=np.float32).reshape(3, 5))

        for _ in range(200):
            data = original.to_numpy()
            tmp = Blob()
            tmp.from_numpy(data)
            np.testing.assert_array_equal(tmp.to_numpy(), data)
            del tmp

        self._gc()
        # Only original should remain
        expected = 3 * 5 * 4 * 2
        assert caffe_ffi.total_allocated_bytes() - mem0 == expected

        del original
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0, \
            f"Memory leak after serialization loop: +{caffe_ffi.total_allocated_bytes() - mem0} bytes"


@require_cpp_extension
class TestBlobExceptionSafety:
    """Tests that memory is correctly freed when operations fail mid-way.

    If an exception is thrown during Reshape/from_numpy (e.g., invalid shape),
    the blob must either remain in its previous valid state or be fully cleaned up;
    no memory should be leaked.
    """

    @staticmethod
    def _gc():
        import gc
        gc.collect(); gc.collect(); gc.collect()

    def test_reshape_invalid_shape_no_leak(self):
        """Reshape with invalid dimensions (e.g., negative) must not leak."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()
        live0 = caffe_ffi.live_blob_count()

        b = Blob([4, 4])
        initial_bytes = 4 * 4 * 4 * 2
        assert caffe_ffi.total_allocated_bytes() - mem0 == initial_bytes

        # Try various invalid shapes — negative dimensions are invalid
        invalid_shapes = [[-1], [2, -3], [-1, -2]]
        for shape in invalid_shapes:
            try:
                b.Reshape(shape)
            except (ValueError, RuntimeError, Exception):
                pass  # Expected; any exception type is fine as long as no leak

        # Blob should either be back to [4,4] or [0]; memory must be valid
        # Regardless, no extra bytes should be leaked
        self._gc()
        del b
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0, \
            f"Memory leak after invalid Reshape: +{caffe_ffi.total_allocated_bytes() - mem0} bytes"
        assert caffe_ffi.live_blob_count() == live0

    def test_empty_lifecycle_no_leak(self):
        """Blob created and destroyed without ever Reshape-ing to non-zero must net 0."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()
        live0 = caffe_ffi.live_blob_count()

        for _ in range(100):
            b = Blob()  # shape [0], 0 tensor bytes
            assert b.shape == (0,)
            del b

        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0
        assert caffe_ffi.live_blob_count() == live0

    def test_partial_update_gc_no_leak(self):
        """Blob with references dropped mid-operation must be fully collected."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()

        for _ in range(100):
            b = Blob([10, 10])
            b.fill(3.14)
            b.from_numpy(np.ones((5, 5), dtype=np.float32))
            # Don't del explicitly — let gc handle it when ref drops
            b = None  # type: ignore

        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0, \
            f"Memory leak after reassignment: +{caffe_ffi.total_allocated_bytes() - mem0} bytes"


@require_cpp_extension
class TestBlobInterleavedLifecycle:
    """Tests with multiple blobs whose lifetimes interleave in complex ways,
    verifying that counter bookkeeping handles out-of-order destruction correctly.
    """

    @staticmethod
    def _gc():
        import gc
        gc.collect(); gc.collect(); gc.collect()

    @staticmethod
    def _nbytes(shape):
        count = 1
        for d in shape:
            count *= d
        return count * 4 * 2

    def test_out_of_order_destruction(self):
        """Create blobs A,B,C,D and delete in order B,D,A,C; counters must stay correct."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()
        live0 = caffe_ffi.live_blob_count()

        a = Blob([2, 2])  # 32B
        b = Blob([3, 3])  # 72B
        c = Blob([4, 4])  # 128B
        d = Blob([5, 5])  # 200B

        total = self._nbytes([2,2]) + self._nbytes([3,3]) + self._nbytes([4,4]) + self._nbytes([5,5])
        assert caffe_ffi.total_allocated_bytes() - mem0 == total
        assert caffe_ffi.live_blob_count() == live0 + 4

        # Delete in non-creation order: B, D, A, C
        del b  # -72B
        self._gc()
        assert caffe_ffi.total_allocated_bytes() - mem0 == total - self._nbytes([3,3])
        assert caffe_ffi.live_blob_count() == live0 + 3

        del d  # -200B
        self._gc()
        assert caffe_ffi.total_allocated_bytes() - mem0 == total - self._nbytes([3,3]) - self._nbytes([5,5])
        assert caffe_ffi.live_blob_count() == live0 + 2

        del a  # -32B
        self._gc()
        assert caffe_ffi.total_allocated_bytes() - mem0 == self._nbytes([4,4])  # only C left
        assert caffe_ffi.live_blob_count() == live0 + 1

        del c  # -128B
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0
        assert caffe_ffi.live_blob_count() == live0

    def test_nested_blob_references(self):
        """Blobs referencing each other via copy_from, then deleting in nested order."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()

        outer = Blob([6, 6])
        outer.fill(1.0)

        def inner_scope():
            inner1 = Blob()
            inner1.copy_from(outer)
            inner2 = Blob()
            inner2.copy_from(outer)
            inner3 = Blob([3, 3])
            inner3.from_numpy(outer.to_numpy()[:3, :3])
            # inner1, inner2, inner3 go out of scope here

        inner_scope()
        self._gc()

        # Only outer should remain
        expected_outer = self._nbytes([6, 6])
        assert caffe_ffi.total_allocated_bytes() - mem0 == expected_outer, \
            f"Nested scope leaked: +{caffe_ffi.total_allocated_bytes() - mem0 - expected_outer} bytes"

        del outer
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0

    def test_blob_list_append_pop(self):
        """Simulate a dynamic blob pool: append N blobs, pop half, verify counts."""
        self._gc()
        mem0 = caffe_ffi.total_allocated_bytes()
        live0 = caffe_ffi.live_blob_count()

        pool = []
        shapes = [(2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)]

        for shape in shapes:
            b = Blob(list(shape))
            b.fill(0.0)
            pool.append(b)

        total = sum(self._nbytes(s) for s in shapes)
        assert caffe_ffi.total_allocated_bytes() - mem0 == total
        assert caffe_ffi.live_blob_count() == live0 + len(shapes)

        # Pop from middle and end, deleting those blobs
        popped = pool.pop(3)  # remove (5,5)
        del popped
        self._gc()
        remaining_shapes = shapes[:3] + shapes[4:]
        total_after_pop = sum(self._nbytes(s) for s in remaining_shapes)
        assert caffe_ffi.total_allocated_bytes() - mem0 == total_after_pop
        assert caffe_ffi.live_blob_count() == live0 + len(remaining_shapes)

        # Clear pool entirely
        pool.clear()
        self._gc()
        assert caffe_ffi.total_allocated_bytes() == mem0
        assert caffe_ffi.live_blob_count() == live0
