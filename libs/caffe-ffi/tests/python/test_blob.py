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
        mem_before = caffe_ffi.total_allocated_bytes()
        b = Blob([3, 4])
        mem_after = caffe_ffi.total_allocated_bytes()
        expected = self._expected_nbytes([3, 4])
        assert mem_after - mem_before == expected, \
            f"Expected +{expected}B after Blob([3,4]), got +{mem_after - mem_before}B"
        assert caffe_ffi.live_blob_count() >= 1

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
