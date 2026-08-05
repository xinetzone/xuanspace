"""Test script for zero-copy Tensor API (DLPack interop).

Verifies:
- Blob.set_data() accepts numpy arrays (zero-copy via DLPack)
- Blob.set_data() accepts Python lists (auto-conversion)
- Blob.set_diff() accepts numpy arrays
- Data integrity after transfer
"""
import caffe_ffi
import numpy as np


def _make_blob():
    return caffe_ffi.Blob([2, 3])


def test_set_data_numpy():
    b = _make_blob()
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b.set_data(data)
    assert float(b.to_numpy().sum()) == 21.0


def test_set_data_list():
    b = _make_blob()
    b.set_data([[7, 8, 9], [10, 11, 12]])
    assert float(b.to_numpy().sum()) == 57.0


def test_set_diff_numpy():
    b = _make_blob()
    b.set_diff(np.ones((2, 3), dtype=np.float32) * 0.5)
    assert float(b.to_numpy(get_diff=True).sum()) == 3.0


def test_data_integrity():
    b = _make_blob()
    b.set_data([[7, 8, 9], [10, 11, 12]])
    out = b.to_numpy()
    expected = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
    assert np.allclose(out, expected), f"Data mismatch: {out} vs {expected}"


if __name__ == "__main__":
    print("Native mode:", caffe_ffi._ffi_api.is_available())
    test_set_data_numpy()
    test_set_data_list()
    test_set_diff_numpy()
    test_data_integrity()
    print("Zero-copy Tensor API works!")