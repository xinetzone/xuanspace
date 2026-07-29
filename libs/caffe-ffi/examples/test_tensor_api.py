"""Test script for zero-copy Tensor API (DLPack interop).

Verifies:
- Blob.set_data() accepts numpy arrays (zero-copy via DLPack)
- Blob.set_data() accepts Python lists (auto-conversion)
- Blob.set_diff() accepts numpy arrays
- Data integrity after transfer
"""
import sys
from pathlib import Path

# Add python directory to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / 'python'))

import caffe_ffi
import numpy as np

print('Native mode:', caffe_ffi._ffi_api.is_available())

b = caffe_ffi.Blob([2, 3])
data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
b.set_data(data)
print('set_data(numpy) OK, sum:', float(b.to_numpy().sum()))

b.set_data([[7, 8, 9], [10, 11, 12]])
print('set_data(list) OK, sum:', float(b.to_numpy().sum()))

b.set_diff(np.ones((2, 3), dtype=np.float32) * 0.5)
print('set_diff(numpy) OK, diff sum:', float(b.to_numpy(get_diff=True).sum()))

# Verify data matches
out = b.to_numpy()
expected = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
assert np.allclose(out, expected), f"Data mismatch: {out} vs {expected}"
print('Data verification passed!')

print('Zero-copy Tensor API works!')
