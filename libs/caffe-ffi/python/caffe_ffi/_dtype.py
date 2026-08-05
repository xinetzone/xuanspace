"""Internal dtype validation utilities for caffe-ffi FFI boundary.

All Blob/Net data passes through float32 real-valued tensors. This module
centralizes the guard that rejects incompatible dtypes (complex, etc.)
before conversion, providing a single source of truth used across all
Python binding entry points.

These functions are NOT public API — they are internal helpers for the
binding layer. External code should use ``blob.data = arr`` or
``net.forward({...})`` which already enforce dtype validation.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_float32(arr: Any, field: str = "data") -> np.ndarray:
    """Convert array-like to ``np.float32`` ndarray, rejecting complex dtypes.

    This is the single entry point for dtype validation at the FFI boundary.
    It replaces the pattern ``np.asarray(arr, dtype=np.float32)`` which
    silently casts complex inputs to real (dropping the imaginary part)
    while emitting a ``ComplexWarning`` that is easy to miss.

    Parameters
    ----------
    arr : array-like
        Input value (numpy array, list, tuple, scalar, etc.).
    field : str
        Human-readable field name for the error message (e.g. ``"data"``,
        ``"diff"``, ``"weights"``). Included in ``TypeError`` message so
        users can identify which argument caused the error.

    Returns
    -------
    np.ndarray
        C-contiguous ``float32`` ndarray.

    Raises
    ------
    TypeError
        If *arr* has a complex dtype (``complex64``/``complex128``).
        Complex numbers cannot be represented in caffe-ffi's float32
        tensors; users should cast explicitly with ``.real`` (or
        ``np.real()``) if discarding the imaginary part is intentional.
    """
    if np.iscomplexobj(arr):
        raise TypeError(
            f"Complex dtypes are not supported for blob {field} "
            f"(got dtype={np.asarray(arr).dtype}); cast to real first with `.real`."
        )
    return np.asarray(arr, dtype=np.float32)
