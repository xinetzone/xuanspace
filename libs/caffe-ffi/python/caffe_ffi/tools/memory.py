"""Memory lifecycle tracking tools for caffe_ffi Blob objects.

Since caffe_ffi.Blob is a C++ FFI extension type (exported via TVM FFI), it does
not support Python weakref (missing Py_TPFLAGS_MANAGED_WEAKREF/tp_weaklistoffset).
This module provides three alternatives for tracking Blob lifecycle:

  1. BlobRef      - Pure-Python wrapper class supporting weakref and destroy callbacks
  2. tracked_blob - Context manager that auto-verifies memory release on block exit
  3. blob_snapshot / mem_check - Detection tools based on total_allocated_bytes() counter

Usage:
    from caffe_ffi.tools import BlobRef, tracked_blob, blob_snapshot, mem_check

    # Context manager (auto-reports memory status on exit)
    with tracked_blob([2,3,4,5], "my_blob") as b:
        b.data_tensor[:] = 1.0
    # NOTE message appears because 'as b' keeps a reference

    # Weakref wrapper (callback on destroy)
    br = BlobRef([2,3,4,5], label="test")
    ref = weakref.ref(br, lambda r: print("destroyed!"))
    del br  # triggers callback
"""

from __future__ import annotations

import sys
import weakref
from contextlib import contextmanager
from typing import Any, Callable, Optional

from .. import Blob
from .. import total_allocated_bytes as _total_allocated_bytes


class BlobRef:
    """Weakref-compatible Blob wrapper.

    Holds an underlying caffe_ffi.Blob instance, provides transparent attribute
    access (__getattr__ proxies to the underlying Blob), and supports
    weakref.ref(callback) to be notified when the wrapper is destroyed.

    Note: When the wrapper is destroyed, the underlying Blob's Python reference
    count drops to zero, immediately triggering C++ destruction. Callbacks should
    NOT access the underlying Blob (it may already be destructed); use them for
    notification only.

    Usage:
        from caffe_ffi.tools import BlobRef
        import weakref

        br = BlobRef([2, 3, 4, 5], label="test")
        ref = weakref.ref(br, lambda r: print("Blob destroyed"))
        del br  # triggers callback
    """

    __slots__ = ("_blob", "_data_ptr", "_diff_ptr", "_shape", "_label", "__dict__", "__weakref__")

    def __init__(self, shape, label: str = "") -> None:
        self._blob = Blob(shape)
        self._shape = tuple(shape) if not isinstance(shape, (list, tuple)) else tuple(shape)
        try:
            self._data_ptr = self._blob.data_tensor.ctypes.data
            self._diff_ptr = self._blob.diff_tensor.ctypes.data
        except Exception:
            self._data_ptr = 0
            self._diff_ptr = 0
        self._label = label

    def __getattr__(self, name: str) -> Any:
        return getattr(self._blob, name)

    def __repr__(self) -> str:
        nbytes = 0
        try:
            nbytes = self._blob.data_tensor.nbytes + self._blob.diff_tensor.nbytes
        except Exception:
            pass
        return (
            f"BlobRef(shape={self._shape}, data_ptr=0x{self._data_ptr:016x}, "
            f"diff_ptr=0x{self._diff_ptr:016x}, nbytes={nbytes}, "
            f"label={self._label!r})"
        )

    @property
    def data_ptr(self) -> int:
        return self._data_ptr

    @property
    def diff_ptr(self) -> int:
        return self._diff_ptr

    @property
    def nbytes(self) -> int:
        try:
            return self._blob.data_tensor.nbytes + self._blob.diff_tensor.nbytes
        except Exception:
            return 0

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def label(self) -> str:
        return self._label


@contextmanager
def tracked_blob(shape, label: str = "blob", verbose: bool = True):
    """Context manager: create a Blob and report memory status on block exit.

    Even if an exception is raised inside the with-block, the finally clause
    detects memory status and prints a log.

    Note: Python's ``with ... as b`` creates variable ``b`` in the caller's
    scope, which holds a Blob reference. When the with-block exits, ``b``
    is still alive (refcount > 0), so C++ destruction is not triggered immediately.
    tracked_blob reports this state faithfully rather than false-positive LEAK.
    To fully verify release, call ``del b`` (or let the function return so ``b``
    goes out of scope) and then call mem_check().

    Args:
        shape: Blob shape (list/tuple of ints).
        label: Label string used in log output to identify this Blob.
        verbose: True=print logs to stdout.

    Yields:
        caffe_ffi.Blob instance.

    Usage (function scope; b auto-freed on return):
        def test():
            with tracked_blob([10, 10], "test1") as b:
                b.data_tensor[:] = 1.0
            # On with-exit: blob still referenced by b, prints NOTE
        # After function return, b goes out of scope, C++ destructor fires
        mem_check("after_test")
    """
    mem_before = _total_allocated_bytes()
    b = Blob(shape)
    expected = b.data_tensor.nbytes + b.diff_tensor.nbytes
    dp = b.data_tensor.ctypes.data
    dfp = b.diff_tensor.ctypes.data
    exception_info: Optional[str] = None
    if verbose:
        print(
            f"[TRACK:{label}] created, data_ptr=0x{dp:016x}, "
            f"diff_ptr=0x{dfp:016x}, nbytes={expected}"
        )
    try:
        yield b
    except Exception:
        exc_type, exc_val, _ = sys.exc_info()
        exception_info = f"{exc_type.__name__}: {exc_val}"
        raise
    finally:
        del b
        mem_after = _total_allocated_bytes()
        if verbose:
            held_by_as = mem_after >= mem_before + expected
            exc_note = f" (exception: {exception_info})" if exception_info else ""
            if mem_after == mem_before:
                print(f"[TRACK:{label}] OK: freed {expected} bytes{exc_note}")
            elif held_by_as:
                remaining = mem_after - mem_before
                print(
                    f"[TRACK:{label}] NOTE: {remaining} bytes still held "
                    f"(expected with 'as b' binding; del b or exit scope to free)"
                    f"{exc_note}"
                )
            else:
                delta = mem_after - mem_before
                if delta < 0:
                    print(
                        f"[TRACK:{label}] WARNING: memory decreased by {-delta} bytes "
                        f"beyond this blob (other blobs freed concurrently){exc_note}"
                    )
                else:
                    print(
                        f"[TRACK:{label}] LEAK? +{delta} bytes "
                        f"(unexpected allocations during block){exc_note}"
                    )


class MemoryTrace:
    """Context manager that records memory delta on enter/exit.

    Usage:
        with MemoryTrace("my_operation") as mt:
            b = Blob([100, 100])
            b.data_tensor[:] = 1.0
        print(f"Delta: {mt.delta} bytes")
    """

    def __init__(self, label: str = "", verbose: bool = True) -> None:
        self.label = label
        self.verbose = verbose
        self.mem_before: int = 0
        self.mem_after: int = 0
        self.delta: int = 0

    def __enter__(self) -> "MemoryTrace":
        self.mem_before = _total_allocated_bytes()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.mem_after = _total_allocated_bytes()
        self.delta = self.mem_after - self.mem_before
        if self.verbose:
            prefix = f"[{self.label}] " if self.label else ""
            status = "OK" if self.delta <= 0 else f"+{self.delta} bytes"
            print(
                f"[MEM-TRACE] {prefix}before={self.mem_before}, "
                f"after={self.mem_after}, delta={status}"
            )


def blob_snapshot(label: str = "", verbose: bool = True) -> int:
    """Print and return the current global allocated byte count.

    Args:
        label: Label string.
        verbose: True=print to stdout.

    Returns:
        Current total_allocated_bytes value.
    """
    nbytes = _total_allocated_bytes()
    if verbose:
        prefix = f"[{label}] " if label else ""
        print(
            f"[MEM-SNAPSHOT] {prefix}total_allocated_bytes={nbytes} "
            f"({nbytes/1024:.2f} KB)"
        )
    return nbytes


def mem_check(label: str = "check", verbose: bool = True) -> bool:
    """Check whether current memory allocation is zero (leak detection).

    Args:
        label: Checkpoint label.
        verbose: True=print result.

    Returns:
        True=no leak (zero), False=leak detected.
    """
    nbytes = _total_allocated_bytes()
    ok = nbytes == 0
    if verbose:
        status = "OK (zero)" if ok else f"LEAK DETECTED ({nbytes} bytes still allocated)"
        print(f"[MEM-CHECK] {label}: {status}")
    return ok
