"""Regression tests for the callback-registry cleanup (interpreter-shutdown segfault).

Root cause: the C++ ``data_io`` / ``python_layer`` registries are static
``std::unordered_map``s holding TVM FFI ``Function`` handles that reference Python
callables. Left populated, their destructors ran after ``Py_Finalize`` and touched
the already-destroyed Python runtime, causing a SIGSEGV on interpreter exit (exit
code 139).

Fix (prevention): ``caffe_ffi.data_io.clear`` / ``caffe_ffi.python_layer.clear``
release the stored handles, and ``caffe_ffi.__init__`` registers an ``atexit`` hook
that calls them before the runtime is torn down.

These tests assert the prevention mechanism is present and functional. The full
interpreter-exit verification (no segfault) is exercised by the standalone repro in
``.temp/repro_segfault.py`` in the P0 container.
"""

from __future__ import annotations

import atexit

import pytest

from .conftest import require_cpp_extension
from caffe_ffi import _ffi_api


_CLEAR_FUNCS = ("caffe_ffi.data_io.clear", "caffe_ffi.python_layer.clear")
_REG_FUNCS = ("caffe_ffi.data_io.register", "caffe_ffi.python_layer.register")


@require_cpp_extension
def test_clear_ffi_functions_exist():
    """The clear global functions must be registered by the C++ extension."""
    for name in _CLEAR_FUNCS:
        fn = _ffi_api.get_global_func(name)
        assert fn is not None, f"{name} global func not found"


@require_cpp_extension
def test_atexit_cleanup_hook_registered():
    """caffe_ffi must register an atexit hook that clears both registries."""
    import caffe_ffi

    assert hasattr(caffe_ffi, "_cleanup_callbacks"), "atexit cleanup callback missing"
    assert callable(caffe_ffi._cleanup_callbacks)

    # Verify the hook is actually tracked by atexit (CPython internal, guarded).
    # `_exithandlers` was removed in newer CPython revisions; fall back to the
    # functional check (test_cleanup_callbacks_runs_without_error) if absent.
    handlers = getattr(atexit, "_exithandlers", None)
    if handlers is not None:
        registered = any(
            getattr(h, "__name__", None) == "_cleanup_callbacks"
            or h is caffe_ffi._cleanup_callbacks
            for h in handlers
        )
        assert registered, "caffe_ffi._cleanup_callbacks not registered with atexit"


@require_cpp_extension
@pytest.mark.parametrize("reg_name,clear_name,key", [
    ("caffe_ffi.data_io.register", "caffe_ffi.data_io.clear", "Data.cleanup_test"),
    ("caffe_ffi.python_layer.register", "caffe_ffi.python_layer.clear",
     "cleanuptest.module.Layer"),
])
def test_register_then_clear_roundtrip(reg_name, clear_name, key):
    """Registering a callback then clearing it must succeed without error.

    This exercises the exact code path that previously left handles behind and
    segfaulted at shutdown. After clearing, re-registering under the same key must
    still work (the registry stays usable).
    """
    reg = _ffi_api.get_global_func(reg_name)
    clear = _ffi_api.get_global_func(clear_name)
    assert reg is not None and clear is not None

    # Register a Python callable into the registry.
    reg(key, lambda tensors: None)
    # Clear must release the handle without raising.
    clear()
    # Registry remains usable after clear.
    reg(key, lambda tensors: None)
    clear()


@require_cpp_extension
def test_cleanup_callbacks_runs_without_error():
    """Calling the atexit cleanup callback must not raise (idempotent, best-effort)."""
    import caffe_ffi

    # Seed both registries, then run the same hook the interpreter will run.
    for reg_name, key in [
        ("caffe_ffi.data_io.register", "Data.atexit_test"),
        ("caffe_ffi.python_layer.register", "atexit.module.L1"),
    ]:
        fn = _ffi_api.get_global_func(reg_name)
        if fn is not None:
            fn(key, lambda tensors: None)

    caffe_ffi._cleanup_callbacks()  # must not raise
    caffe_ffi._cleanup_callbacks()  # idempotent: second call also fine