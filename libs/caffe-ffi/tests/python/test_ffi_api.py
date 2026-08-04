"""Unit tests for caffe_ffi._ffi_api (FFI native-lib loading / diagnostics / registry).

These tests cover the pure-Python logic in `_ffi_api.py`:

  * Platform-specific library name candidates (`_get_lib_names_for_platform`)
  * Native library search directory list (`_get_search_dirs`)
  * Library path discovery (`_find_lib_path` / `lib_path()`)
  * Structured init diagnostics (`_FFIInitDiagnostics` / `get_init_diagnostics`)
  * Strict-mode raising behaviour
  * FFI function registry caching (`_FFIRegistry` / `get_global_func`)
  * Availability / lib-path consistency (`is_available` / `lib_path`)

All tests are pure Python and do not require the native C++ extension to be
present, so they run in both Python-only fallback and full FFI modes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

try:
    import caffe_ffi
    from caffe_ffi import _ffi_api
except ImportError:
    pytest.skip("caffe_ffi not installed", allow_module_level=True)


# ─── Platform library-name candidates ────────────────────────────────

class TestGetLibNamesForPlatform:
    def test_returns_list_of_strings(self):
        names = _ffi_api._get_lib_names_for_platform()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_contains_caffe_ffi_identifier(self):
        names = _ffi_api._get_lib_names_for_platform()
        assert any("_caffe_ffi" in n or "caffe_ffi" in n for n in names)

    def test_win32_has_dll_and_pyd(self):
        if sys.platform != "win32":
            pytest.skip("Windows-only assertion")
        names = _ffi_api._get_lib_names_for_platform()
        assert any(n.endswith(".dll") for n in names)
        assert any(n.endswith(".pyd") for n in names)

    def test_posix_has_so(self):
        if sys.platform == "win32":
            pytest.skip("POSIX-only assertion")
        names = _ffi_api._get_lib_names_for_platform()
        assert any(n.endswith(".so") for n in names)


# ─── Search directories ──────────────────────────────────────────────

class TestGetSearchDirs:
    def test_returns_list_of_paths(self):
        dirs = _ffi_api._get_search_dirs()
        assert isinstance(dirs, list)
        assert len(dirs) > 0
        assert all(isinstance(d, Path) for d in dirs)

    def test_package_dir_is_first(self):
        dirs = _ffi_api._get_search_dirs()
        # The package's own directory is searched first.
        assert dirs[0] == Path(_ffi_api.__file__).resolve().parent

    def test_search_dirs_are_absolute(self):
        dirs = _ffi_api._get_search_dirs()
        assert all(d.is_absolute() for d in dirs)


# ─── Library path discovery ──────────────────────────────────────────

class TestFindLibPath:
    def test_find_lib_path_returns_none_or_path(self):
        p = _ffi_api._find_lib_path()
        assert p is None or isinstance(p, Path)

    def test_lib_path_consistency(self):
        """lib_path() must equal the module-level _find_lib_path() result."""
        assert _ffi_api.lib_path() == _ffi_api._lib_path


# ─── Init diagnostics (pure Python) ──────────────────────────────────

class TestFFIInitDiagnostics:
    def test_initial_state(self):
        diag = _ffi_api._FFIInitDiagnostics()
        assert diag.success is False
        assert diag.errors == []
        assert diag.warnings == []
        assert diag.search_dirs_checked == []
        assert diag.lib_names_searched == []
        assert diag.tvm_ffi_importable is None
        assert diag.lib_path_found is None
        assert diag.load_module_error is None

    def test_record_search(self):
        diag = _ffi_api._FFIInitDiagnostics()
        dirs = [Path("/a"), Path("/b")]
        names = ["lib.so", "lib.dylib"]
        diag.record_search(dirs, names)
        assert diag.search_dirs_checked == [str(d) for d in dirs]
        assert diag.lib_names_searched == ["lib.so", "lib.dylib"]

    def test_record_lib_not_found_adds_error(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag.record_search([Path(__file__)], ["lib.so"])
        diag.record_lib_not_found()
        assert len(diag.errors) == 1
        err = diag.errors[0]
        assert err["stage"] == "find_lib"
        assert err["code"] == "LIB_NOT_FOUND"
        assert "Searched" in err["message"]

    def test_record_tvm_ffi_import_error(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag.record_tvm_ffi_import_error(ImportError("no module"))
        assert diag.tvm_ffi_importable is False
        assert diag.errors[0]["code"] == "TVM_FFI_IMPORT_ERROR"
        assert diag.errors[0]["exception_type"] == "ImportError"

    def test_record_load_module_error(self):
        diag = _ffi_api._FFIInitDiagnostics()
        lib = Path("/x/lib.so")
        diag.lib_path_found = lib
        diag.record_load_module_error(RuntimeError("boom"))
        assert diag.load_module_error == "boom"
        assert diag.errors[0]["code"] == "LOAD_MODULE_FAILED"
        assert diag.errors[0]["lib_path"] == str(lib)

    def test_record_unexpected_error(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag.record_unexpected_error(ValueError("bad"))
        assert diag.errors[0]["code"] == "UNEXPECTED_ERROR"
        assert diag.errors[0]["exception_type"] == "ValueError"
        assert "traceback" in diag.errors[0]

    def test_record_success(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag.record_success(Path("/lib/_caffe_ffi.so"))
        assert diag.success is True
        assert diag.lib_path_found == Path("/lib/_caffe_ffi.so")
        assert diag.tvm_ffi_importable is True

    def test_record_warning(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag.record_warning("dll_path", "ADD_DLL_DIR_FAILED", "msg", extra=1)
        assert len(diag.warnings) == 1
        assert diag.warnings[0]["stage"] == "dll_path"
        assert diag.warnings[0]["code"] == "ADD_DLL_DIR_FAILED"
        assert diag.warnings[0]["extra"] == 1

    def test_summary_success(self):
        diag = _ffi_api._FFIInitDiagnostics()
        lib = Path("/lib.so")
        diag.record_success(lib)
        s = diag.summary()
        assert "initialized successfully" in s
        assert str(lib) in s

    def test_summary_failure(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag.record_search([Path("/nonexistent")], ["lib.so"])
        diag.record_lib_not_found()
        s = diag.summary()
        assert "FAILED" in s
        assert "LIB_NOT_FOUND" in s

    def test_strict_mode_raises_on_error(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag._strict_init = True
        diag.record_search([Path("/nonexistent")], ["lib.so"])
        with pytest.raises(RuntimeError, match="CAFFE_FFI_STRICT_INIT"):
            diag.record_lib_not_found()

    def test_strict_mode_off_does_not_raise(self):
        diag = _ffi_api._FFIInitDiagnostics()
        diag._strict_init = False
        diag.record_search([Path("/nonexistent")], ["lib.so"])
        diag.record_lib_not_found()  # must not raise
        assert diag.success is False


class TestGetInitDiagnostics:
    def test_returns_diagnostics_object(self):
        diag = _ffi_api.get_init_diagnostics()
        assert isinstance(diag, _ffi_api._FFIInitDiagnostics)
        assert hasattr(diag, "success")
        assert hasattr(diag, "errors")
        assert hasattr(diag, "warnings")
        assert hasattr(diag, "summary")

    def test_diagnostics_persistent_singleton(self):
        """get_init_diagnostics() always returns the same module-level object."""
        assert _ffi_api.get_init_diagnostics() is _ffi_api._diagnostics


# ─── FFI registry ────────────────────────────────────────────────────

class TestFFIRegistry:
    def test_available_matches_module(self):
        reg = _ffi_api._FFIRegistry()
        assert reg.available == _ffi_api.is_available()

    def test_get_global_func_caches_result(self):
        reg = _ffi_api._FFIRegistry()
        first = reg.get_global_func("_nonexistent_func")
        second = reg.get_global_func("_nonexistent_func")
        assert first is second

    def test_get_global_func_returns_none_when_unavailable(self):
        reg = _ffi_api._FFIRegistry()
        reg._get_global_func = None  # simulate tvm_ffi unavailable
        assert reg.get_global_func("_x") is None

    def test_register_object_records_type(self):
        reg = _ffi_api._FFIRegistry()
        reg._register_object = None  # avoid touching real tvm_ffi
        class _Dummy:
            pass
        reg.register_object("test.dummy", _Dummy)
        assert reg._types["test.dummy"] is _Dummy

    def test_object_raises_when_unavailable(self):
        reg = _ffi_api._FFIRegistry()
        reg._Object = None
        with pytest.raises(RuntimeError, match="tvm_ffi not available"):
            _ = reg.Object


class TestTopLevelAPI:
    def test_is_available_returns_bool(self):
        assert isinstance(_ffi_api.is_available(), bool)

    def test_lib_path_documented_contract(self):
        """lib_path() is None when unavailable, a Path when available."""
        p = _ffi_api.lib_path()
        if _ffi_api.is_available():
            assert isinstance(p, Path) and p.exists()
        else:
            assert p is None

    def test_get_global_func_wraps_registry(self):
        assert _ffi_api.get_global_func("_also_missing") == \
            _ffi_api.registry.get_global_func("_also_missing")