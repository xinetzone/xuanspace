from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_init_logger = logging.getLogger("caffe_ffi.ffi_init")

# Disable C++ backtrace capture by default to avoid crashes in Python test
# frameworks (pytest/unittest) where backtrace_symbols() may crash on Python
# stack frames. Users can opt in by setting CAFFE_FFI_DISABLE_BACKTRACE=0
# before importing caffe_ffi.
if os.environ.get("CAFFE_FFI_DISABLE_BACKTRACE") is None:
    os.environ["CAFFE_FFI_DISABLE_BACKTRACE"] = "1"


def _get_lib_names_for_platform() -> list[str]:
    """Return the platform-specific library filename candidates."""
    if sys.platform == "win32":
        return ["_caffe_ffi.dll", "_caffe_ffi.pyd", "_caffe_ffi.cp314-win_amd64.pyd"]
    elif sys.platform == "darwin":
        return ["lib_caffe_ffi.dylib", "_caffe_ffi.so", "_caffe_ffi.cpython-314-darwin.so"]
    else:
        return ["lib_caffe_ffi.so", "_caffe_ffi.so", "_caffe_ffi.cpython-314-x86_64-linux-gnu.so"]


def _get_search_dirs() -> list[Path]:
    """Return the list of directories to search for the native library."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    return [
        Path(__file__).parent,
        base_dir / "build-ninja" / "Release",
        base_dir / "build-ninja" / "lib",
        base_dir / "build-ninja",
        base_dir / "build-cmake" / "Release",
        base_dir / "build-cmake" / "lib",
        base_dir / "build-cmake",
        base_dir / "build-wheel" / "Release",
        base_dir / "build-wheel" / "lib",
        base_dir / "build-wheel",
        base_dir / "build" / "Release",
        base_dir / "build" / "lib",
        base_dir / "build" / "python" / "caffe_ffi",
        base_dir / "build",
        base_dir,
    ]


def _find_lib_path() -> Optional[Path]:
    search_dirs = _get_search_dirs()
    lib_names = _get_lib_names_for_platform()

    for search_dir in search_dirs:
        if search_dir.exists():
            for lib_name in lib_names:
                lib_path = search_dir / lib_name
                if lib_path.exists():
                    return lib_path

    return None


_lib_path = _find_lib_path()
_ffi_available = False


class _FFIInitDiagnostics:
    """Captures detailed diagnostics during FFI initialization.

    This class addresses the "silent fallback" anti-pattern: when the native
    library fails to load, callers need to know *why* (not just that it failed)
    to diagnose build/installation issues. All failure paths record structured
    information here instead of merely logging a warning.

    Usage:
        from caffe_ffi._ffi_api import get_init_diagnostics
        diag = get_init_diagnostics()
        if not diag.success:
            print(diag.summary())
            for err in diag.errors:
                print(f"  - {err['stage']}: {err['message']}")
    """

    def __init__(self):
        self.success: bool = False
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.search_dirs_checked: list[str] = []
        self.lib_names_searched: list[str] = []
        self.tvm_ffi_importable: Optional[bool] = None
        self.lib_path_found: Optional[Path] = None
        self.load_module_error: Optional[str] = None
        self._strict_init: bool = os.environ.get("CAFFE_FFI_STRICT_INIT", "0") == "1"

    def record_search(self, search_dirs: list[Path], lib_names: list[str]):
        """Record which directories and library names were searched."""
        self.search_dirs_checked = [str(d) for d in search_dirs]
        self.lib_names_searched = list(lib_names)

    def record_lib_not_found(self):
        """Record that no native library was found in any search directory."""
        existing_dirs = [d for d in self.search_dirs_checked if Path(d).exists()]
        missing_dirs = [d for d in self.search_dirs_checked if not Path(d).exists()]
        self.errors.append({
            "stage": "find_lib",
            "code": "LIB_NOT_FOUND",
            "message": (
                f"No native library found. Searched {len(existing_dirs)} existing "
                f"directories for {self.lib_names_searched}. "
                f"{len(missing_dirs)} directories did not exist."
            ),
            "existing_dirs": existing_dirs,
            "missing_dirs": missing_dirs,
        })
        self._maybe_raise_strict("Native library not found in any search directory")

    def record_tvm_ffi_import_error(self, error: ImportError):
        """Record that tvm_ffi could not be imported."""
        self.tvm_ffi_importable = False
        self.errors.append({
            "stage": "import_tvm_ffi",
            "code": "TVM_FFI_IMPORT_ERROR",
            "message": f"Failed to import tvm_ffi: {error}",
            "exception_type": type(error).__name__,
            "exception_msg": str(error),
        })
        self._maybe_raise_strict(f"tvm_ffi import failed: {error}")

    def record_load_module_error(self, error: Exception):
        """Record that tvm_ffi.load_module() failed."""
        self.load_module_error = str(error)
        self.errors.append({
            "stage": "load_module",
            "code": "LOAD_MODULE_FAILED",
            "message": f"Failed to load native module via tvm_ffi: {error}",
            "exception_type": type(error).__name__,
            "exception_msg": str(error),
            "lib_path": str(self.lib_path_found) if self.lib_path_found else None,
        })
        self._maybe_raise_strict(f"Native module load failed: {error}")

    def record_unexpected_error(self, error: Exception):
        """Record an unexpected error during initialization."""
        import traceback
        self.errors.append({
            "stage": "init",
            "code": "UNEXPECTED_ERROR",
            "message": f"Unexpected error during FFI init: {error}",
            "exception_type": type(error).__name__,
            "exception_msg": str(error),
            "traceback": traceback.format_exc(),
        })
        self._maybe_raise_strict(f"Unexpected FFI init error: {error}")

    def record_success(self, loaded_from: Path):
        """Record successful initialization."""
        self.success = True
        self.lib_path_found = loaded_from
        self.tvm_ffi_importable = True

    def record_warning(self, stage: str, code: str, message: str, **kwargs):
        """Record a non-fatal warning."""
        entry = {"stage": stage, "code": code, "message": message, **kwargs}
        self.warnings.append(entry)

    def summary(self) -> str:
        """Return a human-readable summary of the init diagnostics."""
        if self.success:
            return f"FFI initialized successfully from {self.lib_path_found}"
        lines = [
            "=== caffe-ffi FFI Initialization Diagnostics ===",
            f"Status: FAILED ({len(self.errors)} error(s), {len(self.warnings)} warning(s))",
            f"Strict mode: {'ON' if self._strict_init else 'OFF'}",
            "",
        ]
        if self.lib_names_searched:
            lines.append(f"Library names searched: {self.lib_names_searched}")
        if self.search_dirs_checked:
            existing = [d for d in self.search_dirs_checked if Path(d).exists()]
            missing = [d for d in self.search_dirs_checked if not Path(d).exists()]
            lines.append(f"Directories checked ({len(existing)} exist, {len(missing)} missing):")
            for d in existing:
                lines.append(f"  [exists] {d}")
            for d in missing:
                lines.append(f"  [missing] {d}")
        for i, err in enumerate(self.errors, 1):
            lines.append(f"\nError #{i} [{err['stage']}/{err['code']}]:")
            lines.append(f"  {err['message']}")
        if self.warnings:
            lines.append(f"\nWarnings:")
            for w in self.warnings:
                lines.append(f"  [{w['stage']}/{w['code']}] {w['message']}")
        lines.append("")
        lines.append("Hint: Build the C++ extension with 'pip install -e .' or "
                      "set CAFFE_FFI_STRICT_INIT=0 to suppress strict mode.")
        return "\n".join(lines)

    def _maybe_raise_strict(self, message: str):
        """In strict mode (CI/testing), raise instead of silently falling back."""
        if self._strict_init:
            raise RuntimeError(
                f"[CAFFE_FFI_STRICT_INIT] {message}\n"
                f"Diagnostics:\n{self.summary()}"
            )


_diagnostics = _FFIInitDiagnostics()


def get_init_diagnostics() -> _FFIInitDiagnostics:
    """Return diagnostics from FFI initialization.

    Returns an object with:
        - success (bool): Whether native FFI loaded successfully
        - errors (list[dict]): Structured error records with stage/code/message
        - warnings (list[dict]): Non-fatal warnings
        - search_dirs_checked (list[str]): Directories that were searched for .so
        - lib_names_searched (list[str]): Library filenames that were searched
        - summary() -> str: Human-readable diagnostic summary

    Useful for debugging installation/build issues and for CI checks.
    """
    return _diagnostics


def _try_init_tvm_ffi():
    global _ffi_available

    # Preset to False at entry — single source of truth for fallback state
    _ffi_available = False

    try:
        if sys.platform == "win32":
            _setup_windows_dll_paths()

        # Record search diagnostics for all outcomes (found+loaded, found+load_failed, not_found)
        _diagnostics.record_search(_get_search_dirs(), _get_lib_names_for_platform())

        import tvm_ffi
        _diagnostics.tvm_ffi_importable = True

        if _lib_path is not None:
            _diagnostics.lib_path_found = _lib_path
            lib_dir = _lib_path.parent
            _init_logger.info("Loading native library from: %s", _lib_path)
            if sys.platform == "win32":
                try:
                    os.add_dll_directory(str(lib_dir))
                except (OSError, AttributeError) as e:
                    _diagnostics.record_warning(
                        "dll_path", "ADD_DLL_DIR_FAILED",
                        f"os.add_dll_directory failed for {lib_dir}: {e}",
                    )
            try:
                tvm_ffi.load_module(str(_lib_path))
            except Exception as load_err:
                _diagnostics.record_load_module_error(load_err)
                _init_logger.warning(
                    "Failed to load caffe-ffi native library: %s. "
                    "Falling back to Python-only mode.", load_err,
                )
                return False
            _init_logger.debug("Loaded caffe-ffi native library from %s", _lib_path)
            _ffi_available = True
            _diagnostics.record_success(_lib_path)
            return True
        else:
            _diagnostics.record_lib_not_found()
            _init_logger.warning(
                "caffe-ffi native library not found. "
                "Searched build/Release, build/, build-cmake/, build-wheel/Release, "
                "and package directories. "
                "Falling back to Python-only mode. Build the C++ extension for full "
                "functionality."
            )
            return False
    except ImportError as e:
        _diagnostics.record_tvm_ffi_import_error(e)
        _init_logger.warning(
            "Failed to import tvm_ffi: %s. Falling back to Python-only mode.", e,
        )
        return False
    except Exception as e:
        _diagnostics.record_unexpected_error(e)
        _init_logger.warning(
            "Failed to load caffe-ffi native library: %s. Falling back to Python-only mode.", e,
        )
        return False


def _setup_windows_dll_paths():
    """Setup Windows DLL search paths to use current conda env DLLs, not base env."""
    prefix = Path(sys.prefix)
    
    dll_dirs = [
        prefix / "Library" / "bin",
        prefix / "DLLs",
        prefix / "bin",
        prefix / "Lib" / "site-packages" / "tvm_ffi" / "lib",
    ]
    
    path_entries = []
    for d in dll_dirs:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except (OSError, AttributeError):
                pass
            path_entries.append(str(d))
    
    if path_entries:
        new_path = os.pathsep.join(path_entries) + os.pathsep + os.environ.get("PATH", "")
        os.environ["PATH"] = new_path


_try_init_tvm_ffi()


class _FFIRegistry:
    
    def __init__(self):
        self._funcs: Dict[str, Any] = {}
        self._types: Dict[str, type] = {}
        self._Object = None
        self._register_object = None
        self._get_global_func = None
        
        if _ffi_available:
            try:
                import tvm_ffi
                self._Object = tvm_ffi.Object
                self._register_object = tvm_ffi.register_object
                self._get_global_func = tvm_ffi.get_global_func
            except ImportError:
                pass
    
    @property
    def available(self) -> bool:
        return _ffi_available
    
    @property
    def Object(self):
        if self._Object is None:
            raise RuntimeError(
                "tvm_ffi not available. C++ extension not loaded. "
                "Please build caffe-ffi first."
            )
        return self._Object
    
    def get_global_func(self, name: str) -> Any:
        if name not in self._funcs:
            if self._get_global_func is not None:
                self._funcs[name] = self._get_global_func(name, allow_missing=True)
            else:
                self._funcs[name] = None
        return self._funcs[name]
    
    def register_object(self, type_key: str, cls: type):
        if self._register_object is not None:
            self._register_object(type_key)(cls)
        self._types[type_key] = cls


registry = _FFIRegistry()


def get_global_func(name: str) -> Any:
    return registry.get_global_func(name)


def is_available() -> bool:
    return registry.available


def lib_path() -> Optional[Path]:
    return _lib_path
