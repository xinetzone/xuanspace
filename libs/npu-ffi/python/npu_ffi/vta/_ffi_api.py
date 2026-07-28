import os
import sys
from pathlib import Path

import tvm_ffi
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB

_MIN_TVM_FFI_VERSION = (0, 0, 0)
_EXPECTED_TVM_FFI_VERSION = "0.0.1"

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

_EXTRA_LIB_PATHS = [
    _PROJECT_ROOT / "build" / "lib",
    _PROJECT_ROOT / "build" / "src" / "vta" / "Release",
    _PROJECT_ROOT / "build" / "src" / "vta",
]


def _check_tvm_ffi_version() -> None:
    version_str = getattr(tvm_ffi, "__version__", "0.0.0")
    try:
        version_parts = []
        for part in version_str.split(".")[:3]:
            for i, c in enumerate(part):
                if not c.isdigit():
                    part = part[:i]
                    break
            version_parts.append(int(part) if part else 0)
        while len(version_parts) < 3:
            version_parts.append(0)
        version_tuple = tuple(version_parts)
    except (ValueError, AttributeError):
        version_tuple = (0, 0, 0)

    if version_tuple < _MIN_TVM_FFI_VERSION:
        raise ImportError(
            f"npu-ffi requires tvm-ffi >= {_EXPECTED_TVM_FFI_VERSION}, "
            f"but found version {version_str}.\n"
            f"Please upgrade tvm-ffi:\n"
            f"  pip install --no-build-isolation -e vendor/tvm-ffi\n"
            f"\n"
            f"Note: Both tvm-ffi and npu-ffi must be installed with --no-build-isolation."
        )


def _init_lib_path() -> None:
    build_dirs = list(_PROJECT_ROOT.glob("build/lib.*"))
    build_dirs.extend(_EXTRA_LIB_PATHS)

    for lib_dir in build_dirs:
        if not lib_dir.is_dir():
            continue
        if sys.platform.startswith("win"):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(lib_dir))
                except (OSError, FileNotFoundError):
                    pass
            else:
                os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
        elif sys.platform.startswith("darwin"):
            os.environ["DYLD_LIBRARY_PATH"] = (
                str(lib_dir) + os.pathsep + os.environ.get("DYLD_LIBRARY_PATH", "")
            )
        else:
            os.environ["LD_LIBRARY_PATH"] = (
                str(lib_dir) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            )


_check_tvm_ffi_version()
_init_lib_path()

try:
    LIB = _FFI_LOAD_LIB("npu-ffi", "npu_ffi_vta", extra_lib_paths=_EXTRA_LIB_PATHS)
except RuntimeError as e:
    raise RuntimeError(
        f"Failed to load npu_ffi_vta shared library.\n"
        f"Error: {e}\n\n"
        f"Possible causes:\n"
        f"1. C++ extension not built. Run one of:\n"
        f"   - python scripts/dev.ps1 (Windows) or ./scripts/dev.sh (Linux/macOS)\n"
        f"   - pip install --no-build-isolation -e .\n"
        f"2. Build directory not in library search path.\n"
        f"3. On Windows, make sure KMP_DUPLICATE_LIB_OK=TRUE is set.\n"
        f"\n"
        f"Searched directories:\n"
        + "\n".join(f"  - {p}" for p in _EXTRA_LIB_PATHS)
    ) from e

_FFI_INIT_FUNC("vta", __name__)
