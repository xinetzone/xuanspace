from __future__ import annotations
from pathlib import Path
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

_EXTRA_LIB_PATHS = [
    _PROJECT_ROOT / "build" / "lib",
    _PROJECT_ROOT / "build" / "src" / "demo" / "Release",
    _PROJECT_ROOT / "build" / "src" / "demo",
]

LIB = _FFI_LOAD_LIB("demo-ffi", "demo_ffi_demo", extra_lib_paths=_EXTRA_LIB_PATHS)

_FFI_INIT_FUNC("demo", __name__)
_FFI_INIT_FUNC("math", __name__)
