from pathlib import Path
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

_EXTRA_LIB_PATHS = [
    _PROJECT_ROOT / "build" / "lib",
    _PROJECT_ROOT / "build" / "src" / "vta" / "Release",
    _PROJECT_ROOT / "build" / "src" / "vta",
]

LIB = _FFI_LOAD_LIB("npu-ffi", "npu_ffi_vta", extra_lib_paths=_EXTRA_LIB_PATHS)

_FFI_INIT_FUNC("vta", __name__)
