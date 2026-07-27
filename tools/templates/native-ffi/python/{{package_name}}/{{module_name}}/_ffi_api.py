from pathlib import Path
from tvm_ffi import Object as _ffi_Object, init_ffi_api as _FFI_INIT_FUNC, register_object as _FFI_REG_OBJ
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm_ffi import Object

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent

# DLL path handling pattern (insight 4):
# When building in editable mode, the compiled shared library is not installed
# to site-packages. We need to search multiple possible build output directories
# to find the shared library across different platforms and build configurations:
#   - build/lib/ : standard output for some CMake configs
#   - build/src/{{module_name}}/Release/ : Windows/MSVC Release output
#   - build/src/{{module_name}}/ : Linux/macOS and general output
# os.add_dll_directory() is called on Windows via load_lib_module to add these
# paths to the DLL search directory before loading the library.
_EXTRA_LIB_PATHS = [
    _PROJECT_ROOT / "build" / "lib",
    _PROJECT_ROOT / "build" / "src" / "{{module_name}}" / "Release",
    _PROJECT_ROOT / "build" / "src" / "{{module_name}}",
]

LIB = _FFI_LOAD_LIB("{{name}}", "{{package_name}}_{{module_name}}", extra_lib_paths=_EXTRA_LIB_PATHS)

# IMPORTANT: The prefix "{{module_name}}" here MUST exactly match the prefix used
# in TVM_FFI_STATIC_INIT_BLOCK() .def() calls in src/{{module_name}}/ffi_registry.cc
# Use scripts/check_ffi_prefix.py to verify C++/Python prefix consistency.
_FFI_INIT_FUNC("{{module_name}}", __name__)
