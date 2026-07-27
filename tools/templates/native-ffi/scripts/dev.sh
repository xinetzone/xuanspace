#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TVM_FFI_PATH="$PROJECT_ROOT/../../vendor/tvm-ffi"
BUILD_DIR="$PROJECT_ROOT/build"
BUILD_LIB_DIR="$BUILD_DIR/lib"
BUILD_MODULE_DIR="$BUILD_DIR/src/{{module_name}}"

export KMP_DUPLICATE_LIB_OK=TRUE

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step() { echo -e "${CYAN}==>${NC} $1"; }
pass() { echo -e "    ${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "    ${RED}[FAIL]${NC} $1"; }
warn() { echo -e "    ${YELLOW}[WARN]${NC} $1"; }

python_module_exists() { python -c "import $1" 2>/dev/null; }

install_tvm_ffi() {
    step "Checking tvm-ffi installation..."
    if python_module_exists tvm_ffi; then
        pass "tvm-ffi already installed"
    else
        if [ ! -d "$TVM_FFI_PATH" ]; then
            fail "tvm-ffi not found at $TVM_FFI_PATH"
            echo "    Please clone tvm-ffi to vendor/tvm-ffi first"
            exit 1
        fi
        step "Installing tvm-ffi from $TVM_FFI_PATH..."
        pip install --no-build-isolation -e "$TVM_FFI_PATH"
        pass "tvm-ffi installed"
    fi
}

build_cpp() {
    step "Configuring and building C++ code..."
    if [ ! -d "$BUILD_DIR" ]; then
        step "Running CMake configure..."
        (cd "$PROJECT_ROOT" && cmake -B build -G Ninja -D{{package_name|upper}}_USE_STUB=ON)
        pass "CMake configured"
    fi
    step "Running CMake build (Release)..."
    (cd "$PROJECT_ROOT" && cmake --build build --config Release)
    pass "C++ build complete"
}

install_package() {
    step "Checking if {{package_name}} is already installed..."
    if python_module_exists {{package_name}}; then
        pass "{{package_name}} already installed in editable mode"
    else
        step "Installing {{package_name}} (editable, no-build-isolation)..."
        (cd "$PROJECT_ROOT" && pip install --no-build-isolation -e .)
        pass "{{package_name}} installed"
    fi
}

setup_library_path() {
    step "Setting up library search paths..."
    if [[ "$(uname)" == "Darwin" ]]; then
        LIB_PATH_VAR="DYLD_LIBRARY_PATH"
    else
        LIB_PATH_VAR="LD_LIBRARY_PATH"
    fi
    local paths_added=0
    for p in "$BUILD_LIB_DIR" "$BUILD_MODULE_DIR" "$BUILD_MODULE_DIR/Release"; do
        if [ -d "$p" ]; then
            export "$LIB_PATH_VAR"="$p:${!LIB_PATH_VAR:-}"
            pass "Added: $p"
            paths_added=$((paths_added + 1))
        fi
    done
    [ "$paths_added" -eq 0 ] && warn "No build directories found"
}

verify_import() {
    step "Verifying {{package_name}} import..."
    setup_library_path
    python -c "
import os, sys
for p in ['$BUILD_LIB_DIR', '$BUILD_MODULE_DIR', '$BUILD_MODULE_DIR/Release']:
    if os.path.exists(p) and hasattr(os, 'add_dll_directory'):
        try: os.add_dll_directory(p)
        except: pass
try:
    import {{package_name}}
    from {{package_name}} import {{module_name}}
    print('{{package_name}} imported successfully')
    cmd = {{module_name}}.tls_command_handle()
    print(f'tls_command_handle returned: {cmd}')
    {{module_name}}.runtime_shutdown()
except Exception as e:
    print(f'Import failed: {e}', file=sys.stderr)
    sys.exit(1)
"
    pass "{{package_name}} import verified"
}

run_tests() {
    step "Running pytest..."
    setup_library_path
    (cd "$PROJECT_ROOT" && pytest tests/python -v)
    pass "All tests passed"
}

clean_build() {
    step "Cleaning build directory..."
    [ -d "$BUILD_DIR" ] && rm -rf "$BUILD_DIR" && pass "Build directory removed" || warn "No build directory"
}

show_help() {
    echo "{{package_name}} Development Setup Script"
    echo "Usage: $0 [-b|-i|-t|-c|-r|-h]"
    echo "  -b  Build only, -i  Install only, -t  Test, -c  Clean, -r  Rebuild, -h  Help"
}

BUILD_ONLY=0; INSTALL_ONLY=0; RUN_TEST=0; CLEAN_ONLY=0; REBUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD_ONLY=1; shift ;;
        -i|--install) INSTALL_ONLY=1; shift ;;
        -t|--test) RUN_TEST=1; shift ;;
        -c|--clean) CLEAN_ONLY=1; shift ;;
        -r|--rebuild) REBUILD=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option"; show_help; exit 1 ;;
    esac
done

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  {{package_name}} Development Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo "Project root: $PROJECT_ROOT"; echo ""

[ "$CLEAN_ONLY" -eq 1 ] && { clean_build; exit 0; }
[ "$REBUILD" -eq 1 ] && { clean_build; install_tvm_ffi; build_cpp; install_package; verify_import; exit 0; }
[ "$BUILD_ONLY" -eq 1 ] && { install_tvm_ffi; build_cpp; exit 0; }
[ "$INSTALL_ONLY" -eq 1 ] && { install_tvm_ffi; install_package; exit 0; }
[ "$RUN_TEST" -eq 1 ] && { run_tests; exit 0; }

install_tvm_ffi; build_cpp; install_package; verify_import

echo ""; echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"; echo -e "${GREEN}========================================${NC}"; echo ""
