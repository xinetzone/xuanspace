#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TVM_FFI_PATH="$PROJECT_ROOT/../../vendor/tvm-ffi"
BUILD_DIR="$PROJECT_ROOT/build"
BUILD_LIB_DIR="$BUILD_DIR/lib"
BUILD_VTA_DIR="$BUILD_DIR/src/vta"

export KMP_DUPLICATE_LIB_OK=TRUE

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step() {
    echo -e "${CYAN}==>${NC} $1"
}

pass() {
    echo -e "    ${GREEN}[PASS]${NC} $1"
}

fail() {
    echo -e "    ${RED}[FAIL]${NC} $1"
}

warn() {
    echo -e "    ${YELLOW}[WARN]${NC} $1"
}

python_module_exists() {
    python -c "import $1" 2>/dev/null
}

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
        (cd "$PROJECT_ROOT" && cmake -B build -G Ninja -DNPU_FFI_VTA_USE_STUB=ON)
        pass "CMake configured"
    fi
    step "Running CMake build (Release)..."
    (cd "$PROJECT_ROOT" && cmake --build build --config Release)
    pass "C++ build complete"
}

install_package() {
    step "Checking if npu-ffi is already installed..."
    if python_module_exists npu_ffi; then
        pass "npu-ffi already installed in editable mode"
    else
        step "Installing npu-ffi (editable, no-build-isolation)..."
        (cd "$PROJECT_ROOT" && pip install --no-build-isolation -e .)
        pass "npu-ffi installed"
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
    if [ -d "$BUILD_LIB_DIR" ]; then
        export "$LIB_PATH_VAR"="$BUILD_LIB_DIR:${!LIB_PATH_VAR:-}"
        pass "Added: $BUILD_LIB_DIR"
        paths_added=$((paths_added + 1))
    fi
    if [ -d "$BUILD_VTA_DIR" ]; then
        export "$LIB_PATH_VAR"="$BUILD_VTA_DIR:${!LIB_PATH_VAR:-}"
        pass "Added: $BUILD_VTA_DIR"
        paths_added=$((paths_added + 1))
    fi
    
    if [ "$paths_added" -eq 0 ]; then
        warn "No build directories found - skipping library path setup"
    fi
}

verify_import() {
    step "Verifying npu_ffi import..."
    local python_code=$(cat <<ENDOFPYTHON
import os
import sys
build_lib = r'$BUILD_LIB_DIR'
build_vta = r'$BUILD_VTA_DIR'
for p in [build_lib, build_vta]:
    if os.path.exists(p):
        try:
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(p)
        except (OSError, AttributeError):
            pass
try:
    import npu_ffi
    from npu_ffi import vta
    print('npu_ffi imported successfully')
    print(f'vta module available: {vta is not None}')
    cmd = vta.tls_command_handle()
    print(f'tls_command_handle returned: {cmd}')
    vta.runtime_shutdown()
except Exception as e:
    print(f'Import failed: {e}', file=sys.stderr)
    sys.exit(1)
ENDOFPYTHON
)
    if python -c "$python_code"; then
        pass "npu_ffi import and basic test passed"
    else
        fail "npu_ffi import verification failed"
        exit 1
    fi
}

run_tests() {
    step "Running pytest..."
    (cd "$PROJECT_ROOT" && pytest tests/python -v)
    pass "All tests passed"
}

clean_build() {
    step "Cleaning build directory..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        pass "Build directory removed"
    else
        warn "No build directory to clean"
    fi
}

show_help() {
    echo "NPU-FFI Development Setup Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -b, --build     Only build C++ code"
    echo "  -i, --install   Only install pip package"
    echo "  -t, --test      Run pytest"
    echo "  -c, --clean     Clean build directory"
    echo "  -r, --rebuild   Clean + rebuild + reinstall"
    echo "  -h, --help      Show this help message"
    echo ""
}

BUILD_ONLY=0
INSTALL_ONLY=0
RUN_TEST=0
CLEAN_ONLY=0
REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD_ONLY=1; shift ;;
        -i|--install) INSTALL_ONLY=1; shift ;;
        -t|--test) RUN_TEST=1; shift ;;
        -c|--clean) CLEAN_ONLY=1; shift ;;
        -r|--rebuild) REBUILD=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  NPU-FFI Development Environment Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo "Project root: $PROJECT_ROOT"
echo ""

if [ "$CLEAN_ONLY" -eq 1 ]; then
    clean_build
    exit 0
fi

if [ "$REBUILD" -eq 1 ]; then
    clean_build
    install_tvm_ffi
    build_cpp
    install_package
    setup_library_path
    verify_import
    exit 0
fi

if [ "$BUILD_ONLY" -eq 1 ]; then
    install_tvm_ffi
    build_cpp
    exit 0
fi

if [ "$INSTALL_ONLY" -eq 1 ]; then
    install_tvm_ffi
    install_package
    exit 0
fi

if [ "$RUN_TEST" -eq 1 ]; then
    setup_library_path
    run_tests
    exit 0
fi

install_tvm_ffi
build_cpp
install_package
setup_library_path
verify_import

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Quick commands:${NC}"
echo "  -b, --build     Only build C++"
echo "  -i, --install   Only install pip package"
echo "  -t, --test      Run pytest"
echo "  -c, --clean     Clean build"
echo "  -r, --rebuild   Clean + rebuild + install"
echo ""
