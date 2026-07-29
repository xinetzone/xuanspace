#!/bin/bash
# =============================================================================
# conda.recipe/build.sh — Linux/macOS conda-build 构建脚本
#
# conda-build 环境变量:
#   $PREFIX     — 安装目标前缀（conda 环境目录）
#   $PYTHON     — Python 解释器路径
#   $SRC_DIR    — 源码目录（即 ..，由 meta.yaml source.path 决定）
#   $CPU_COUNT  — CPU 核心数（并行编译）
#   $PIP_NO_BUILD_ISOLATION — 禁用构建隔离（conda 已提供依赖）
# =============================================================================
set -ex

export CMAKE_GENERATOR=Ninja

# ── 查找 tvm-ffi cmake config 目录 ──
TVM_FFI_CMAKE_DIR="$($PYTHON -c 'import tvm_ffi, os; print(os.path.dirname(tvm_ffi.__file__))' 2>/dev/null || echo '')"

# ── 使用 pip + scikit-build-core 构建 ──
# 通过 SKBUILD_CMAKE_ARGS 传递 CMake 参数，确保找到 conda 环境中的依赖
SKBUILD_CMAKE_ARGS="\
-DCMAKE_PREFIX_PATH=${PREFIX};\
-DCMAKE_INSTALL_PREFIX=${PREFIX};\
-DCMAKE_INSTALL_RPATH=${PREFIX}/lib;\
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON;\
-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON;\
-DCMAKE_SKIP_BUILD_RPATH=OFF;\
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON;\
-DCMAKE_BUILD_TYPE=Release;\
-DCAFFE_CPU_ONLY=ON;\
-DCAFFE_USE_BLAS=ON;\
-DTVM_FFI_USE_LIBBACKTRACE=OFF;\
-DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF;\
-DCAFFE_FFI_BUILD_TESTS=OFF;\
"

# 如果找到 tvm-ffi cmake 目录，添加到 CMAKE_PREFIX_PATH
if [ -n "$TVM_FFI_CMAKE_DIR" ]; then
    SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-Dcaffe-ffi_DIR=${TVM_FFI_CMAKE_DIR}"
    echo "[build.sh] tvm-ffi cmake dir: $TVM_FFI_CMAKE_DIR"
fi

echo "[build.sh] CMake args: $SKBUILD_CMAKE_ARGS"
echo "[build.sh] PREFIX: $PREFIX"
echo "[build.sh] Python: $PYTHON ($($PYTHON --version))"
echo "[build.sh] CPU_COUNT: ${CPU_COUNT:-4}"

SKBUILD_CMAKE_ARGS="$SKBUILD_CMAKE_ARGS" \
$PYTHON -m pip install . --no-deps -vv --no-build-isolation
