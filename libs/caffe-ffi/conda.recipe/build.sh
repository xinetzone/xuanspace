#!/bin/bash
# =============================================================================
# conda.recipe/build.sh — Linux/macOS conda-build 构建脚本
#
# 策略：优先使用本地 tvm-ffi 源码通过 pip install 安装（Docker/SpecWeave 环境），
#       回退到 pip 安装 apache-tvm-ffi wheel/sdist。
#       构建后自动修复 RPATH 和共享库依赖。
#
# conda-build 环境变量:
#   $PREFIX     — 安装目标前缀（conda 环境目录）
#   $PYTHON     — Python 解释器路径
#   $SRC_DIR    — 源码目录（即 ..，由 meta.yaml source.path 决定）
#   $SP_DIR     — site-packages 目录
#   $CPU_COUNT  — CPU 核心数（并行编译）
# =============================================================================
set -eux -o pipefail

export CMAKE_GENERATOR=Ninja

# ── 0. 源码预处理：CRLF 修复 + in-tree 构建残留清理 ──
echo "[build.sh] Preprocessing source directory..."

# 修复 CRLF 换行符（防止 Windows  checkout 导致的脚本执行问题）
if command -v dos2unix &>/dev/null; then
    find "$SRC_DIR" -type f \( -name "*.sh" -o -name "*.py" -o -name "*.cmake" -o -name "CMakeLists.txt" \) -exec dos2unix {} \; 2>/dev/null || true
fi

# 清理 in-tree 构建残留
rm -rf "${SRC_DIR}/build" "${SRC_DIR}/_skbuild" "${SRC_DIR}/dist" "${SRC_DIR}/*.egg-info" 2>/dev/null || true

# ── 1. 检测 tvm-ffi 来源：优先本地源码 pip 安装，回退 PyPI pip ──
TVM_FFI_SOURCE="none"
LOCAL_TVM_FFI_DIR=""

# 1a. 检查是否通过环境变量显式指定
if [ -n "${CAFFE_FFI_TVM_FFI_DIR:-}" ] && [ -f "${CAFFE_FFI_TVM_FFI_DIR}/CMakeLists.txt" ]; then
    LOCAL_TVM_FFI_DIR="${CAFFE_FFI_TVM_FFI_DIR}"
    TVM_FFI_SOURCE="local-pip"
    echo "[build.sh] Using tvm-ffi from CAFFE_FFI_TVM_FFI_DIR: ${LOCAL_TVM_FFI_DIR}"
fi

# 1b. 自动检测 SpecWeave 挂载路径（Docker 容器中 /SpecWeave 是 bind mount）
if [ "$TVM_FFI_SOURCE" = "none" ]; then
    for candidate in \
        "/SpecWeave/projects/xuanspace/vendor/tvm-ffi" \
        "${SRC_DIR}/../../../vendor/tvm-ffi" \
        "${SRC_DIR}/../../vendor/tvm-ffi"; do
        if [ -f "$candidate/CMakeLists.txt" ]; then
            LOCAL_TVM_FFI_DIR="$candidate"
            TVM_FFI_SOURCE="local-pip"
            echo "[build.sh] Auto-detected local tvm-ffi: $candidate"
            break
        fi
    done
fi

# 1c. 本地源码模式：pip install 本地 tvm-ffi
if [ "$TVM_FFI_SOURCE" = "local-pip" ]; then
    echo "[build.sh] Installing local tvm-ffi via pip..."

    # Bypass setuptools_scm git describe (git submodule paths may be Windows paths in Docker)
    export SETUPTOOLS_SCM_PRETEND_VERSION="0.1.13"

    # Clean tvm-ffi in-tree build artifacts before pip install
    rm -rf "${LOCAL_TVM_FFI_DIR}/build" "${LOCAL_TVM_FFI_DIR}/_skbuild" "${LOCAL_TVM_FFI_DIR}/dist" 2>/dev/null || true
    find "${LOCAL_TVM_FFI_DIR}" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    find "${LOCAL_TVM_FFI_DIR}/python" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    rm -f "${LOCAL_TVM_FFI_DIR}/python/tvm_ffi/"*.so 2>/dev/null || true
    rm -f "${LOCAL_TVM_FFI_DIR}/python/tvm_ffi/lib/"libtvm_ffi* 2>/dev/null || true

    # 保存当前 SKBUILD_CMAKE_ARGS（为 caffe-ffi 准备的）
    _OLD_SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:-}"
    unset SKBUILD_CMAKE_ARGS || true

    # 为 tvm-ffi 设置独立的 CMake 参数
    # 注意：pyproject.toml 中的 cmake.args 会被 scikit-build-core 自动应用（TVM_FFI_BUILD_PYTHON_MODULE=ON等）
    # 我们临时清空 conda 的 CMAKE_ARGS（避免其 FIND_ROOT_PATH 等设置干扰 tvm-ffi 独立构建），
    # 仅追加必要的关闭选项
    _OLD_CMAKE_ARGS="${CMAKE_ARGS:-}"
    export CMAKE_ARGS="-DTVM_FFI_USE_LIBBACKTRACE=OFF -DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF"

    $PYTHON -m pip install "${LOCAL_TVM_FFI_DIR}" --no-deps -vv --no-build-isolation || {
        echo "[build.sh] ERROR: Failed to install local tvm-ffi via pip"
        exit 1
    }

    # 恢复或清理 SKBUILD_CMAKE_ARGS
    if [ -n "$_OLD_SKBUILD_CMAKE_ARGS" ]; then
        export SKBUILD_CMAKE_ARGS="$_OLD_SKBUILD_CMAKE_ARGS"
    else
        unset SKBUILD_CMAKE_ARGS || true
    fi

    # 恢复 CMAKE_ARGS
    if [ -n "$_OLD_CMAKE_ARGS" ]; then
        export CMAKE_ARGS="$_OLD_CMAKE_ARGS"
    else
        unset CMAKE_ARGS || true
    fi

    # 验证 tvm-ffi 安装成功
    echo "[build.sh] Verifying tvm_ffi import..."
    TVM_FFI_BASE=$($PYTHON -c "import tvm_ffi; import os; print(os.path.dirname(tvm_ffi.__file__))")
    echo "[build.sh] tvm_ffi installed at: $TVM_FFI_BASE"

    # 验证 libtvm_ffi.so 存在
    TVM_FFI_LIB="${TVM_FFI_BASE}/lib/libtvm_ffi.so"
    if [ ! -f "$TVM_FFI_LIB" ]; then
        echo "[build.sh] ERROR: libtvm_ffi.so not found at expected location: $TVM_FFI_LIB"
        echo "[build.sh] Contents of tvm_ffi directory:"
        ls -laR "$TVM_FFI_BASE"
        exit 1
    fi
    echo "[build.sh] libtvm_ffi.so found: $TVM_FFI_LIB"

    # 用 nm 检查 TVMFFIGetCustomAllocator 符号（应该是 T 符号）
    echo "[build.sh] Checking for TVMFFIGetCustomAllocator symbol..."
    if command -v nm &>/dev/null; then
        if ! nm -D "$TVM_FFI_LIB" | grep -q "T TVMFFIGetCustomAllocator"; then
            echo "[build.sh] WARNING: TVMFFIGetCustomAllocator symbol not found as T (global text) in libtvm_ffi.so"
            nm -D "$TVM_FFI_LIB" | grep -i TVMFFIGetCustomAllocator || true
        else
            echo "[build.sh] TVMFFIGetCustomAllocator symbol verified (T)"
        fi
    fi
fi

# 1d. 回退：pip 安装 apache-tvm-ffi（需要编译工具链）
if [ "$TVM_FFI_SOURCE" = "none" ]; then
    echo "[build.sh] No local tvm-ffi found, installing apache-tvm-ffi via pip..."
    $PYTHON -m pip install --no-deps 'apache-tvm-ffi' || {
        echo "[build.sh] ERROR: Failed to install apache-tvm-ffi via pip"
        exit 1
    }
    TVM_FFI_SOURCE="pip"
    echo "[build.sh] Installed tvm-ffi via pip (PyPI)"

    # 验证 tvm-ffi 安装成功
    TVM_FFI_BASE=$($PYTHON -c "import tvm_ffi; import os; print(os.path.dirname(tvm_ffi.__file__))")
    echo "[build.sh] tvm_ffi installed at: $TVM_FFI_BASE"
fi

# ── 2. 配置 CMake 参数 ──
# 注意：
#   - 不要设置 CMAKE_INSTALL_PREFIX，scikit-build-core 会自动管理 wheel 安装目录
#   - 使用空格分隔（不是分号），避免被 CMake 解析为列表分隔符
#   - 临时清空 conda 的 CMAKE_ARGS（包含 -DCMAKE_INSTALL_PREFIX=$PREFIX 等），
#     避免干扰 scikit-build-core 的 wheel 构建过程
# RPATH 使用相对路径：
#   $ORIGIN               — _caffe_ffi.so 所在目录（SP_DIR/caffe_ffi/）
#   $ORIGIN/lib           — 同目录下的 lib/ 子目录（预留）
#   $ORIGIN/../tvm_ffi/lib — 同级 tvm_ffi 包的 lib/ 目录（libtvm_ffi.so 在这里）
#   $PREFIX/lib           — conda 环境标准库路径（protobuf、openblas 等兜底）
_CAFFE_OLD_CMAKE_ARGS="${CMAKE_ARGS:-}"
export CMAKE_ARGS=""

export SKBUILD_CMAKE_ARGS="\
-DCMAKE_PREFIX_PATH=${PREFIX} \
-DCMAKE_INSTALL_RPATH=\$ORIGIN:\$ORIGIN/lib:\$ORIGIN/../tvm_ffi/lib:${PREFIX}/lib \
-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
-DCMAKE_SKIP_BUILD_RPATH=OFF \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCAFFE_CPU_ONLY=ON \
-DCAFFE_USE_BLAS=ON \
-DCAFFE_FFI_PREFER_SYSTEM_TVM_FFI=ON \
-DTVM_FFI_USE_LIBBACKTRACE=OFF \
-DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF \
-DCAFFE_FFI_BUILD_TESTS=OFF"

echo "[build.sh] ============================================================"
echo "[build.sh] CMake args (SKBUILD_CMAKE_ARGS): $SKBUILD_CMAKE_ARGS"
echo "[build.sh] --- conda-build environment variables ---"
echo "[build.sh] PREFIX:        $PREFIX"
echo "[build.sh] BUILD_PREFIX:  ${BUILD_PREFIX:-not set}"
echo "[build.sh] SP_DIR:        ${SP_DIR:-unknown}"
echo "[build.sh] SRC_DIR:       $SRC_DIR"
echo "[build.sh] Python:        $PYTHON ($($PYTHON --version 2>&1))"
echo "[build.sh] CPU_COUNT:     ${CPU_COUNT:-4}"
echo "[build.sh] tvm-ffi src:   $TVM_FFI_SOURCE (${LOCAL_TVM_FFI_DIR:-PyPI wheel})"
echo "[build.sh] ============================================================"

# ── 3. 构建并安装 ──
# 清理源码目录中的 in-tree 构建残留
rm -rf build _skbuild dist *.egg-info 2>/dev/null || true
rm -f python/caffe_ffi/*.so python/caffe_ffi/lib/*.so 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

$PYTHON -m pip install . --no-deps -vv --no-build-isolation

# Remove editable finder files that pip may have incorrectly installed
# (scikit-build-core should not install these for non-editable installs,
# but --no-build-isolation can cause leftover artifacts from previous builds)
_SP_DIR=$($PYTHON -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
if [ -n "$_SP_DIR" ] && [ -d "$_SP_DIR" ]; then
  rm -f "$_SP_DIR"/_editable_skbc_*.pth "$_SP_DIR"/_editable_skbc_*.py 2>/dev/null || true
  rm -f "$_SP_DIR"/__editable__*.pth 2>/dev/null || true
  echo "[build.sh] Cleaned editable finder files from $_SP_DIR"
fi

# 恢复 CMAKE_ARGS
if [ -n "$_CAFFE_OLD_CMAKE_ARGS" ]; then
    export CMAKE_ARGS="$_CAFFE_OLD_CMAKE_ARGS"
else
    unset CMAKE_ARGS || true
fi

# ── 4. 构建后修复：共享库依赖与 RPATH ──
echo "[build.sh] ============================================================"
echo "[build.sh] Post-build: verifying shared library dependencies..."

# 4a. 定位 _caffe_ffi.so
CAFFE_FFI_LIB_DIR=""
CAFFE_FFI_SO=""

# 优先从 SP_DIR 查找
if [ -n "${SP_DIR:-}" ] && [ -d "${SP_DIR}/caffe_ffi" ]; then
    CAFFE_FFI_SO=$(find "${SP_DIR}/caffe_ffi" -name "_caffe_ffi*.so" -type f 2>/dev/null | head -1 || true)
fi

# 回退：从 PREFIX 全局查找
if [ -z "$CAFFE_FFI_SO" ]; then
    CAFFE_FFI_SO=$(find "${PREFIX}" -name "_caffe_ffi*.so" -type f 2>/dev/null | head -1 || true)
fi

if [ -z "$CAFFE_FFI_SO" ]; then
    echo "[build.sh] ERROR: Cannot find _caffe_ffi shared library after build"
    echo "[build.sh] PREFIX contents (lib):"
    ls -la "${PREFIX}/lib/" 2>/dev/null || true
    echo "[build.sh] SP_DIR contents:"
    find "${SP_DIR:-$PREFIX}" -name "*.so" -type f 2>/dev/null | head -20 || true
    exit 1
fi

CAFFE_FFI_LIB_DIR=$(dirname "$CAFFE_FFI_SO")
echo "[build.sh] Found _caffe_ffi.so: $CAFFE_FFI_SO"
echo "[build.sh] _caffe_ffi.so directory: $CAFFE_FFI_LIB_DIR"

# 4b. 验证 tvm_ffi 包和 libtvm_ffi.so
echo "[build.sh] Verifying tvm_ffi package..."
TVM_FFI_FILE=$($PYTHON -c "import tvm_ffi; print(tvm_ffi.__file__)")
echo "[build.sh] tvm_ffi package location: $TVM_FFI_FILE"

TVM_FFI_LIB=$($PYTHON -c "import tvm_ffi, os; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'lib', 'libtvm_ffi.so'))")
echo "[build.sh] libtvm_ffi.so expected at: $TVM_FFI_LIB"
ls -la "$TVM_FFI_LIB"

# 用 nm 再次确认 TVMFFIGetCustomAllocator 符号存在
if command -v nm &>/dev/null; then
    echo "[build.sh] Verifying TVMFFIGetCustomAllocator symbol in installed libtvm_ffi.so..."
    nm -D "$TVM_FFI_LIB" | grep TVMFFIGetCustomAllocator || {
        echo "[build.sh] WARNING: TVMFFIGetCustomAllocator symbol not found"
    }
fi

# 4c. 用 patchelf 设置 RPATH（相对路径 + PREFIX/lib 兜底）
if command -v patchelf &>/dev/null; then
    echo "[build.sh] Current RPATH of _caffe_ffi.so:"
    patchelf --print-rpath "$CAFFE_FFI_SO" 2>/dev/null || echo "  (no RPATH set)"

    # 相对路径：
    #   $ORIGIN                 — caffe_ffi/ 目录
    #   $ORIGIN/lib             — caffe_ffi/lib/（预留）
    #   $ORIGIN/../tvm_ffi/lib  — 指向 tvm_ffi 包的 lib/ 目录
    # 绝对路径兜底：PREFIX/lib（conda 标准库路径，用于 protobuf、openblas 等）
    NEW_RPATH="\$ORIGIN:\$ORIGIN/lib:\$ORIGIN/../tvm_ffi/lib:${PREFIX}/lib"
    patchelf --set-rpath "$NEW_RPATH" "$CAFFE_FFI_SO"
    echo "[build.sh] Set RPATH to: $NEW_RPATH"
    echo "[build.sh] New RPATH: $(patchelf --print-rpath "$CAFFE_FFI_SO" 2>/dev/null)"
else
    echo "[build.sh] WARNING: patchelf not available, skipping RPATH fix"
fi

# 4d. 运行 ldd 验证依赖解析
echo "[build.sh] ============================================================"
echo "[build.sh] ldd check on _caffe_ffi.so:"
if command -v ldd &>/dev/null; then
    ldd "$CAFFE_FFI_SO" || true

    # 检查是否有 "not found" 的依赖
    if ldd "$CAFFE_FFI_SO" 2>&1 | grep -q "not found"; then
        echo "[build.sh] ERROR: Some shared library dependencies are unresolved!"
        ldd "$CAFFE_FFI_SO" | grep "not found"
        exit 1
    fi
    echo "[build.sh] All shared library dependencies resolved successfully"
else
    echo "[build.sh] ldd not available, skipping dependency check"
fi

echo "[build.sh] ============================================================"
echo "[build.sh] Build completed successfully!"
echo "[build.sh] _caffe_ffi.so location: $CAFFE_FFI_SO"
echo "[build.sh] libtvm_ffi.so location: $TVM_FFI_LIB"
echo "[build.sh] ============================================================"
