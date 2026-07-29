#!/bin/bash
# =============================================================================
# conda.recipe/build.sh — Linux/macOS conda-build 构建脚本
#
# 策略：优先使用本地 tvm-ffi 源码（Docker/SpecWeave 环境），
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

# ── 1. 检测 tvm-ffi 来源：优先本地源码，回退 pip ──
TVM_FFI_SOURCE="none"

# 1a. 检查是否通过环境变量显式指定
if [ -n "${CAFFE_FFI_TVM_FFI_DIR:-}" ] && [ -f "${CAFFE_FFI_TVM_FFI_DIR}/CMakeLists.txt" ]; then
    TVM_FFI_SOURCE="local-env"
    echo "[build.sh] Using tvm-ffi from CAFFE_FFI_TVM_FFI_DIR: ${CAFFE_FFI_TVM_FFI_DIR}"
fi

# 1b. 自动检测 SpecWeave 挂载路径（Docker 容器中 /SpecWeave 是 bind mount）
if [ "$TVM_FFI_SOURCE" = "none" ]; then
    for candidate in \
        "/SpecWeave/projects/xuanspace/vendor/tvm-ffi" \
        "${SRC_DIR}/../../../vendor/tvm-ffi" \
        "${SRC_DIR}/../../vendor/tvm-ffi"; do
        if [ -f "$candidate/CMakeLists.txt" ]; then
            export CAFFE_FFI_TVM_FFI_DIR="$candidate"
            TVM_FFI_SOURCE="local-auto"
            echo "[build.sh] Auto-detected local tvm-ffi: $candidate"
            break
        fi
    done
fi

# 1c. 回退：pip 安装 apache-tvm-ffi（需要编译工具链）
if [ "$TVM_FFI_SOURCE" = "none" ]; then
    echo "[build.sh] No local tvm-ffi found, installing apache-tvm-ffi via pip..."
    $PYTHON -m pip install --no-deps 'apache-tvm-ffi' || {
        echo "[build.sh] ERROR: Failed to install apache-tvm-ffi via pip"
        exit 1
    }
    TVM_FFI_SOURCE="pip"
    echo "[build.sh] Installed tvm-ffi via pip"
fi

# ── 2. 配置 CMake 参数 ──
# 注意：不要设置 CMAKE_INSTALL_PREFIX，scikit-build-core 会自动管理安装目录。
# RPATH 使用相对路径：
#   $ORIGIN       — _caffe_ffi.so 所在目录（SP_DIR/caffe_ffi/）
#   $ORIGIN/lib   — 同目录下的 lib/ 子目录（存放 libtvm_ffi.so 等）
SKBUILD_CMAKE_ARGS="\
-DCMAKE_PREFIX_PATH=${PREFIX};\
-DCMAKE_INSTALL_RPATH=\$ORIGIN:\$ORIGIN/lib;\
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

if [ -n "${CAFFE_FFI_TVM_FFI_DIR:-}" ]; then
    SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DCAFFE_FFI_TVM_FFI_DIR=${CAFFE_FFI_TVM_FFI_DIR}"
fi

export SKBUILD_CMAKE_ARGS

echo "[build.sh] ============================================================"
echo "[build.sh] CMake args: $SKBUILD_CMAKE_ARGS"
echo "[build.sh] PREFIX:      $PREFIX"
echo "[build.sh] SP_DIR:      ${SP_DIR:-unknown}"
echo "[build.sh] Python:      $PYTHON ($($PYTHON --version 2>&1))"
echo "[build.sh] CPU_COUNT:   ${CPU_COUNT:-4}"
echo "[build.sh] tvm-ffi src: $TVM_FFI_SOURCE (${CAFFE_FFI_TVM_FFI_DIR:-pip wheel})"
echo "[build.sh] ============================================================"

# ── 3. 构建并安装 ──
$PYTHON -m pip install . --no-deps -vv --no-build-isolation

# ── 4. 构建后修复：共享库依赖与 RPATH ──
echo "[build.sh] ============================================================"
echo "[build.sh] Post-build: verifying shared library dependencies..."

# 4a. 定位 _caffe_ffi.so
CAFFE_FFI_LIB_DIR=""
CAFFE_FFI_SO=""

# 优先从 SP_DIR 查找
if [ -n "${SP_DIR:-}" ] && [ -d "${SP_DIR}/caffe_ffi" ]; then
    CAFFE_FFI_SO=$(find "${SP_DIR}/caffe_ffi" -name "_caffe_ffi*.so" -type f 2>/dev/null | head -1)
fi

# 回退：从 PREFIX 全局查找
if [ -z "$CAFFE_FFI_SO" ]; then
    CAFFE_FFI_SO=$(find "${PREFIX}" -name "_caffe_ffi*.so" -type f 2>/dev/null | head -1)
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

# 4b. 定位 libtvm_ffi.so
TVM_FFI_SO=""

# 本地源码模式：搜索多个可能的位置
# scikit-build-core 会将文件安装到 wheel 临时目录，再复制到 SP_DIR
if [ "$TVM_FFI_SOURCE" = "local-env" ] || [ "$TVM_FFI_SOURCE" = "local-auto" ]; then
    # 优先从 SP_DIR 搜索（pip install 最终位置）
    if [ -n "${SP_DIR:-}" ]; then
        TVM_FFI_SO=$(find "${SP_DIR}" -name "libtvm_ffi*.so*" -type f 2>/dev/null | grep -v "\.a$" | head -1)
    fi
    # 回退：PREFIX 全局搜索
    if [ -z "$TVM_FFI_SO" ]; then
        TVM_FFI_SO=$(find "${PREFIX}" -name "libtvm_ffi*.so*" -type f 2>/dev/null | grep -v "\.a$" | head -1)
    fi
    # 再回退：在 CMake build 目录搜索
    if [ -z "$TVM_FFI_SO" ] && [ -d "${SRC_DIR}/build" ]; then
        TVM_FFI_SO=$(find "${SRC_DIR}/build" -name "libtvm_ffi*.so*" -type f 2>/dev/null | grep -v "\.a$" | head -1)
    fi
fi

# pip 模式：从 tvm_ffi 包中获取
if [ -z "$TVM_FFI_SO" ]; then
    TVM_FFI_SO=$($PYTHON -c "
import tvm_ffi, os
base = os.path.dirname(tvm_ffi.__file__)
lib = os.path.join(base, 'lib')
found = ''
if os.path.isdir(lib):
    for f in os.listdir(lib):
        if f.startswith('libtvm_ffi') and (f.endswith('.so') or '.so.' in f):
            found = os.path.join(lib, f)
            break
if not found:
    # fallback: search recursively
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.startswith('libtvm_ffi') and (f.endswith('.so') or '.so.' in f):
                found = os.path.join(root, f)
                break
        if found:
            break
print(found)
" 2>/dev/null || echo "")
fi

# 全局兜底搜索
if [ -z "$TVM_FFI_SO" ]; then
    TVM_FFI_SO=$(find "${PREFIX}" -name "libtvm_ffi*.so*" -type f 2>/dev/null | grep -v "\.a$" | head -1)
fi

if [ -n "$TVM_FFI_SO" ]; then
    TVM_FFI_SO_DIR=$(dirname "$TVM_FFI_SO")
    echo "[build.sh] Found libtvm_ffi.so: $TVM_FFI_SO (dir: $TVM_FFI_SO_DIR)"

    # 将 libtvm_ffi.so 复制到 PREFIX/lib/（conda 标准库路径）
    mkdir -p "${PREFIX}/lib"
    if [ "$TVM_FFI_SO_DIR" != "${PREFIX}/lib" ]; then
        cp -ad "${TVM_FFI_SO}"* "${PREFIX}/lib/" 2>/dev/null || true
        echo "[build.sh] Copied libtvm_ffi to ${PREFIX}/lib/"
    fi

    # 也复制到 _caffe_ffi.so 同级目录（$ORIGIN 查找）
    if [ "$CAFFE_FFI_LIB_DIR" != "${PREFIX}/lib" ] && [ "$CAFFE_FFI_LIB_DIR" != "$TVM_FFI_SO_DIR" ]; then
        cp -ad "${TVM_FFI_SO}"* "${CAFFE_FFI_LIB_DIR}/" 2>/dev/null || true
        echo "[build.sh] Copied libtvm_ffi to ${CAFFE_FFI_LIB_DIR}/ (\$ORIGIN)"
    fi
else
    echo "[build.sh] WARNING: libtvm_ffi.so not found after build, DSO resolution may fail"
fi

# 4c. 用 patchelf 设置 RPATH（$ORIGIN 相对路径 + PREFIX/lib 绝对路径兜底）
if command -v patchelf &>/dev/null; then
    echo "[build.sh] Current RPATH of _caffe_ffi.so:"
    patchelf --print-rpath "$CAFFE_FFI_SO" 2>/dev/null || echo "  (no RPATH set)"

    # 相对路径：$ORIGIN（同目录）、$ORIGIN/lib（子目录）
    # 绝对路径兜底：PREFIX/lib（conda 标准库路径，运行时环境一致）
    NEW_RPATH="\$ORIGIN:\$ORIGIN/lib:${PREFIX}/lib"
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
echo "[build.sh] ============================================================"
