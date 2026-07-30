#!/bin/bash
# =============================================================================
# conda.recipe/build.sh — Linux/macOS conda-build 构建脚本
#
# 策略：优先使用本地 tvm-ffi 源码通过 pip install 安装（Docker/SpecWeave 环境），
#       回退到 pip 安装 apache-tvm-ffi wheel/sdist。
#       构建后自动修复 RPATH 和共享库依赖。
#
# 跨平台支持：
#   - Linux:   patchelf + $ORIGIN        + ldd + nm -D
#   - macOS:   install_name_tool + @loader_path + otool -L + nm -gU
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

# ── 平台检测 ──
UNAME_S=$(uname -s)
IS_MACOS=0
IS_LINUX=0
if [ "$UNAME_S" = "Darwin" ]; then
    IS_MACOS=1
    echo "[build.sh] Detected platform: macOS (Darwin)"
elif [ "$UNAME_S" = "Linux" ]; then
    IS_LINUX=1
    echo "[build.sh] Detected platform: Linux"
else
    echo "[build.sh] WARNING: Unknown platform: $UNAME_S, assuming Linux-like behavior"
    IS_LINUX=1
fi

# ── 跨平台工具函数封装 ──

# get_rpath <binary>  — 打印当前RPATH/RUNPATH
get_rpath() {
    local binary="$1"
    if [ "$IS_MACOS" -eq 1 ]; then
        otool -l "$binary" 2>/dev/null | grep -A2 LC_RPATH | grep path | awk '{print $2}' || echo "(no RPATH set)"
    else
        patchelf --print-rpath "$binary" 2>/dev/null || echo "(no RPATH set)"
    fi
}

# set_rpath <binary> <rpath1> [<rpath2> ...]  — 设置RPATH（覆盖旧值）
set_rpath() {
    local binary="$1"
    shift
    local new_rpath=""
    for rp in "$@"; do
        if [ -z "$new_rpath" ]; then
            new_rpath="$rp"
        else
            new_rpath="$new_rpath:$rp"
        fi
    done

    if [ "$IS_MACOS" -eq 1 ]; then
        # macOS: 删除所有旧RPATH，再逐个添加新RPATH
        local old_rpaths
        old_rpaths=$(otool -l "$binary" 2>/dev/null | grep -A2 LC_RPATH | grep path | awk '{print $2}' || true)
        for old_rp in $old_rpaths; do
            install_name_tool -delete_rpath "$old_rp" "$binary" 2>/dev/null || true
        done
        # 添加新RPATH
        local IFS=':'
        for rp in $new_rpath; do
            install_name_tool -add_rpath "$rp" "$binary"
        done
        unset IFS
        # 设置install name id（macOS dylib需要）
        local basename
        basename=$(basename "$binary")
        install_name_tool -id "@rpath/$basename" "$binary" 2>/dev/null || true
    else
        # Linux: patchelf直接覆盖
        patchelf --set-rpath "$new_rpath" "$binary"
    fi
}

# check_symbol <binary> <symbol_name>  — 检查符号是否为T类型（定义在text段），返回0=找到
check_symbol() {
    local binary="$1"
    local symbol="$2"
    if [ "$IS_MACOS" -eq 1 ]; then
        # macOS: nm -gU 显示全局外部符号，T = text section
        nm -gU "$binary" 2>/dev/null | grep -E " T _?$symbol$" >/dev/null
    else
        # Linux: nm -D 显示动态符号表，T = text section
        nm -D "$binary" 2>/dev/null | grep -q " T $symbol$"
    fi
}

# check_deps <binary>  — 检查依赖是否都能解析，找不到则返回非0
check_deps() {
    local binary="$1"
    if [ "$IS_MACOS" -eq 1 ]; then
        # macOS: otool -L，检查是否有指向构建目录的引用（非系统/非@rpath/@loader_path路径）
        local bad_refs
        bad_refs=$(otool -L "$binary" 2>/dev/null | grep -v "@rpath\|@loader_path\|/usr/lib\|/System/Library\|$(basename "$binary")" | grep -v "^$binary" | grep -v "^$(dirname "$binary")" | grep "/" || true)
        if [ -n "$bad_refs" ]; then
            echo "[build.sh] WARNING: Potentially non-portable references found:"
            echo "$bad_refs"
            # 不直接报错，因为cross-compiled dylib可能引用build prefix，这在conda-build中会被post-process处理
            return 0
        fi
        return 0
    else
        # Linux: ldd 检查not found
        if ldd "$binary" 2>&1 | grep -q "not found"; then
            ldd "$binary" | grep "not found"
            return 1
        fi
        return 0
    fi
}

# fix_deps <binary> <dep_basename> <new_ref>  — 修复依赖库引用路径
fix_dep_ref() {
    local binary="$1"
    local dep_name="$2"
    local new_ref="$3"
    if [ "$IS_MACOS" -eq 1 ]; then
        # macOS: 找出旧的引用路径，替换为@rpath/<dep_name>
        local old_ref
        old_ref=$(otool -L "$binary" 2>/dev/null | grep "$dep_name" | head -1 | awk '{print $1}' || true)
        if [ -n "$old_ref" ] && [ "$old_ref" != "$new_ref" ] && [[ "$old_ref" != "@rpath/"* ]] && [[ "$old_ref" != "@loader_path/"* ]]; then
            echo "[build.sh] Fixing install name: $old_ref -> $new_ref"
            install_name_tool -change "$old_ref" "$new_ref" "$binary"
        fi
    fi
    # Linux: 通过RPATH解决，不需要修改NEEDED条目
}

# ── Helper: clean editable finder files (triple-protection strategy) ──
clean_editable_files() {
    for _sp in $($PYTHON -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null) "${SP_DIR:-}"; do
        if [ -n "$_sp" ] && [ -d "$_sp" ]; then
            # Remove all _editable* and __editable__* files (both .pth and .py, any variant)
            find "$_sp" -maxdepth 1 \( -name "_editable_*" -o -name "__editable__*" \) -type f -delete 2>/dev/null || true
            # Remove any .pth files that point to source paths (xuanspace/SpecWeave/build/_skbuild)
            find "$_sp" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | while read -r f; do
                if grep -q "xuanspace\|SpecWeave\|_skbuild" "$f" 2>/dev/null; then
                    rm -f "$f"
                fi
            done
        fi
    done
}

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

    # Remove editable finder files from tvm-ffi pip install (triple-protection: stage 1)
    clean_editable_files
    echo "[build.sh] Cleaned tvm-ffi editable finder files"

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

    # 检查关键符号是否为T类型（定义在text段，外部可见）
    echo "[build.sh] Checking for TVMFFIGetCustomAllocator symbol..."
    if check_symbol "$TVM_FFI_LIB" "TVMFFIGetCustomAllocator"; then
        echo "[build.sh] TVMFFIGetCustomAllocator symbol verified (T) on $UNAME_S"
    else
        echo "[build.sh] WARNING: TVMFFIGetCustomAllocator symbol not found as T in libtvm_ffi.so"
        if [ "$IS_MACOS" -eq 1 ]; then
            nm -gU "$TVM_FFI_LIB" 2>/dev/null | grep -i TVMFFIGetCustomAllocator || true
        else
            nm -D "$TVM_FFI_LIB" 2>/dev/null | grep -i TVMFFIGetCustomAllocator || true
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
# RPATH 使用全相对路径（避免 conda-build prefix replacement "Placeholder too short" 错误）：
#
# Linux ($ORIGIN):
#   $ORIGIN                    — _caffe_ffi.so 所在目录（SP_DIR/caffe_ffi/）
#   $ORIGIN/lib                — 同目录下的 lib/ 子目录（预留私有库）
#   $ORIGIN/../tvm_ffi/lib     — 同级 tvm_ffi 包的 lib/ 目录（libtvm_ffi.so 在这里）
#   $ORIGIN/../../..           — 上溯3级到达 PREFIX/lib（conda 标准系统库路径）
#
# macOS (@loader_path):
#   @loader_path                    — _caffe_ffi.so 所在目录（SP_DIR/caffe_ffi/）
#   @loader_path/lib                — 同目录下的 lib/ 子目录
#   @loader_path/../tvm_ffi/lib     — 同级 tvm_ffi 包的 lib/ 目录
#   @loader_path/../../..           — 上溯3级到达 PREFIX/lib
#
# 注意：禁止使用 ${PREFIX}/lib 绝对路径！必须用相对路径。

# 平台相关的RPATH前缀
# Note: CMAKE_MACOSX_RPATH/CMAKE_INSTALL_NAME_DIR (macOS) and CMAKE_BUILD_RPATH_USE_ORIGIN (Linux)
# are now set via [tool.scikit-build.overrides] in pyproject.toml, so we don't need _EXTRA_CMAKE_ARGS here.
if [ "$IS_MACOS" -eq 1 ]; then
    RPATH_PREFIX="@loader_path"
else
    RPATH_PREFIX='$ORIGIN'
fi

_CAFFE_OLD_CMAKE_ARGS="${CMAKE_ARGS:-}"
export CMAKE_ARGS=""

# 构建_caffe_ffi.so和libtvm_ffi.so的RPATH列表（按平台前缀）
# Note: The following CMake args are set in pyproject.toml (project defaults) and do NOT need
# to be repeated here:
#   - CMAKE_BUILD_TYPE=Release              → cmake.build-type
#   - CAFFE_CPU_ONLY=ON                     → cmake.define
#   - CAFFE_USE_BLAS=ON                     → cmake.define
#   - CAFFE_FFI_BUILD_TESTS=OFF             → cmake.define
#   - CMAKE_SKIP_BUILD_RPATH=OFF            → cmake.define
#   - CMAKE_BUILD_WITH_INSTALL_RPATH=ON     → cmake.define
#   - CMAKE_POSITION_INDEPENDENT_CODE=ON    → cmake.define
#   - CMAKE_BUILD_RPATH_USE_ORIGIN=ON       → Linux override (if.platform-system="linux")
#   - CMAKE_MACOSX_RPATH=ON                 → macOS override (if.platform-system="^darwin")
#   - CMAKE_INSTALL_NAME_DIR=@rpath         → macOS override
# Only conda-specific args (runtime-dependent or post-processed) remain here:
_CAFFE_RPATH="${RPATH_PREFIX}:${RPATH_PREFIX}/lib:${RPATH_PREFIX}/../tvm_ffi/lib:${RPATH_PREFIX}/../../.."
_TVM_RPATH="${RPATH_PREFIX}:${RPATH_PREFIX}/..:${RPATH_PREFIX}/../../../../lib"

export SKBUILD_CMAKE_ARGS="\
-DCMAKE_PREFIX_PATH=${PREFIX} \
-DCMAKE_INSTALL_RPATH=${_CAFFE_RPATH} \
-DCAFFE_FFI_PREFER_SYSTEM_TVM_FFI=ON \
-DTVM_FFI_USE_LIBBACKTRACE=OFF \
-DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF"

echo "[build.sh] ============================================================"
echo "[build.sh] CMake args (SKBUILD_CMAKE_ARGS): $SKBUILD_CMAKE_ARGS"
echo "[build.sh] RPATH prefix: $RPATH_PREFIX"
echo "[build.sh] --- conda-build environment variables ---"
echo "[build.sh] PREFIX:        $PREFIX"
echo "[build.sh] BUILD_PREFIX:  ${BUILD_PREFIX:-not set}"
echo "[build.sh] SP_DIR:        ${SP_DIR:-unknown}"
echo "[build.sh] SRC_DIR:       $SRC_DIR"
echo "[build.sh] Python:        $PYTHON ($($PYTHON --version 2>&1))"
echo "[build.sh] CPU_COUNT:     ${CPU_COUNT:-4}"
echo "[build.sh] tvm-ffi src:   $TVM_FFI_SOURCE (${LOCAL_TVM_FFI_DIR:-PyPI wheel})"
echo "[build.sh] Platform:      $UNAME_S"
echo "[build.sh] ============================================================"

# ── 3. 构建并安装 ──
# 清理源码目录中的 in-tree 构建残留
rm -rf build _skbuild dist *.egg-info 2>/dev/null || true
rm -f python/caffe_ffi/*.so python/caffe_ffi/lib/*.so 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

$PYTHON -m pip install . --no-deps -vv --no-build-isolation

# Remove editable finder files after caffe-ffi pip install (triple-protection: stage 2)
# (scikit-build-core should not install these for non-editable installs,
# but --no-build-isolation can cause leftover artifacts from previous builds)
_n_editable=0
for _sp in $($PYTHON -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null) "${SP_DIR:-}"; do
  if [ -n "$_sp" ] && [ -d "$_sp" ]; then
    _n=$(find "$_sp" -maxdepth 1 \( -name "_editable_*" -o -name "__editable__*" \) -type f 2>/dev/null | wc -l)
    _n_editable=$((_n_editable + _n))
  fi
done
if [ "$_n_editable" -gt 0 ]; then
  echo "[build.sh] Found $_n_editable editable finder files, removing..."
  clean_editable_files
  echo "[build.sh] Editable files cleaned"
else
  echo "[build.sh] No editable finder files found (good)"
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

# 用 nm/otool 再次确认 TVMFFIGetCustomAllocator 符号存在
echo "[build.sh] Verifying TVMFFIGetCustomAllocator symbol in installed libtvm_ffi..."
if check_symbol "$TVM_FFI_LIB" "TVMFFIGetCustomAllocator"; then
    echo "[build.sh] TVMFFIGetCustomAllocator symbol present"
else
    echo "[build.sh] WARNING: TVMFFIGetCustomAllocator symbol not found after install"
    if [ "$IS_MACOS" -eq 1 ]; then
        nm -gU "$TVM_FFI_LIB" | grep -i TVMFFI || true
    else
        nm -D "$TVM_FFI_LIB" | grep TVMFFI || true
    fi
fi

# 4c. 设置 RPATH（全部使用相对路径，避免 conda prefix placeholder 长度问题）
echo "[build.sh] Current RPATH of _caffe_ffi.so: $(get_rpath "$CAFFE_FFI_SO")"

# 相对路径（避免绝对路径导致 conda-build prefix replacement 时 placeholder 长度不足）：
# Linux:   $ORIGIN/...
# macOS:   @loader_path/...
# _caffe_ffi.so 在 SP_DIR/caffe_ffi/：
#   <prefix>/lib/python3.x/site-packages/caffe_ffi/_caffe_ffi.so
#   到 PREFIX/lib 需要上溯3级：caffe_ffi → site-packages → python3.x → lib
#   到 tvm_ffi/lib 需要上溯1级：caffe_ffi → site-packages，然后 tvm_ffi/lib
set_rpath "$CAFFE_FFI_SO" \
    "${RPATH_PREFIX}" \
    "${RPATH_PREFIX}/lib" \
    "${RPATH_PREFIX}/../tvm_ffi/lib" \
    "${RPATH_PREFIX}/../../.."
echo "[build.sh] Set _caffe_ffi.so RPATH to platform-relative paths"
echo "[build.sh] New RPATH: $(get_rpath "$CAFFE_FFI_SO")"

# 4c-bis. 同样为 libtvm_ffi.so 设置相对 RPATH
# libtvm_ffi.so 在 SP_DIR/tvm_ffi/lib/（比 _caffe_ffi.so 深一级）：
#   <prefix>/lib/python3.x/site-packages/tvm_ffi/lib/libtvm_ffi.so
#   到 PREFIX/lib 需要上溯4级：lib → tvm_ffi → site-packages → python3.x → lib
if [ -n "${TVM_FFI_LIB:-}" ] && [ -f "$TVM_FFI_LIB" ]; then
    echo "[build.sh] Current RPATH of libtvm_ffi.so: $(get_rpath "$TVM_FFI_LIB")"
    set_rpath "$TVM_FFI_LIB" \
        "${RPATH_PREFIX}" \
        "${RPATH_PREFIX}/.." \
        "${RPATH_PREFIX}/../../../../lib"
    echo "[build.sh] Set libtvm_ffi.so RPATH to platform-relative paths"
    echo "[build.sh] New libtvm_ffi.so RPATH: $(get_rpath "$TVM_FFI_LIB")"

    # macOS: 修复_caffe_ffi.so对libtvm_ffi.dylib的引用路径（install name）
    if [ "$IS_MACOS" -eq 1 ]; then
        fix_dep_ref "$CAFFE_FFI_SO" "libtvm_ffi" "@rpath/libtvm_ffi.dylib"
    fi
fi

# 4d. 运行依赖检查
echo "[build.sh] ============================================================"
echo "[build.sh] Dependency check on _caffe_ffi.so:"
if [ "$IS_MACOS" -eq 1 ]; then
    otool -L "$CAFFE_FFI_SO" || true
    check_deps "$CAFFE_FFI_SO"
else
    ldd "$CAFFE_FFI_SO" || true
    if ! check_deps "$CAFFE_FFI_SO"; then
        echo "[build.sh] ERROR: Some shared library dependencies are unresolved!"
        exit 1
    fi
    echo "[build.sh] All shared library dependencies resolved successfully"
fi

echo "[build.sh] ============================================================"
echo "[build.sh] Build completed successfully!"
echo "[build.sh] _caffe_ffi.so location: $CAFFE_FFI_SO"
echo "[build.sh] libtvm_ffi.so location: $TVM_FFI_LIB"
echo "[build.sh] ============================================================"
