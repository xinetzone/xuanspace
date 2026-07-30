#!/bin/bash
# =============================================================================
# conda.recipe/build.sh — Linux/macOS/Windows conda-build 构建脚本
#
# 策略：优先使用本地 tvm-ffi 源码通过 pip install 安装（Docker/SpecWeave 环境），
#       回退到 pip 安装 apache-tvm-ffi wheel/sdist。
#       构建后自动修复 RPATH 和共享库依赖。
#
# 跨平台支持：
#   - Native Linux:     patchelf + $ORIGIN        + ldd + nm -D
#   - Native macOS:     install_name_tool + @loader_path + otool -L + nm -gU
#   - Cross Linux→macOS: cctools (x86_64-apple-darwin20.0.0-*) / llvm-bin-tools + @loader_path
#   - Cross Linux→Windows: 跳过RPATH（PE格式），llvm-objdump验证
#
# conda-build 环境变量:
#   $PREFIX     — 安装目标前缀（conda 环境目录）
#   $HOST_PREFIX — host 前缀（交叉编译时工具可能带三元组前缀）
#   $BUILD_PREFIX — build 前缀
#   $PYTHON     — Python 解释器路径
#   $SRC_DIR    — 源码目录（即 ..，由 meta.yaml source.path 决定）
#   $SP_DIR     — site-packages 目录
#   $CPU_COUNT  — CPU 核心数（并行编译）
#   $CONDA_BUILD_CROSS_COMPILATION — "1" 表示交叉编译
#   $target_platform — 目标平台（如 osx-64, win-64, linux-64）
# =============================================================================
set -eux -o pipefail

export CMAKE_GENERATOR=Ninja

# ── 交叉编译检测 ──
IS_CROSS=0
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-}" == "1" ]]; then
    IS_CROSS=1
fi

# ── 确定可执行的 Python 解释器 ──
# 原生构建：$PYTHON 指向 PREFIX/bin/python，可执行
# 交叉编译：$PYTHON 指向 host PREFIX 的目标平台 Python（不可直接执行），
#           cross-python 包在 PATH 中放置构建平台 python 并设置正确的 sysconfig 环境变量
if [ "$IS_CROSS" -eq 1 ]; then
    PYTHON_EXE="python"
    echo "[build.sh] Cross-compile: using 'python' from PATH (build Python with cross-env)"
else
    PYTHON_EXE="$PYTHON"
    echo "[build.sh] Native build: using \$PYTHON"
fi

# ── 目标平台检测 ──
UNAME_S=$(uname -s)
IS_MACOS=0
IS_LINUX=0
IS_WIN=0
TARGET_OS=""
EXT_SUFFIX=".so"
SHARED_LIB_EXT="so"

if [ "$IS_CROSS" -eq 1 ]; then
    case "${target_platform}" in
        osx-64|osx-arm64)
            IS_MACOS=1
            TARGET_OS="osx"
            EXT_SUFFIX=".so"
            SHARED_LIB_EXT="dylib"
            echo "[build.sh] Cross-compiling to macOS (${target_platform}) from ${UNAME_S}"
            ;;
        win-64)
            IS_WIN=1
            TARGET_OS="win"
            EXT_SUFFIX=".pyd"
            SHARED_LIB_EXT="dll"
            echo "[build.sh] Cross-compiling to Windows (${target_platform}) from ${UNAME_S}"
            ;;
        linux-64|linux-aarch64)
            IS_LINUX=1
            TARGET_OS="linux"
            EXT_SUFFIX=".so"
            SHARED_LIB_EXT="so"
            echo "[build.sh] Cross-compiling to Linux (${target_platform}) from ${UNAME_S}"
            ;;
        *)
            echo "[build.sh] WARNING: Unknown target_platform: ${target_platform}, assuming Linux-like"
            IS_LINUX=1
            TARGET_OS="linux"
            ;;
    esac
else
    if [ "$UNAME_S" = "Darwin" ]; then
        IS_MACOS=1
        TARGET_OS="osx"
        EXT_SUFFIX=".so"
        SHARED_LIB_EXT="dylib"
        echo "[build.sh] Detected platform: macOS (Darwin) [native]"
    elif [ "$UNAME_S" = "Linux" ]; then
        IS_LINUX=1
        TARGET_OS="linux"
        echo "[build.sh] Detected platform: Linux [native]"
    else
        echo "[build.sh] WARNING: Unknown platform: $UNAME_S, assuming Linux-like behavior"
        IS_LINUX=1
        TARGET_OS="linux"
    fi
fi

# ── 工具路径设置（交叉编译时使用host前缀的工具或llvm工具）──
OTOOL="otool"
INSTALL_NAME_TOOL="install_name_tool"
NM="nm"
OBJDUMP="objdump"

if [ "$IS_CROSS" -eq 1 ] && [ "$IS_MACOS" -eq 1 ]; then
    TRIPLE=""
    case "${target_platform}" in
        osx-64) TRIPLE="x86_64-apple-darwin20.0.0" ;;
        osx-arm64) TRIPLE="arm64-apple-darwin20.0.0" ;;
    esac
    for candidate in \
        "${HOST_PREFIX}/bin/${TRIPLE}-otool" \
        "${BUILD_PREFIX}/bin/${TRIPLE}-otool" \
        "${PREFIX}/bin/${TRIPLE}-otool" \
        "${HOST_PREFIX}/bin/llvm-otool" \
        "${BUILD_PREFIX}/bin/llvm-otool" \
        "${PREFIX}/bin/llvm-otool"; do
        if [ -x "$candidate" ]; then
            OTOOL="$candidate"
            break
        fi
    done
    for candidate in \
        "${HOST_PREFIX}/bin/${TRIPLE}-install_name_tool" \
        "${BUILD_PREFIX}/bin/${TRIPLE}-install_name_tool" \
        "${PREFIX}/bin/${TRIPLE}-install_name_tool" \
        "${HOST_PREFIX}/bin/llvm-install-name-tool" \
        "${BUILD_PREFIX}/bin/llvm-install-name-tool" \
        "${PREFIX}/bin/llvm-install-name-tool"; do
        if [ -x "$candidate" ]; then
            INSTALL_NAME_TOOL="$candidate"
            break
        fi
    done
    for candidate in \
        "${HOST_PREFIX}/bin/llvm-nm" \
        "${BUILD_PREFIX}/bin/llvm-nm" \
        "${PREFIX}/bin/llvm-nm" \
        "${HOST_PREFIX}/bin/${TRIPLE}-nm" \
        "${BUILD_PREFIX}/bin/${TRIPLE}-nm"; do
        if [ -x "$candidate" ]; then
            NM="$candidate"
            break
        fi
    done
    echo "[build.sh] Using cross tools for macOS:"
    echo "[build.sh]   OTOOL=$OTOOL"
    echo "[build.sh]   INSTALL_NAME_TOOL=$INSTALL_NAME_TOOL"
    echo "[build.sh]   NM=$NM"
elif [ "$IS_CROSS" -eq 1 ] && [ "$IS_WIN" -eq 1 ]; then
    for candidate in \
        "${HOST_PREFIX}/bin/llvm-objdump" \
        "${BUILD_PREFIX}/bin/llvm-objdump" \
        "${PREFIX}/bin/llvm-objdump"; do
        if [ -x "$candidate" ]; then
            OBJDUMP="$candidate"
            break
        fi
    done
    for candidate in \
        "${HOST_PREFIX}/bin/llvm-nm" \
        "${BUILD_PREFIX}/bin/llvm-nm" \
        "${PREFIX}/bin/llvm-nm"; do
        if [ -x "$candidate" ]; then
            NM="$candidate"
            break
        fi
    done
    echo "[build.sh] Using cross tools for Windows:"
    echo "[build.sh]   OBJDUMP=$OBJDUMP"
    echo "[build.sh]   NM=$NM"
fi

# ── 跨平台工具函数封装 ──

get_rpath() {
    local binary="$1"
    if [ "$IS_MACOS" -eq 1 ]; then
        "$OTOOL" -l "$binary" 2>/dev/null | grep -A2 LC_RPATH | grep path | awk '{print $2}' || echo "(no RPATH set)"
    elif [ "$IS_LINUX" -eq 1 ]; then
        patchelf --print-rpath "$binary" 2>/dev/null || echo "(no RPATH set)"
    else
        echo "(RPATH not applicable for Windows)"
    fi
}

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
        local old_rpaths
        old_rpaths=$("$OTOOL" -l "$binary" 2>/dev/null | grep -A2 LC_RPATH | grep path | awk '{print $2}' || true)
        for old_rp in $old_rpaths; do
            "$INSTALL_NAME_TOOL" -delete_rpath "$old_rp" "$binary" 2>/dev/null || true
        done
        local IFS=':'
        for rp in $new_rpath; do
            "$INSTALL_NAME_TOOL" -add_rpath "$rp" "$binary"
        done
        unset IFS
        local basename
        basename=$(basename "$binary")
        "$INSTALL_NAME_TOOL" -id "@rpath/$basename" "$binary" 2>/dev/null || true
    elif [ "$IS_LINUX" -eq 1 ]; then
        patchelf --set-rpath "$new_rpath" "$binary"
    else
        echo "[build.sh] Skipping RPATH setup for Windows PE binary: $binary"
    fi
}

check_symbol() {
    local binary="$1"
    local symbol="$2"
    if [ "$IS_WIN" -eq 1 ]; then
        "$NM" "$binary" 2>/dev/null | grep -iE " T _?$symbol$" >/dev/null
    elif [ "$IS_MACOS" -eq 1 ]; then
        "$NM" -gU "$binary" 2>/dev/null | grep -E " T _?$symbol$" >/dev/null
    else
        "$NM" -D "$binary" 2>/dev/null | grep -q " T $symbol$"
    fi
}

check_deps() {
    local binary="$1"
    if [ "$IS_MACOS" -eq 1 ]; then
        local bad_refs
        bad_refs=$("$OTOOL" -L "$binary" 2>/dev/null | grep -v "@rpath\|@loader_path\|/usr/lib\|/System/Library\|$(basename "$binary")" | grep -v "^$binary" | grep -v "^$(dirname "$binary")" | grep "/" || true)
        if [ -n "$bad_refs" ]; then
            echo "[build.sh] WARNING: Potentially non-portable references found:"
            echo "$bad_refs"
            return 0
        fi
        return 0
    elif [ "$IS_LINUX" -eq 1 ]; then
        if ldd "$binary" 2>&1 | grep -q "not found"; then
            ldd "$binary" | grep "not found"
            return 1
        fi
        return 0
    else
        echo "[build.sh] Skipping dependency check for Windows (PE format)"
        return 0
    fi
}

fix_dep_ref() {
    local binary="$1"
    local dep_name="$2"
    local new_ref="$3"
    if [ "$IS_MACOS" -eq 1 ]; then
        local old_ref
        old_ref=$("$OTOOL" -L "$binary" 2>/dev/null | grep "$dep_name" | head -1 | awk '{print $1}' || true)
        if [ -n "$old_ref" ] && [ "$old_ref" != "$new_ref" ] && [[ "$old_ref" != "@rpath/"* ]] && [[ "$old_ref" != "@loader_path/"* ]]; then
            echo "[build.sh] Fixing install name: $old_ref -> $new_ref"
            "$INSTALL_NAME_TOOL" -change "$old_ref" "$new_ref" "$binary"
        fi
    fi
}

verify_binary_arch() {
    local binary="$1"
    local expected_arch=""
    if [ "$IS_CROSS" -eq 1 ]; then
        case "${target_platform}" in
            osx-64) expected_arch="Mach-O 64-bit x86_64" ;;
            osx-arm64) expected_arch="Mach-O 64-bit arm64" ;;
            win-64) expected_arch="PE32+ executable" ;;
            linux-64) expected_arch="ELF 64-bit x86-64" ;;
        esac
        if command -v file &>/dev/null && [ -n "$expected_arch" ]; then
            echo "[build.sh] Verifying binary architecture: $binary"
            local file_out
            file_out=$(file "$binary" 2>/dev/null || true)
            echo "[build.sh] file output: $file_out"
            if echo "$file_out" | grep -q "$expected_arch"; then
                echo "[build.sh] Binary architecture verified: $expected_arch"
            else
                echo "[build.sh] WARNING: Expected $expected_arch but got: $file_out"
            fi
        fi
    fi
}

clean_editable_files() {
    if [ "$IS_CROSS" -eq 1 ]; then
        if [ -n "${SP_DIR:-}" ] && [ -d "$SP_DIR" ]; then
            find "$SP_DIR" -maxdepth 1 \( -name "_editable_*" -o -name "__editable__*" \) -type f -delete 2>/dev/null || true
            find "$SP_DIR" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | while read -r f; do
                if grep -q "xuanspace\|SpecWeave\|_skbuild" "$f" 2>/dev/null; then
                    rm -f "$f"
                fi
            done
        fi
    else
        for _sp in $($PYTHON_EXE -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null) "${SP_DIR:-}"; do
            if [ -n "$_sp" ] && [ -d "$_sp" ]; then
                find "$_sp" -maxdepth 1 \( -name "_editable_*" -o -name "__editable__*" \) -type f -delete 2>/dev/null || true
                find "$_sp" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | while read -r f; do
                    if grep -q "xuanspace\|SpecWeave\|_skbuild" "$f" 2>/dev/null; then
                        rm -f "$f"
                    fi
                done
            fi
        done
    fi
}

# ── 0. 源码预处理：CRLF 修复 + in-tree 构建残留清理 ──
echo "[build.sh] Preprocessing source directory..."

if command -v dos2unix &>/dev/null; then
    find "$SRC_DIR" -type f \( -name "*.sh" -o -name "*.py" -o -name "*.cmake" -o -name "CMakeLists.txt" \) -exec dos2unix {} \; 2>/dev/null || true
fi

rm -rf "${SRC_DIR}/build" "${SRC_DIR}/_skbuild" "${SRC_DIR}/dist" "${SRC_DIR}/*.egg-info" 2>/dev/null || true

# ── 1. 检测 tvm-ffi 来源：优先本地源码 pip 安装，回退 PyPI pip ──
TVM_FFI_SOURCE="none"
LOCAL_TVM_FFI_DIR=""

if [ -n "${CAFFE_FFI_TVM_FFI_DIR:-}" ] && [ -f "${CAFFE_FFI_TVM_FFI_DIR}/CMakeLists.txt" ]; then
    LOCAL_TVM_FFI_DIR="${CAFFE_FFI_TVM_FFI_DIR}"
    TVM_FFI_SOURCE="local-pip"
    echo "[build.sh] Using tvm-ffi from CAFFE_FFI_TVM_FFI_DIR: ${LOCAL_TVM_FFI_DIR}"
fi

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

if [ "$TVM_FFI_SOURCE" = "local-pip" ]; then
    echo "[build.sh] Installing local tvm-ffi via pip..."

    export SETUPTOOLS_SCM_PRETEND_VERSION="0.1.13"

    rm -rf "${LOCAL_TVM_FFI_DIR}/build" "${LOCAL_TVM_FFI_DIR}/_skbuild" "${LOCAL_TVM_FFI_DIR}/dist" 2>/dev/null || true
    find "${LOCAL_TVM_FFI_DIR}" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    find "${LOCAL_TVM_FFI_DIR}/python" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    rm -f "${LOCAL_TVM_FFI_DIR}/python/tvm_ffi/"*.so 2>/dev/null || true
    rm -f "${LOCAL_TVM_FFI_DIR}/python/tvm_ffi/"*.pyd 2>/dev/null || true
    rm -f "${LOCAL_TVM_FFI_DIR}/python/tvm_ffi/lib/"libtvm_ffi* 2>/dev/null || true

    _OLD_SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS:-}"
    unset SKBUILD_CMAKE_ARGS || true

    _OLD_CMAKE_ARGS="${CMAKE_ARGS:-}"
    _TVM_CMAKE_ARGS="-DTVM_FFI_USE_LIBBACKTRACE=OFF -DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF"
    if [ -n "$_OLD_CMAKE_ARGS" ]; then
        export CMAKE_ARGS="${_OLD_CMAKE_ARGS} ${_TVM_CMAKE_ARGS}"
    else
        export CMAKE_ARGS="${_TVM_CMAKE_ARGS}"
    fi

    $PYTHON_EXE -m pip install "${LOCAL_TVM_FFI_DIR}" --no-deps -vv --no-build-isolation || {
        echo "[build.sh] ERROR: Failed to install local tvm-ffi via pip"
        exit 1
    }

    clean_editable_files
    echo "[build.sh] Cleaned tvm-ffi editable finder files"

    if [ -n "$_OLD_SKBUILD_CMAKE_ARGS" ]; then
        export SKBUILD_CMAKE_ARGS="$_OLD_SKBUILD_CMAKE_ARGS"
    else
        unset SKBUILD_CMAKE_ARGS || true
    fi

    if [ -n "$_OLD_CMAKE_ARGS" ]; then
        export CMAKE_ARGS="$_OLD_CMAKE_ARGS"
    else
        unset CMAKE_ARGS || true
    fi

    if [ "$IS_CROSS" -eq 0 ]; then
        echo "[build.sh] Verifying tvm_ffi import..."
        TVM_FFI_BASE=$($PYTHON_EXE -c "import tvm_ffi; import os; print(os.path.dirname(tvm_ffi.__file__))")
        echo "[build.sh] tvm_ffi installed at: $TVM_FFI_BASE"
    else
        echo "[build.sh] Cross-compile: skipping tvm_ffi Python import verification (host Python cannot run target binaries)"
        TVM_FFI_BASE="${SP_DIR}/tvm_ffi"
        echo "[build.sh] Assuming tvm_ffi installed at: $TVM_FFI_BASE"
    fi

    TVM_FFI_LIB_CANDIDATES=(
        "${TVM_FFI_BASE}/lib/libtvm_ffi.so"
        "${TVM_FFI_BASE}/lib/libtvm_ffi.dylib"
        "${TVM_FFI_BASE}/lib/tvm_ffi.dll"
        "${TVM_FFI_BASE}/libtvm_ffi.so"
        "${TVM_FFI_BASE}/libtvm_ffi.dylib"
        "${TVM_FFI_BASE}/tvm_ffi.dll"
    )
    TVM_FFI_LIB=""
    for cand in "${TVM_FFI_LIB_CANDIDATES[@]}"; do
        if [ -f "$cand" ]; then
            TVM_FFI_LIB="$cand"
            break
        fi
    done

    if [ -z "$TVM_FFI_LIB" ]; then
        echo "[build.sh] Searching for libtvm_ffi in $SP_DIR..."
        TVM_FFI_LIB=$(find "${SP_DIR}" -name "libtvm_ffi*" -type f 2>/dev/null | head -1 || true)
    fi

    if [ -z "$TVM_FFI_LIB" ] || [ ! -f "$TVM_FFI_LIB" ]; then
        echo "[build.sh] WARNING: Could not locate libtvm_ffi shared library after install"
        echo "[build.sh] Contents of $SP_DIR/tvm_ffi:"
        ls -laR "${SP_DIR}/tvm_ffi" 2>/dev/null || true
    else
        echo "[build.sh] Found libtvm_ffi: $TVM_FFI_LIB"
        verify_binary_arch "$TVM_FFI_LIB"
        echo "[build.sh] Checking for TVMFFIGetCustomAllocator symbol..."
        if check_symbol "$TVM_FFI_LIB" "TVMFFIGetCustomAllocator"; then
            echo "[build.sh] TVMFFIGetCustomAllocator symbol verified"
        else
            echo "[build.sh] WARNING: TVMFFIGetCustomAllocator symbol not found"
            "$NM" "$TVM_FFI_LIB" 2>/dev/null | grep -i TVMFFIGetCustomAllocator || true
        fi
    fi
fi

if [ "$TVM_FFI_SOURCE" = "none" ]; then
    echo "[build.sh] No local tvm-ffi found, installing apache-tvm-ffi via pip..."
    $PYTHON_EXE -m pip install --no-deps 'apache-tvm-ffi' || {
        echo "[build.sh] ERROR: Failed to install apache-tvm-ffi via pip"
        exit 1
    }
    TVM_FFI_SOURCE="pip"
    echo "[build.sh] Installed tvm-ffi via pip (PyPI)"

    if [ "$IS_CROSS" -eq 0 ]; then
        TVM_FFI_BASE=$($PYTHON_EXE -c "import tvm_ffi; import os; print(os.path.dirname(tvm_ffi.__file__))")
        echo "[build.sh] tvm_ffi installed at: $TVM_FFI_BASE"
    else
        echo "[build.sh] Cross-compile: skipping tvm_ffi import verification"
    fi
fi

# ── 2. 配置 CMake 参数 ──
if [ "$IS_MACOS" -eq 1 ]; then
    RPATH_PREFIX="@loader_path"
elif [ "$IS_LINUX" -eq 1 ]; then
    RPATH_PREFIX='$ORIGIN'
else
    RPATH_PREFIX=""
fi

_CAFFE_OLD_CMAKE_ARGS="${CMAKE_ARGS:-}"

_CAFFE_RPATH=""
_TVM_RPATH=""
if [ -n "$RPATH_PREFIX" ]; then
    _CAFFE_RPATH="${RPATH_PREFIX}:${RPATH_PREFIX}/lib:${RPATH_PREFIX}/../tvm_ffi/lib:${RPATH_PREFIX}/../../.."
    _TVM_RPATH="${RPATH_PREFIX}:${RPATH_PREFIX}/..:${RPATH_PREFIX}/../../../../lib"
fi

SKBUILD_EXTRA_ARGS=""
if [ -n "$_CAFFE_RPATH" ]; then
    SKBUILD_EXTRA_ARGS="-DCMAKE_INSTALL_RPATH=${_CAFFE_RPATH}"
fi

export SKBUILD_CMAKE_ARGS="\
-DCMAKE_PREFIX_PATH=${PREFIX} \
${SKBUILD_EXTRA_ARGS} \
-DCAFFE_FFI_PREFER_SYSTEM_TVM_FFI=ON \
-DTVM_FFI_USE_LIBBACKTRACE=OFF \
-DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF"

if [ -n "$_CAFFE_OLD_CMAKE_ARGS" ]; then
    export CMAKE_ARGS="$_CAFFE_OLD_CMAKE_ARGS"
else
    unset CMAKE_ARGS || true
fi

echo "[build.sh] ============================================================"
echo "[build.sh] CMake args (SKBUILD_CMAKE_ARGS): $SKBUILD_CMAKE_ARGS"
if [ -n "$RPATH_PREFIX" ]; then
    echo "[build.sh] RPATH prefix: $RPATH_PREFIX"
else
    echo "[build.sh] RPATH: skipped (Windows target)"
fi
echo "[build.sh] --- conda-build environment variables ---"
echo "[build.sh] IS_CROSS:       $IS_CROSS"
echo "[build.sh] TARGET_OS:      $TARGET_OS"
echo "[build.sh] PREFIX:         $PREFIX"
echo "[build.sh] HOST_PREFIX:    ${HOST_PREFIX:-not set}"
echo "[build.sh] BUILD_PREFIX:   ${BUILD_PREFIX:-not set}"
echo "[build.sh] SP_DIR:         ${SP_DIR:-unknown}"
echo "[build.sh] SRC_DIR:        $SRC_DIR"
echo "[build.sh] Python:         $PYTHON ($($PYTHON_EXE --version 2>&1))"
echo "[build.sh] CPU_COUNT:      ${CPU_COUNT:-4}"
echo "[build.sh] tvm-ffi src:    $TVM_FFI_SOURCE (${LOCAL_TVM_FFI_DIR:-PyPI wheel})"
echo "[build.sh] Build platform: $UNAME_S"
echo "[build.sh] ============================================================"

# ── 3. 构建并安装 ──
rm -rf build _skbuild dist *.egg-info 2>/dev/null || true
rm -f python/caffe_ffi/*.so python/caffe_ffi/*.pyd python/caffe_ffi/lib/*.so python/caffe_ffi/lib/*.pyd python/caffe_ffi/lib/*.dll 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

$PYTHON_EXE -m pip install . --no-deps -vv --no-build-isolation

_n_editable=0
if [ -n "${SP_DIR:-}" ] && [ -d "$SP_DIR" ]; then
    _n=$(find "$SP_DIR" -maxdepth 1 \( -name "_editable_*" -o -name "__editable__*" \) -type f 2>/dev/null | wc -l)
    _n_editable=$((_n_editable + _n))
fi
if [ "$IS_CROSS" -eq 0 ]; then
    for _sp in $($PYTHON_EXE -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null); do
        if [ -n "$_sp" ] && [ -d "$_sp" ]; then
            _n=$(find "$_sp" -maxdepth 1 \( -name "_editable_*" -o -name "__editable__*" \) -type f 2>/dev/null | wc -l)
            _n_editable=$((_n_editable + _n))
        fi
    done
fi
if [ "$_n_editable" -gt 0 ]; then
    echo "[build.sh] Found $_n_editable editable finder files, removing..."
    clean_editable_files
    echo "[build.sh] Editable files cleaned"
else
    echo "[build.sh] No editable finder files found (good)"
fi

# ── 4. 构建后验证与修复 ──
echo "[build.sh] ============================================================"
echo "[build.sh] Post-build: verifying shared library..."

CAFFE_FFI_LIB_DIR=""
CAFFE_FFI_SO=""

if [ -n "${SP_DIR:-}" ] && [ -d "${SP_DIR}/caffe_ffi" ]; then
    CAFFE_FFI_SO=$(find "${SP_DIR}/caffe_ffi" -name "_caffe_ffi*" -type f \( -name "*.so" -o -name "*.pyd" -o -name "*.dylib" \) 2>/dev/null | head -1 || true)
fi

if [ -z "$CAFFE_FFI_SO" ]; then
    CAFFE_FFI_SO=$(find "${PREFIX}" -name "_caffe_ffi*" -type f \( -name "*.so" -o -name "*.pyd" -o -name "*.dylib" \) 2>/dev/null | head -1 || true)
fi

if [ -z "$CAFFE_FFI_SO" ]; then
    echo "[build.sh] ERROR: Cannot find _caffe_ffi shared library after build"
    echo "[build.sh] PREFIX contents (lib):"
    ls -la "${PREFIX}/lib/" 2>/dev/null || true
    echo "[build.sh] SP_DIR contents:"
    find "${SP_DIR:-$PREFIX}" -name "*caffe_ffi*" -type f 2>/dev/null | head -20 || true
    exit 1
fi

CAFFE_FFI_LIB_DIR=$(dirname "$CAFFE_FFI_SO")
echo "[build.sh] Found _caffe_ffi: $CAFFE_FFI_SO"
echo "[build.sh] _caffe_ffi directory: $CAFFE_FFI_LIB_DIR"

verify_binary_arch "$CAFFE_FFI_SO"

if [ "$IS_CROSS" -eq 0 ]; then
    echo "[build.sh] Verifying tvm_ffi package..."
    TVM_FFI_FILE=$($PYTHON_EXE -c "import tvm_ffi; print(tvm_ffi.__file__)")
    echo "[build.sh] tvm_ffi package location: $TVM_FFI_FILE"
    TVM_FFI_LIB=$($PYTHON_EXE -c "import tvm_ffi, os; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'lib', 'libtvm_ffi.so'))")
else
    echo "[build.sh] Cross-compile: locating tvm_ffi lib via filesystem..."
    TVM_FFI_LIB=""
    for cand in \
        "${SP_DIR}/tvm_ffi/lib/libtvm_ffi.so" \
        "${SP_DIR}/tvm_ffi/lib/libtvm_ffi.dylib" \
        "${SP_DIR}/tvm_ffi/lib/tvm_ffi.dll" \
        "${SP_DIR}/tvm_ffi/libtvm_ffi.so" \
        "${SP_DIR}/tvm_ffi/libtvm_ffi.dylib" \
        "${SP_DIR}/tvm_ffi/tvm_ffi.dll"; do
        if [ -f "$cand" ]; then
            TVM_FFI_LIB="$cand"
            break
        fi
    done
    if [ -z "$TVM_FFI_LIB" ]; then
        TVM_FFI_LIB=$(find "${SP_DIR}/tvm_ffi" -name "libtvm_ffi*" -o -name "tvm_ffi.dll" 2>/dev/null | head -1 || true)
    fi
fi

if [ -n "$TVM_FFI_LIB" ] && [ -f "$TVM_FFI_LIB" ]; then
    echo "[build.sh] libtvm_ffi found at: $TVM_FFI_LIB"
    ls -la "$TVM_FFI_LIB"
    verify_binary_arch "$TVM_FFI_LIB"
    echo "[build.sh] Verifying TVMFFIGetCustomAllocator symbol..."
    if check_symbol "$TVM_FFI_LIB" "TVMFFIGetCustomAllocator"; then
        echo "[build.sh] TVMFFIGetCustomAllocator symbol present"
    else
        echo "[build.sh] WARNING: TVMFFIGetCustomAllocator symbol not found"
        "$NM" "$TVM_FFI_LIB" 2>/dev/null | grep -i TVMFFI || true
    fi
else
    echo "[build.sh] WARNING: libtvm_ffi not found for post-build verification"
fi

if [ "$IS_WIN" -eq 0 ]; then
    echo "[build.sh] Current RPATH of _caffe_ffi: $(get_rpath "$CAFFE_FFI_SO")"
    if [ -n "$RPATH_PREFIX" ]; then
        set_rpath "$CAFFE_FFI_SO" \
            "${RPATH_PREFIX}" \
            "${RPATH_PREFIX}/lib" \
            "${RPATH_PREFIX}/../tvm_ffi/lib" \
            "${RPATH_PREFIX}/../../.."
        echo "[build.sh] Set _caffe_ffi RPATH to platform-relative paths"
        echo "[build.sh] New RPATH: $(get_rpath "$CAFFE_FFI_SO")"

        if [ -n "${TVM_FFI_LIB:-}" ] && [ -f "$TVM_FFI_LIB" ]; then
            echo "[build.sh] Current RPATH of libtvm_ffi: $(get_rpath "$TVM_FFI_LIB")"
            set_rpath "$TVM_FFI_LIB" \
                "${RPATH_PREFIX}" \
                "${RPATH_PREFIX}/.." \
                "${RPATH_PREFIX}/../../../../lib"
            echo "[build.sh] Set libtvm_ffi RPATH to platform-relative paths"
            echo "[build.sh] New libtvm_ffi RPATH: $(get_rpath "$TVM_FFI_LIB")"

            if [ "$IS_MACOS" -eq 1 ]; then
                fix_dep_ref "$CAFFE_FFI_SO" "libtvm_ffi" "@rpath/libtvm_ffi.dylib"
            fi
        fi
    fi
else
    echo "[build.sh] Windows target: skipping RPATH setup (PE/DLL format)"
fi

echo "[build.sh] ============================================================"
echo "[build.sh] Dependency check:"
if [ "$IS_MACOS" -eq 1 ]; then
    "$OTOOL" -L "$CAFFE_FFI_SO" || true
    check_deps "$CAFFE_FFI_SO"
elif [ "$IS_LINUX" -eq 1 ]; then
    ldd "$CAFFE_FFI_SO" || true
    if ! check_deps "$CAFFE_FFI_SO"; then
        echo "[build.sh] WARNING: Some shared library dependencies are unresolved (may be fixed by conda-build post-processing)"
    else
        echo "[build.sh] All shared library dependencies resolved successfully"
    fi
else
    echo "[build.sh] Windows target: listing DLL imports with llvm-objdump:"
    "$OBJDUMP" -p "$CAFFE_FFI_SO" 2>/dev/null | grep -i "DLL Name" || true
fi

echo "[build.sh] ============================================================"
echo "[build.sh] Build completed successfully!"
echo "[build.sh] _caffe_ffi location: $CAFFE_FFI_SO"
echo "[build.sh] libtvm_ffi location: ${TVM_FFI_LIB:-not found}"
if [ "$IS_CROSS" -eq 1 ]; then
    echo "[build.sh] Cross-compilation to ${target_platform} complete."
fi
echo "[build.sh] ============================================================"
