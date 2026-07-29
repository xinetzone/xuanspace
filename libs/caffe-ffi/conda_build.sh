#!/bin/bash
# caffe-ffi Conda 环境 Linux/macOS 构建脚本
# 使用方法：conda activate caffe-ffi && bash conda_build.sh

set -euo pipefail

echo "========================================"
echo " caffe-ffi Conda Build (Linux/macOS)"
echo "========================================"

# ── 检查 conda 环境 ──
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda 未找到，请先安装 Miniconda/Anaconda"
    exit 1
fi

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "[WARN] 未检测到激活的 conda 环境，尝试激活 caffe-ffi..."
    eval "$(conda shell.bash hook)"
    conda activate caffe-ffi || {
        echo "[ERROR] 无法激活 caffe-ffi 环境，请先运行: conda env create -f environment.yml"
        exit 1
    }
fi

echo "[INFO] Conda 环境: $CONDA_DEFAULT_ENV"
echo "[INFO] Python: $(python --version)"

# ── 检测 CPU 线程数 ──
if command -v nproc &> /dev/null; then
    NPROC=$(nproc)
else
    NPROC=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

# ── 创建构建目录 ──
mkdir -p build

# ── 配置 CMake ──
echo ""
echo "[STEP 1/3] CMake Configure..."
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DProtobuf_DIR="$CONDA_PREFIX/lib/cmake/protobuf" \
    -DBLAS_HOME="$CONDA_PREFIX"

# ── 编译 ──
echo ""
echo "[STEP 2/3] Building (Ninja, -j${NPROC})..."
cmake --build build --config Release -j"${NPROC}"

# ── 安装 Python 包（editable模式）──
echo ""
echo "[STEP 3/3] Installing Python package (editable)..."
pip install -e . --no-build-isolation

# ── 运行测试 ──
echo ""
echo "[TEST] Running pytest..."
python -m pytest tests/python/ -v --tb=short

echo ""
echo "========================================"
echo " Build complete!"
echo " Shared library: build/_caffe_ffi.$(uname -s | tr '[:upper:]' '[:lower:]')"
echo "========================================"
