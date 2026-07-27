#!/bin/bash
# 设置Conda开发环境
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)/vendor"

ENV_NAME=${1:-npu-ffi-dev}

echo "========================================"
echo "Setting up npu-ffi Conda dev environment"
echo "========================================"
echo "Environment name: $ENV_NAME"
echo "Project root: $PROJECT_ROOT"
echo "Vendor root: $VENDOR_ROOT"

cd "$PROJECT_ROOT"

echo ""
echo "[1/4] Creating conda environment..."
conda env create -f environment.yml -n "$ENV_NAME" --force

echo ""
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ""
echo "[3/4] Installing tvm-ffi (editable)..."
if [ -d "$VENDOR_ROOT/tvm-ffi" ]; then
    echo "Found local tvm-ffi at $VENDOR_ROOT/tvm-ffi"
    pip install --no-build-isolation -e "$VENDOR_ROOT/tvm-ffi"
else
    echo "Local tvm-ffi not found, installing from PyPI..."
    pip install apache-tvm-ffi
fi

echo ""
echo "[4/4] Installing npu-ffi (editable)..."
pip install --no-build-isolation -e .

echo ""
echo "========================================"
echo "Development environment setup complete!"
echo "========================================"
echo ""
echo "Activate with: conda activate $ENV_NAME"
echo ""
echo "Run tests with:"
echo "  export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/macOS"
echo "  pytest tests/python -v"
echo ""
