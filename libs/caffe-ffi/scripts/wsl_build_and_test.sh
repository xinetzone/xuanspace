#!/bin/bash
# WSL build & test script for Phase 3.0/3.1
#
# IMPORTANT: This script must be run from INSIDE a WSL terminal where conda is
# already initialized. Running it as `bash script.sh` from a Windows PowerShell
# starts a non-interactive bash that will NOT have conda available.
#
# Correct usage (in WSL terminal):
#   cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi
#   bash scripts/wsl_build_and_test.sh
#
# If conda is not found, first run `conda init bash` in your WSL terminal.
set -eux -o pipefail

PROJECT_DIR="/mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi"
cd "$PROJECT_DIR"

echo "=== Step 1: Activate conda ==="
# Source user's bashrc to pick up conda init (works in interactive WSL shell)
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
# Fallback: try common install paths
if ! hash conda 2>/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda" "/opt/miniconda3"; do
    [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; echo "Conda: $d"; break; }
  done
fi
hash conda 2>/dev/null || { echo "ERROR: conda not found. Run this script from inside a WSL terminal where conda is initialized."; exit 1; }
conda activate caffe-ffi

echo "=== Step 2: Configure CMake (Phase 3 enabled) ==="
cmake --preset default \
  -DCAFFE_FFI_ENABLE_COW=ON \
  -DCAFFE_FFI_ENABLE_COW_PHASE3=ON

echo "=== Step 3: Build ==="
cmake --build --preset default -j$(nproc)

echo "=== Step 4: Install ==="
pip install --no-build-isolation -e .

echo "=== Step 5: Run FFI test ==="
pytest tests/python/test_ffi_set_shape_only.py -v -s

echo "=== DONE ==="