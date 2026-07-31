#!/bin/bash
set -eux -o pipefail

PROJECT_DIR="/mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi"
cd "$PROJECT_DIR"

echo "=== Step 1: Activate conda ==="
source /opt/conda/etc/profile.d/conda.sh
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