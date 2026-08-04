#!/bin/bash
# S5 Dropout 推理模式 COW 零拷贝优化：重建 + Dropout/COW 测试 + 全量回归
set -eux -o pipefail

SRC="/SpecWeave/projects/xuanspace/libs/caffe-ffi"
cd "$SRC"
source /opt/conda/etc/profile.d/conda.sh
conda activate caffe-ffi
export PATH="/opt/conda/envs/caffe-ffi/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=4

echo "=== Step 1: clean rebuild ==="
rm -rf build
SKBUILD_CMAKE_ARGS="-DCAFFE_FFI_ENABLE_COW=ON;-DCAFFE_FFI_ENABLE_COW_PHASE3=ON;-DCAFFE_FFI_BUILD_TESTS=OFF" \
    pip install --no-build-isolation -v -e . 2>&1 | tail -30 || { echo "REBUILD FAILED"; exit 1; }

echo "=== Step 2: sync freshly built .so into python package ==="
cp build/python/caffe_ffi/_caffe_ffi.so python/caffe_ffi/_caffe_ffi.so

echo "=== Step 3: import check ==="
python -c "import caffe_ffi; print('caffe_ffi', caffe_ffi.__version__)"

echo "=== Step 4: Dropout COW + backward + train-mode tests ==="
export CAFFE_FFI_ENABLE_COW=1 CAFFE_FFI_ENABLE_COW_PHASE3=1
pytest tests/python/test_cow.py -v 2>&1 | tail -60 || { echo "COW TESTS FAILED"; exit 1; }
pytest tests/python/test_dropout_backward.py -v 2>&1 | tail -40 || { echo "DROPOUT TESTS FAILED"; exit 1; }

echo "=== Step 5: full regression ==="
pytest tests/python -q 2>&1 | tail -15 || { echo "FULL REGRESSION FAILED"; exit 1; }

echo "========================================="
echo "  S5 DROPOUT COW REBUILD + REGRESSION PASSED"
echo "========================================="