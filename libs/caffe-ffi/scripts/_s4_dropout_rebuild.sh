#!/bin/bash
# S4 TS32-D1 Dropout 训练模式：重建 + Dropout 测试 + 全量回归
set -eux -o pipefail

SRC="/SpecWeave/projects/xuanspace/libs/caffe-ffi"
cd "$SRC"
source /opt/conda/etc/profile.d/conda.sh
conda activate caffe-ffi
# Force the conda env bin ahead of any login-shell PATH (e.g. /opt/venv) so
# pip/python/cmake resolve to the caffe-ffi env even when invoked via docker exec -lc.
export PATH="/opt/conda/envs/caffe-ffi/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=4

echo "=== Step 1: clean rebuild ==="
rm -rf build
SKBUILD_CMAKE_ARGS="-DCAFFE_FFI_ENABLE_COW=ON;-DCAFFE_FFI_ENABLE_COW_PHASE3=ON;-DCAFFE_FFI_BUILD_TESTS=OFF" \
    pip install --no-build-isolation -v -e . 2>&1 | tail -30 || { echo "REBUILD FAILED"; exit 1; }

echo "=== Step 2: sync freshly built .so into python package ==="
# Editable install may leave a stale .so in python/caffe_ffi/; copy the newly
# compiled one so the new FFI bindings (set_train_mode / is_train) are loaded.
cp build/python/caffe_ffi/_caffe_ffi.so python/caffe_ffi/_caffe_ffi.so

echo "=== Step 3: import check ==="
python -c "import caffe_ffi; print('caffe_ffi', caffe_ffi.__version__)"

echo "=== Step 4: Dropout train-mode tests ==="
export CAFFE_FFI_ENABLE_COW=1 CAFFE_FFI_ENABLE_COW_PHASE3=1
pytest tests/python/test_dropout_backward.py -v 2>&1 | tail -50 || { echo "DROPOUT TESTS FAILED"; exit 1; }

echo "=== Step 5: full regression ==="
pytest tests/python -q 2>&1 | tail -15 || { echo "FULL REGRESSION FAILED"; exit 1; }

echo "========================================="
echo "  S4 DROPOUT REBUILD + REGRESSION PASSED"
echo "========================================="