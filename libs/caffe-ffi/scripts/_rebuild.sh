#!/bin/bash
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
source /opt/conda/etc/profile.d/conda.sh
conda activate caffe-ffi
SRC=/SpecWeave/projects/xuanspace/libs/caffe-ffi
cd "$SRC"
rm -rf "$SRC/build"
TVM_FFI_DIR=$(python -c "import tvm_ffi, os; print(os.path.dirname(tvm_ffi.__file__))")
SKBUILD_CMAKE_ARGS="-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON;-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON;-DCMAKE_SKIP_BUILD_RPATH=OFF;-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON;-DCAFFE_FFI_ENABLE_BACKTRACE=OFF;-Dcaffe-ffi_DIR=$TVM_FFI_DIR;-DCMAKE_PREFIX_PATH=$CONDA_PREFIX;-DCAFFE_FFI_BUILD_TESTS=OFF;-DCAFFE_FFI_ENABLE_COW_PHASE3=ON" \
  pip install --no-cache-dir --no-build-isolation -e "$SRC" 2>&1 | tail -8
echo "=== copy fresh .so to source dir ==="
cp "$SRC/build/python/caffe_ffi/_caffe_ffi.so" "$SRC/python/caffe_ffi/_caffe_ffi.so"
ls -la "$SRC/python/caffe_ffi/_caffe_ffi.so"
echo "=== rebuild done ==="