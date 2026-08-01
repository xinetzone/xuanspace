#!/bin/bash
set -e
LOG=/tmp/cow_test_output.log
exec > >(tee -a "$LOG") 2>&1

echo "=== START $(date) ==="

source /opt/conda/etc/profile.d/conda.sh
conda activate caffe-ffi
cd /SpecWeave/projects/xuanspace/libs/caffe-ffi

echo "=== Step 1: CMAKE CONFIGURE ==="
rm -rf build_cpp
mkdir -p build_cpp
cd build_cpp
cmake .. -G Ninja -DCAFFE_FFI_ENABLE_COW=ON -DCAFFE_FFI_ENABLE_COW_PHASE3=ON -DCAFFE_FFI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

echo ""
echo "=== Step 2: BUILD ==="
ninja caffe_ffi_tests -j$(nproc)

echo ""
echo "=== Step 3: FIND TEST BINARY ==="
TEST_BIN=$(find . -name 'caffe_ffi_tests' -type f)
echo "Test binary: $TEST_BIN"

echo ""
echo "=== Step 4: RUN COW TESTS ==="
$TEST_BIN --gtest_filter='*COW*:*ZeroCopy*:*OwnerCOW*:*TwoWay*:*ShareData*:*SplitBackward*'

echo ""
echo "=== Step 5: RUN ALL TESTS ==="
$TEST_BIN

echo ""
echo "=== DONE $(date) ==="
