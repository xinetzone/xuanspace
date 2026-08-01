#!/bin/bash
set -eux -o pipefail

CONTAINER="caffe-ffi-jupyter"
SRC="/SpecWeave/projects/xuanspace/libs/caffe-ffi"

# Helper: run with conda activated
run_conda() {
    docker exec -w "$SRC" "$CONTAINER" bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && $*"
}

echo "=== Step 1: CMake configure (Ninja) ==="
run_conda "rm -rf build_cpp && mkdir -p build_cpp"
docker exec -w "$SRC/build_cpp" "$CONTAINER" bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && cmake .. -G Ninja -DCAFFE_FFI_ENABLE_COW=ON -DCAFFE_FFI_ENABLE_COW_PHASE3=ON -DCAFFE_FFI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release 2>&1" | tail -30

echo ""
echo "=== Step 2: Build caffe_ffi_tests ==="
run_conda "cd build_cpp && ninja caffe_ffi_tests 2>&1" | tail -40

echo ""
echo "=== Step 3: Find test binary ==="
TEST_BIN=$(docker exec "$CONTAINER" bash -c "find $SRC/build_cpp -name 'caffe_ffi_tests' -type f 2>/dev/null | head -1")
echo "Test binary: $TEST_BIN"

echo ""
echo "=== Step 4: Run COW/ZeroCopy/OwnerCOW tests ==="
docker exec "$CONTAINER" bash -c "$TEST_BIN --gtest_filter='*COW*:*ZeroCopy*:*OwnerCOW*:*TwoWay*:*ShareData*:*SplitBackward*' 2>&1"

echo ""
echo "=== DONE ==="
