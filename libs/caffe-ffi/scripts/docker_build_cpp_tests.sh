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
echo "=== Step 4: Run all C++ tests (242 tests, ~60ms) ==="
# Note: caffe-ffi uses a custom header-only test framework (not gtest).
# It accepts a single substring filter as positional argument (no --gtest_filter).
# Running full suite since all 242 tests complete in ~60ms total.
docker exec -w "$SRC/build_cpp" "$CONTAINER" bash -c "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:. && $TEST_BIN 2>&1"

echo ""
echo "=== Step 5: Run COW/ZeroCopy related tests specifically ==="
for pattern in "COW" "ZeroCopy" "OwnerCOW" "ShareData" "SplitBackward"; do
    echo ""
    echo "--- Filter: $pattern ---"
    docker exec -w "$SRC/build_cpp" "$CONTAINER" bash -c "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:. && $TEST_BIN $pattern 2>&1" | tail -1
done

echo ""
echo "=== DONE ==="
