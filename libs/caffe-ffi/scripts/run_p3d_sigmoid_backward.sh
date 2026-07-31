#!/bin/bash
# run_p3d_sigmoid_backward.sh
# WSL编译运行脚本：编译caffe-ffi并执行P3-D Sigmoid反向传播测试
# 用法: bash scripts/run_p3d_sigmoid_backward.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CAFFE_FFI_DIR="$PROJECT_DIR/libs/caffe-ffi"
BUILD_DIR="$CAFFE_FFI_DIR/build"
LOG_DIR="$CAFFE_FFI_DIR/test_logs"
mkdir -p "$LOG_DIR"

echo "=== P3-D Sigmoid Backward Test Runner ==="
echo "Project dir: $PROJECT_DIR"
echo "Caffe-ffi dir: $CAFFE_FFI_DIR"
echo "Build dir: $BUILD_DIR"
echo ""

# ── Step 1: 环境检查 ──
echo "[1/5] Environment check..."
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Please install cmake."
    exit 1
fi
if ! command -v ninja &>/dev/null; then
    echo "WARNING: ninja not found, falling back to make."
    GENERATOR="Unix Makefiles"
else
    GENERATOR="Ninja"
fi
echo "  Generator: $GENERATOR"
echo "  Python: $(python3 --version 2>&1)"
echo "  CMake: $(cmake --version | head -1)"

# ── Step 2: CMake 配置 ──
echo ""
echo "[2/5] CMake configure..."
cd "$CAFFE_FFI_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$LOG_DIR/cmake_config.log"
echo "  Configure done."

# ── Step 3: 编译 ──
echo ""
echo "[3/5] Building..."
if [ "$GENERATOR" = "Ninja" ]; then
    ninja -j"$(nproc)" 2>&1 | tee "$LOG_DIR/build.log"
else
    make -j"$(nproc)" 2>&1 | tee "$LOG_DIR/build.log"
fi
echo "  Build done."

# ── Step 4: 安装Python包（editable模式） ──
echo ""
echo "[4/5] Installing caffe-ffi Python package (editable)..."
cd "$CAFFE_FFI_DIR"
pip install -e . 2>&1 | tee "$LOG_DIR/pip_install.log"
echo "  Install done."

# ── Step 5: 运行P3-D Sigmoid反向传播测试 ──
echo ""
echo "[5/5] Running P3-D Sigmoid Backward tests..."
cd "$CAFFE_FFI_DIR"

# 设置日志级别为INFO以捕获[ACTIVATION-PERF]日志
export CAFFE_FFI_LOG_LEVEL=2  # INFO level

TEST_FILE="tests/python/test_p3c_activations_ip.py"
TEST_CLASS="TestSigmoidBackward"
LOG_FILE="$LOG_DIR/p3d_sigmoid_backward_$(date +%Y%m%d_%H%M%S).log"

echo "  Test file: $TEST_FILE"
echo "  Test class: $TEST_CLASS"
echo "  Log file: $LOG_FILE"
echo ""

python3 -m pytest "$TEST_FILE" -v -s -k "$TEST_CLASS" 2>&1 | tee "$LOG_FILE"

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== Test Results Summary ==="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "PASS: All Sigmoid backward tests passed!"
else
    echo "FAIL: Some tests failed (exit code $TEST_EXIT_CODE)"
fi

# ── Step 6: 分析[ACTIVATION-PERF]日志中的饱和率 ──
echo ""
echo "=== Saturation Analysis (from [ACTIVATION-PERF] backward logs) ==="
echo "Searching for backward performance logs..."
BACKWARD_LOGS=$(grep -E "\[ACTIVATION-PERF\].*Sigmoid backward" "$LOG_FILE" 2>/dev/null || echo "No backward perf logs found (log level may be too high)")
echo "$BACKWARD_LOGS"

# Extract saturate ratios
echo ""
echo "Saturation ratios detected:"
echo "$BACKWARD_LOGS" | grep -oP 'saturate=\K[0-9]+/[0-9]+ \([0-9.]+\)' || echo "  (none - check log level)"

# Check for anomalies (saturation > 50%)
echo ""
echo "Anomaly detection (saturate ratio > 50%):"
echo "$BACKWARD_LOGS" | while IFS= read -r line; do
    ratio=$(echo "$line" | grep -oP '\(\K[0-9.]+' | head -1 || true)
    if [ -n "$ratio" ]; then
        is_high=$(python3 -c "print(1 if $ratio > 0.5 else 0)" 2>/dev/null || echo "0")
        if [ "$is_high" = "1" ]; then
            echo "  WARNING: High saturation: $line"
        fi
    fi
done

echo ""
echo "=== Full logs saved to: $LOG_FILE ==="
echo "=== Done ==="
exit $TEST_EXIT_CODE
