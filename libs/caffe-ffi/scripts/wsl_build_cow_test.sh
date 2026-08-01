#!/bin/bash
# WSL script: Build caffe-ffi and run COW/ZeroCopy related C++ tests
#
# This script builds caffe-ffi with COW enabled and runs the C++ test suite.
# It is designed to verify the COW threshold fix (use_count > 2) which ensures:
#   - N=1 two-party sharing: in-place passthrough (no COW copy)
#   - N>=2 multi-party sharing: COW clone triggers for write isolation
#
# IMPORTANT: Must be run from INSIDE a WSL terminal where conda is initialized.
#
# Usage (in WSL terminal):
#   cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi
#   bash scripts/wsl_build_cow_test.sh [options]
#
# Options:
#   --rebuild     Clean build directory before compiling (full rebuild)
#   --python      Also run Python COW tests (test_cow.py)
#   --filter F    Run only tests matching substring F (e.g. "ShareDataMutation")
#   --suite S     Run only test suite S (exact suite name match via Python helper)
#   -h, --help    Show this help
#
# Examples:
#   bash scripts/wsl_build_cow_test.sh                     # build + run all C++ tests
#   bash scripts/wsl_build_cow_test.sh --rebuild           # clean rebuild + all tests
#   bash scripts/wsl_build_cow_test.sh --filter ShareDataMutationVisibleToBoth
#   bash scripts/wsl_build_cow_test.sh --suite SliceLayerZeroCopyTest
#   bash scripts/wsl_build_cow_test.sh --suite OwnerCOWTest
#   bash scripts/wsl_build_cow_test.sh --python            # also run Python tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step()  { echo -e "${CYAN}==>${NC} ${BOLD}$1${NC}"; }
pass()  { echo -e "    ${GREEN}[PASS]${NC} $1"; }
fail()  { echo -e "    ${RED}[FAIL]${NC} $1"; }
warn()  { echo -e "    ${YELLOW}[WARN]${NC} $1"; }
info()  { echo -e "    $1"; }

# ── Parse arguments ──
REBUILD=0
RUN_PYTHON=0
FILTER=""
SUITE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild)  REBUILD=1; shift ;;
    --python)   RUN_PYTHON=1; shift ;;
    --filter)   FILTER="$2"; shift 2 ;;
    --suite)    SUITE="$2"; shift 2 ;;
    -h|--help)
      head -n 30 "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1"; echo "Use -h for help."; exit 1 ;;
  esac
done

cd "$PROJECT_DIR"

echo ""
echo -e "${CYAN}${BOLD}========================================${NC}"
echo -e "${CYAN}${BOLD}  caffe-ffi COW Fix Build & Test (WSL)${NC}"
echo -e "${CYAN}${BOLD}========================================${NC}"
echo "Project dir: $PROJECT_DIR"
echo "Build dir:   $BUILD_DIR"
echo ""

# ── Step 1: Activate conda ──
step "Step 1: Activating conda environment 'caffe-ffi'"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
if ! hash conda 2>/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda" "/opt/miniconda3"; do
    [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; info "Conda found at: $d"; break; }
  done
fi
hash conda 2>/dev/null || { echo "ERROR: conda not found. Run from WSL terminal with conda initialized."; exit 1; }
conda activate caffe-ffi
pass "conda environment 'caffe-ffi' activated"

# ── Step 2: Clean rebuild if requested ──
if [ "$REBUILD" -eq 1 ]; then
  step "Step 2: Cleaning build directory (--rebuild)"
  if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    pass "Build directory cleaned"
  else
    warn "No build directory to clean"
  fi
else
  step "Step 2: Skipping clean (use --rebuild to force full rebuild)"
fi

# ── Step 3: CMake Configure ──
step "Step 3: Configuring CMake (default preset, COW enabled)"
if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/build.ninja" ]; then
  cmake --preset default \
    -DCAFFE_FFI_ENABLE_COW=ON \
    -DCAFFE_FFI_ENABLE_COW_PHASE3=ON
  pass "CMake configured"
else
  info "Build directory exists, skipping configure (use --rebuild to force)"
fi

# ── Step 4: Build ──
step "Step 4: Building caffe-ffi ($(nproc) parallel jobs)"
cmake --build --preset default -j"$(nproc)"
pass "Build complete"

# ── Step 5: Install (editable pip install) ──
step "Step 5: Installing caffe-ffi in editable mode"
pip install --no-build-isolation -e . --quiet
pass "Install complete"

# ── Step 6: Run C++ tests ──
TEST_BIN="$BUILD_DIR/caffe_ffi_tests"
if [ ! -f "$TEST_BIN" ]; then
  echo "ERROR: Test binary not found at $TEST_BIN"
  echo "Build may have failed. Check build output above."
  exit 1
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/src:${LD_LIBRARY_PATH:-}"

step "Step 6: Running C++ tests"
echo ""

set +e
if [ -n "$SUITE" ]; then
  # The C++ test framework does substring matching on "SuiteName.TestName".
  # Appending a dot ("SuiteName.") ensures exact suite match and avoids
  # false positives (e.g. "COWTest." won't match "COWApiTest.Xxx").
  info "Running suite: $SUITE  (filter='${SUITE}.')"
  echo ""
  "$TEST_BIN" "${SUITE}."
  RC=$?
elif [ -n "$FILTER" ]; then
  info "Running tests matching: '$FILTER'"
  echo ""
  "$TEST_BIN" "$FILTER"
  RC=$?
else
  info "Running all C++ tests (caffe_ffi_tests)..."
  info "  (test_net.cpp and test_insert_splits.cpp excluded by CMake config)"
  echo ""
  "$TEST_BIN"
  RC=$?
fi
set -e

echo ""
if [ $RC -eq 0 ]; then
  pass "C++ tests passed!"
else
  fail "C++ tests failed (exit code $RC)"
fi

# ── Step 7: Run Python COW tests (optional) ──
if [ "$RUN_PYTHON" -eq 1 ]; then
  echo ""
  step "Step 7: Running Python COW tests (test_cow.py)"
  set +e
  python -m pytest tests/python/test_cow.py -v --tb=short
  PY_RC=$?
  set -e
  if [ $PY_RC -eq 0 ]; then
    pass "Python COW tests passed!"
  else
    fail "Python COW tests failed (exit code $PY_RC)"
    RC=$((RC != 0 ? RC : PY_RC))
  fi
fi

# ── Summary ──
echo ""
echo -e "${CYAN}${BOLD}========================================${NC}"
if [ $RC -eq 0 ]; then
  echo -e "${GREEN}${BOLD}  ALL TESTS PASSED!${NC}"
else
  echo -e "${RED}${BOLD}  TEST FAILURES DETECTED (exit code $RC)${NC}"
fi
echo -e "${CYAN}${BOLD}========================================${NC}"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  bash scripts/wsl_build_cow_test.sh --rebuild                     # clean rebuild + all tests"
echo "  bash scripts/wsl_build_cow_test.sh --suite ZeroCopyTest          # run one suite"
echo "  bash scripts/wsl_build_cow_test.sh --suite COWTest"
echo "  bash scripts/wsl_build_cow_test.sh --suite ShareDataRefCount"
echo "  bash scripts/wsl_build_cow_test.sh --suite ShareDiffRefCount"
echo "  bash scripts/wsl_build_cow_test.sh --suite COWApiTest"
echo "  bash scripts/wsl_build_cow_test.sh --suite OwnerCOWTest"
echo "  bash scripts/wsl_build_cow_test.sh --suite SplitBackwardTest"
echo "  bash scripts/wsl_build_cow_test.sh --suite COWIntegrationTest"
echo "  bash scripts/wsl_build_cow_test.sh --suite SliceLayerZeroCopyTest"
echo "  bash scripts/wsl_build_cow_test.sh --suite COWRuntimeSwitchTest"
echo "  bash scripts/wsl_build_cow_test.sh --filter ShareDataMutationVisibleToBoth  # single test"
echo "  bash scripts/wsl_build_cow_test.sh --python                      # include Python tests"

exit $RC
