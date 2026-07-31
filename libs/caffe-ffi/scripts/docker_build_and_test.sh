#!/bin/bash
# Docker-based build & test script for Phase 3.0/3.1
#
# Uses the existing caffe-ffi-jupyter Docker image to build and test.
# No conda detection needed — Docker image has conda pre-installed at /opt/conda.
# Source is mounted via docker-compose: D:\spaces\SpecWeave → /SpecWeave
#
# Prerequisites:
#   1. Docker Desktop running with WSL2 backend
#   2. Image built: cd apps/caffe-ffi-jupyter && bash scripts/build.sh --cn
#
# Usage (from inside WSL terminal, at SpecWeave root):
#   cd /mnt/d/spaces/SpecWeave
#   bash projects/xuanspace/libs/caffe-ffi/scripts/docker_build_and_test.sh
set -eux -o pipefail

CONTAINER_NAME="${CONTAINER_NAME:-caffe-ffi-jupyter}"
SRC_DIR="/SpecWeave/projects/xuanspace/libs/caffe-ffi"
COMPOSE_FILE="apps/caffe-ffi-jupyter/docker-compose.yml"

# Colors for output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[docker-build]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[docker-build]${NC} $*"; }
log_error() { echo -e "${RED}[docker-build]${NC} $*" >&2; }

# ── Helper: run command in container with conda activated ──
docker_conda() {
    docker exec -w "$SRC_DIR" "$CONTAINER_NAME" \
        bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && $*"
}

# ── Preflight checks ──
hash docker 2>/dev/null || { log_error "docker not found. Install Docker Desktop for WSL first."; exit 1; }

# ── Ensure container is running ──
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Container '${CONTAINER_NAME}' not running. Starting it..."
    docker compose -f "$COMPOSE_FILE" up -d
    log_info "Waiting for container entrypoint (editable-install) to complete..."
    # Wait for editable-install to finish (it ends with "Starting service...")
    for i in $(seq 1 60); do
        if docker logs "$CONTAINER_NAME" 2>&1 | tail -5 | grep -q "Editable install phase complete"; then
            log_info "Container ready (editable install completed after ${i}s)"
            break
        fi
        sleep 2
    done
else
    log_info "Container '${CONTAINER_NAME}' already running"
fi

# ── Verify source is mounted ──
if ! docker exec "$CONTAINER_NAME" test -f "$SRC_DIR/pyproject.toml"; then
    log_error "Source not found at $SRC_DIR in container. Check docker-compose volume mount."
    exit 1
fi
log_info "Source directory: $SRC_DIR"

# ── Step 1: Clean previous build (if any) and reconfigure with Phase 3 ──
log_info "Step 1: Configure & Build (Phase 3 enabled: COW + LAZY_RESHAPE)"
docker_conda "rm -rf build && \
    SKBUILD_CMAKE_ARGS='-DCAFFE_FFI_ENABLE_COW=ON;-DCAFFE_FFI_ENABLE_COW_PHASE3=ON;-DCAFFE_FFI_BUILD_TESTS=OFF' \
    pip install --no-build-isolation -v -e ." || { log_error "pip install (build) FAILED — see errors above"; exit 1; }

# ── Step 2: Verify caffe_ffi imports ──
log_info "Step 2: Verify caffe_ffi import"
docker_conda "python -c \"import caffe_ffi; print('caffe_ffi version:', caffe_ffi.__version__)\""
docker_conda "python -c \"from caffe_ffi import Blob; b=Blob(); b.set_shape_only([2,3]); print('SetShapeOnly OK, lazy:', b.is_lazy_allocated())\""

# ── Step 3: Run Phase 3.0 log aggregation tests ──
log_info "Step 3: Run Phase 3.0 log aggregation tests"
docker_conda "CAFFE_FFI_ENABLE_COW=1 CAFFE_FFI_ENABLE_COW_PHASE3=1 \
    pytest tests/python/test_phase3_log_aggregation.py -v -s"

# ── Step 4: Run Phase 3.1 SetShapeOnly tests ──
log_info "Step 4: Run Phase 3.1 SetShapeOnly tests"
docker_conda "CAFFE_FFI_ENABLE_COW=1 CAFFE_FFI_ENABLE_COW_PHASE3=1 \
    pytest tests/python/test_phase3_set_shape_only.py -v -s"

# ── Step 5: Run FFI binding tests ──
log_info "Step 5: Run FFI binding tests"
docker_conda "CAFFE_FFI_ENABLE_COW=1 CAFFE_FFI_ENABLE_COW_PHASE3=1 \
    pytest tests/python/test_ffi_set_shape_only.py -v -s"

log_info "========================================="
log_info "  ALL TESTS PASSED"
log_info "========================================="