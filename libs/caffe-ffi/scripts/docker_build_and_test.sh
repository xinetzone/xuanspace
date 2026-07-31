#!/bin/bash
# Docker-based build & test script for Phase 3.0/3.1
#
# Uses the existing caffe-ffi-jupyter Docker image to build and test.
# No conda detection needed — the Docker image has conda pre-installed at
# /opt/conda/envs/caffe-ffi/.
#
# Prerequisites:
#   docker compose -f apps/caffe-ffi-jupyter/docker-compose.yml up -d
#
# Usage (from inside WSL terminal):
#   cd /mnt/d/spaces/SpecWeave
#   bash projects/xuanspace/libs/caffe-ffi/scripts/docker_build_and_test.sh
set -eux -o pipefail

CONTAINER_NAME="${CONTAINER_NAME:-caffe-ffi-jupyter}"
CONDA_ENV="/opt/conda/envs/caffe-ffi"
SRC_DIR="/SpecWeave/projects/xuanspace/libs/caffe-ffi"

# Ensure Docker is available
hash docker 2>/dev/null || { echo "ERROR: docker not found. Install Docker Desktop for WSL first."; exit 1; }

# Ensure container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container '${CONTAINER_NAME}' not running. Starting it..."
  docker compose -f apps/caffe-ffi-jupyter/docker-compose.yml up -d
  echo "Waiting for container to be ready..."
  sleep 10
fi

echo "=== Step 1: Configure CMake (Phase 3 enabled) ==="
docker exec -w "$SRC_DIR" "$CONTAINER_NAME" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && \
    cmake --preset default \
      -DCAFFE_FFI_ENABLE_COW=ON \
      -DCAFFE_FFI_ENABLE_COW_PHASE3=ON"

echo "=== Step 2: Build ==="
docker exec -w "$SRC_DIR" "$CONTAINER_NAME" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && \
    cmake --build --preset default -j\$(nproc)"

echo "=== Step 3: Install ==="
docker exec -w "$SRC_DIR" "$CONTAINER_NAME" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && \
    pip install --no-build-isolation -e ."

echo "=== Step 4: Run Phase 3.1 SetShapeOnly tests ==="
docker exec -w "$SRC_DIR" "$CONTAINER_NAME" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && \
    pytest tests/python/test_phase3_set_shape_only.py -v -s"

echo "=== Step 5: Run FFI binding tests ==="
docker exec -w "$SRC_DIR" "$CONTAINER_NAME" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi && \
    pytest tests/python/test_ffi_set_shape_only.py -v -s"

echo "=== DONE ==="