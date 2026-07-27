#!/bin/bash
set -e

VENDOR_DIR="/mnt/d/spaces/SpecWeave/projects/xuanspace/vendor"
LOG_FILE="${VENDOR_DIR}/build.log"

echo "Starting build at $(date)" > "$LOG_FILE"
echo "====================" >> "$LOG_FILE"

cd "$VENDOR_DIR"
DOCKER_BUILDKIT=1 docker build --no-cache \
    --build-arg APTPROXY=mirrors.aliyun.com \
    --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    --build-arg PIP_TRUSTED_HOST=mirrors.aliyun.com \
    -t caffe-cpu:customer \
    -f caffe/docker/standalone/pycaffe-customer/Dockerfile . >> "$LOG_FILE" 2>&1
BUILD_EXIT=$?

echo "" >> "$LOG_FILE"
echo "Build finished at $(date) with exit code $BUILD_EXIT" >> "$LOG_FILE"

if [ $BUILD_EXIT -eq 0 ]; then
    echo "Build SUCCESS"
    docker images caffe-cpu:customer --format 'table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}\t{{.ID}}'
else
    echo "Build FAILED"
    tail -150 "$LOG_FILE"
fi
