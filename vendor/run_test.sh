#!/bin/bash
set -e

SCRIPT_DIR="/mnt/d/spaces/SpecWeave/projects/xuanspace/vendor"
PATCH_DIR="${SCRIPT_DIR}/caffe/docker/standalone/pycaffe-customer/python/caffe_patches"
WORKSPACE_DIR="${SCRIPT_DIR}/caffe/workspace"
CAFFE_SITE="/usr/local/lib/python3.14/dist-packages/caffe"

docker rm -f caffe-test 2>/dev/null || true

docker run --rm --name caffe-test \
  -v "${PATCH_DIR}/caffe/__init__.py:${CAFFE_SITE}/__init__.py" \
  -v "${PATCH_DIR}/caffe/io.py:${CAFFE_SITE}/io.py" \
  -v "${PATCH_DIR}/caffe/proto/__init__.py:${CAFFE_SITE}/proto/__init__.py" \
  -v "${WORKSPACE_DIR}:/workspace/user-data" \
  caffe-cpu:customer \
  bash -c 'cd /workspace/user-data && python test_bvlc_compat.py'
