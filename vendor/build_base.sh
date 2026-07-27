#!/bin/bash
set -e

VENDOR_DIR="/mnt/d/spaces/SpecWeave/projects/xuanspace/vendor"
DOCKERFILE="${VENDOR_DIR}/caffe/docker/standalone/pycaffe-customer/Dockerfile"

echo "Building caffe-cpu:customer with context: ${VENDOR_DIR}"
echo "Dockerfile: ${DOCKERFILE}"
echo ""

cd "${VENDOR_DIR}"
docker build --no-cache -t caffe-cpu:customer -f "${DOCKERFILE}" .
