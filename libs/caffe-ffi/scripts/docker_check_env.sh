#!/bin/bash
set -e

CONTAINER="caffe-ffi-jupyter"
SRC="/SpecWeave/projects/xuanspace/libs/caffe-ffi"

echo "=== Checking build tools ==="
docker exec $CONTAINER bash -c "which make; which ninja; which g++; which gcc; ls /usr/bin/make 2>/dev/null; ls /usr/bin/g++ 2>/dev/null; apt list --installed 2>/dev/null | grep -E 'build-essential|ninja|g\+\+'"

echo ""
echo "=== Source check ==="
docker exec $CONTAINER bash -c "ls $SRC/pyproject.toml && echo 'Source OK'"
