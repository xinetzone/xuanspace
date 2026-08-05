#!/usr/bin/env bash
# Rebuild the caffe-ffi C++ extension in the P0 container and refresh the .so
# in the editable-installed package, then run the P2 test suite.
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate caffe-ffi

cd /SpecWeave/projects/xuanspace/libs/caffe-ffi

echo "=== 1. py_compile check on edited tests ==="
python -m py_compile tests/python/test_p2_loss_ops.py tests/python/test_p2_other_ops.py

echo "=== 2. locate build dir ==="
ls -d build build/* 2>/dev/null || echo "no build dir found"

echo "=== 3. rebuild extension ==="
if [ -f build/CMakeCache.txt ]; then
  cmake --build build --target _caffe_ffi -j 4 2>&1 | tail -n 30
else
  echo "no build cache; running editable install"
  pip install --no-build-isolation -e . 2>&1 | tail -n 30
fi

echo "=== 4. refresh .so into package ==="
PKG=python/caffe_ffi
find build -name "_caffe_ffi*.so" 2>/dev/null | while read -r so; do
  echo "copying $so -> $PKG/"
  cp -f "$so" "$PKG/"
done

echo "=== 5. verify import ==="
python -c "import caffe_ffi; print('Version:', caffe_ffi.__version__, 'available:', caffe_ffi._ffi_api.is_available())"