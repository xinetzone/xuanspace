#!/usr/bin/env bash
# Run P2 unit tests in the P0 Docker container (must be run inside the container).
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate caffe-ffi

cd /SpecWeave/projects/xuanspace/libs/caffe-ffi

echo "=== Runtime check ==="
python -c "import caffe_ffi; print('Version:', caffe_ffi.__version__, 'available:', caffe_ffi._ffi_api.is_available())"

echo "=== P2 test files ==="
python -m pytest tests/python/test_p2_loss_ops.py tests/python/test_p2_other_ops.py tests/python/test_p2_data_io_ops.py tests/python/test_callback_registry_cleanup.py -v --tb=short 2>&1 | tail -n 80