#!/bin/bash
# WSL test-only script for Phase 3.0/3.1
# IMPORTANT: Run from INSIDE a WSL terminal where conda is initialized.
#   cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi
#   bash scripts/wsl_run_test.sh
set -eux -o pipefail
cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi

[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null || true
if ! hash conda 2>/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda" "/opt/miniconda3"; do
    [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; echo "Conda: $d"; break; }
  done
fi
hash conda 2>/dev/null || { echo "ERROR: conda not found. Run from inside WSL terminal."; exit 1; }

conda activate caffe-ffi
pytest tests/python/test_phase3_set_shape_only.py -v -s