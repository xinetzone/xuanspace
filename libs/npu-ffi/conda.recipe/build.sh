#!/bin/bash
set -ex

# 注意：apache-tvm-ffi 目前不在 conda-forge 上，需要通过 pip 预先安装
# 在构建 conda 包之前，请确保：
# 1. pip install apache-tvm-ffi
# 或
# 2. pip install -e ../../vendor/tvm-ffi (本地开发版本)

export CMAKE_GENERATOR=Ninja
$PYTHON -m pip install apache-tvm-ffi  # 确保 tvm-ffi 可用
$PYTHON -m pip install . --no-deps -vv
