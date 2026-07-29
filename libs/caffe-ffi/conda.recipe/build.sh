#!/bin/bash
set -ex

export CMAKE_GENERATOR=Ninja
$PYTHON -m pip install apache-tvm-ffi
$PYTHON -m pip install . --no-deps -vv
