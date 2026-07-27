@echo on

REM 注意：apache-tvm-ffi 目前不在 conda-forge 上，需要通过 pip 预先安装
REM 在构建 conda 包之前，请确保：
REM 1. pip install apache-tvm-ffi
REM 或
REM 2. pip install -e ..\..\vendor\tvm-ffi (本地开发版本)

set CMAKE_GENERATOR=Ninja
"%PYTHON%" -m pip install apache-tvm-ffi
if errorlevel 1 exit 1

"%PYTHON%" -m pip install . --no-deps -vv
if errorlevel 1 exit 1
