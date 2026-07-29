@echo on

set CMAKE_GENERATOR=Ninja
"%PYTHON%" -m pip install apache-tvm-ffi
if errorlevel 1 exit 1

"%PYTHON%" -m pip install . --no-deps -vv
if errorlevel 1 exit 1
