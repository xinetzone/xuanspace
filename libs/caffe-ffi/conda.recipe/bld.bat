@echo on
setlocal EnableDelayedExpansion

set CMAKE_GENERATOR=Ninja

echo "[bld.bat] Installing apache-tvm-ffi via pip..."
"%PYTHON%" -m pip install apache-tvm-ffi --no-deps -vv
if errorlevel 1 exit 1

echo "[bld.bat] Cleaning in-tree build artifacts..."
rmdir /s /q build 2>nul
rmdir /s /q _skbuild 2>nul
rmdir /s /q dist 2>nul
rmdir /s /q *.egg-info 2>nul
del /q python\caffe_ffi\*.pyd 2>nul
del /q python\caffe_ffi\*.dll 2>nul

echo "[bld.bat] Building and installing caffe-ffi via pip..."
"%PYTHON%" -m pip install . --no-deps -vv --no-build-isolation
if errorlevel 1 exit 1

echo "[bld.bat] Verifying build output..."
if not exist "%SP_DIR%\caffe_ffi\_caffe_ffi*.pyd" (
    echo "[bld.bat] ERROR: _caffe_ffi*.pyd not found in %SP_DIR%\caffe_ffi\"
    dir "%SP_DIR%\caffe_ffi\" /s /b
    exit 1
)

echo "[bld.bat] Verifying tvm_ffi package..."
if not exist "%SP_DIR%\tvm_ffi" (
    echo "[bld.bat] WARNING: tvm_ffi package directory not found"
) else (
    echo "[bld.bat] tvm_ffi package exists at %SP_DIR%\tvm_ffi"
    dir "%SP_DIR%\tvm_ffi" /s /b
)

echo "[bld.bat] ============================================================"
echo "[bld.bat] Build completed successfully!"
echo "[bld.bat] _caffe_ffi location: %SP_DIR%\caffe_ffi\"
echo "[bld.bat] ============================================================"
