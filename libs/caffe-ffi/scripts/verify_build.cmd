@echo off
setlocal enabledelayedexpansion

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "CONDA_ENV=D:\Users\xinzo\anaconda3\envs\py314"

call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
REM 确保 py314 环境在 PATH 最前面（vcvars 可能追加了系统路径）
set "PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin;%CONDA_ENV%\DLLs;%CONDA_ENV%\Lib\site-packages\tvm_ffi\lib;%PATH%"

cd /d "%~dp0.."

echo ============================================================
echo  Step 0: Environment Diagnostics
echo ============================================================
echo CMake in use: 
where cmake 2>nul | findstr /v "^$"
echo.
echo LIB env var (first 200 chars):
if defined LIB (echo %LIB:~0,200%) else (echo [MISSING] LIB not set)
echo.
echo INCLUDE env var (first 200 chars):
if defined INCLUDE (echo %INCLUDE:~0,200%) else (echo [MISSING] INCLUDE not set)
echo.
echo Checking kernel32.lib:
for %%d in (%LIB:;= %) do (
    if exist "%%d\kernel32.lib" (
        echo   [OK] %%d\kernel32.lib
        set "LIB_OK=1"
    )
)
if not defined LIB_OK (
    echo   [ERROR] kernel32.lib not found in any LIB path
    echo   vcvars64.bat may not have set LIB correctly
    echo   Try running this script from VS Developer Command Prompt
    exit /b 1
)
echo.
echo Checking cl.exe: 
where cl.exe 2>nul | findstr /v "^$" || echo [ERROR] cl.exe not found
echo.
echo [OK] Environment ready

echo.
echo ============================================================
echo  Step 1: Clean CMake cache (force reconfigure)
echo ============================================================
del /q build\CMakeCache.txt 2>nul
rmdir /s /q build\CMakeFiles 2>nul
echo [OK] Cache cleared

echo.
echo ============================================================
echo  Step 2: CMake configure
echo ============================================================
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCAFFE_FFI_BUILD_TESTS=ON
if errorlevel 1 (
    echo [ERROR] CMake configure failed
    echo Check: TVM_FFI_USE_BUILTIN_TYPETRAITS is defined in CompilerConfig.cmake
    echo Check: /WX is set in MSVC compile options
    exit /b 1
)
echo [OK] CMake configure succeeded

echo.
echo ============================================================
echo  Step 3: Build
echo ============================================================
cmake --build build --config Release
if errorlevel 1 (
    echo [ERROR] Build failed
    echo If warnings are treated as errors, check the compile output
    echo If symbol visibility causes issues, comment out -fvisibility flags
    exit /b 1
)
echo [OK] Build succeeded

echo.
echo ============================================================
echo  Step 4: Copy tvm_ffi.dll to build output directory
echo ============================================================
REM tvm_ffi_shared is built as dependency of caffe_ffi_tests
REM but its output goes to python/caffe_ffi/, not build root
REM Manual copy ensures the test executable can find it
set "TVM_FFI_DLL="
for /r build %%f in (tvm_ffi.dll) do if exist "%%f" set "TVM_FFI_DLL=%%f"
if defined TVM_FFI_DLL (
    copy /y "%TVM_FFI_DLL%" build\ >nul
    echo [OK] Copied %TVM_FFI_DLL% to build\
) else (
    echo [WARN] tvm_ffi.dll not found in build tree, checking PATH
)

echo.
echo ============================================================
echo  Step 5: Run C++ unit tests
echo ============================================================
build\caffe_ffi_tests.exe
set "TEST_RESULT=%errorlevel%"
if %TEST_RESULT% equ 0 (
    echo [OK] All C++ tests passed
) else (
    echo [WARN] C++ tests had failures (exit code: %TEST_RESULT%)
)
echo.
echo ============================================================
echo  Verification complete
echo ============================================================
echo.
echo New CMake config applied:
echo   - TVM_FFI_USE_BUILTIN_TYPETRAITS: prevents custom TypeTraits conflicts
echo     (cmake/CompilerConfig.cmake line 62)
echo   - /WX: treats warnings as errors on MSVC
echo     (cmake/CompilerConfig.cmake line 79)
echo   - -fvisibility=hidden: hides WEAK symbols on GCC/Clang
echo     (cmake/CompilerConfig.cmake line 83)
echo   - -Wl,--exclude-libs,ALL: excludes static lib symbols on GNU
echo     (cmake/CompilerConfig.cmake line 101)
echo   - CAFFE_FFI_ENABLE_COW: Phase 2 COW switch (default OFF)
echo     (cmake/Options.cmake line 12)
echo.
echo New test file: tests/cpp/test_objectptr_migration.cpp
echo   - 12 test cases for ObjectPtr migration patterns
echo   - Scenes: ownership transfer, FFI lambda, bulk ops
echo.
echo Run verification: scripts\verify_build.cmd
endlocal & exit /b %TEST_RESULT%