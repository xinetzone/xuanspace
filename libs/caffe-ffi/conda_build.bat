@echo off
REM caffe-ffi Conda 环境 Windows 构建脚本
REM 使用方法：conda activate caffe-ffi 后运行本脚本

setlocal enabledelayedexpansion

echo ========================================
echo  caffe-ffi Conda Build (Windows)
echo ========================================

REM ── 检查 conda 环境 ──
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] conda 未找到，请先安装 Miniconda/Anaconda
    exit /b 1
)

if "%CONDA_DEFAULT_ENV%"=="" (
    echo [WARN] 未检测到激活的 conda 环境，尝试激活 caffe-ffi...
    call conda activate caffe-ffi
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] 无法激活 caffe-ffi 环境，请先运行: conda env create -f environment.yml
        exit /b 1
    )
)

echo [INFO] Conda 环境: %CONDA_DEFAULT_ENV%
echo [INFO] Python:
python --version

REM ── 设置 OpenMP 环境变量（避免 libiomp5md.dll 冲突）──
set KMP_DUPLICATE_LIB_OK=TRUE

REM ── 创建构建目录 ──
if not exist build mkdir build

REM ── 配置 CMake ──
echo.
echo [STEP 1/3] CMake Configure...
cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%/Library" ^
    -DProtobuf_DIR="%CONDA_PREFIX%/Library/lib/cmake/protobuf" ^
    -DOPENBLAS_HOME="%CONDA_PREFIX%/Library"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configure failed
    exit /b 1
)

REM ── 编译 ──
echo.
echo [STEP 2/3] Building (Ninja)...
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed
    exit /b 1
)

REM ── 安装 Python 包（editable模式）──
echo.
echo [STEP 3/3] Installing Python package (editable)...
pip install -e . --no-build-isolation
if %ERRORLEVEL% neq 0 (
    echo [ERROR] pip install failed
    exit /b 1
)

REM ── 运行测试 ──
echo.
echo [TEST] Running pytest...
python -m pytest tests/python/ -v --tb=short

echo.
echo ========================================
echo  Build complete!
echo  DLL location: build\Release\_caffe_ffi.dll
echo ========================================

endlocal
