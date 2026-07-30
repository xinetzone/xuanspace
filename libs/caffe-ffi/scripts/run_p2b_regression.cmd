@echo off
REM ============================================================================
REM  run_p2b_regression.cmd  -  Windows P2-B 回归测试一键运行脚本
REM
REM  用途：在 Visual Studio Developer Command Prompt 环境中，自动完成
REM        环境检测 → CMake 配置构建 → C++ 单元测试 → Python P2-B 回归测试 →
REM        性能日志汇总 的完整流程。
REM
REM  前置条件：
REM    1. 从 "x64 Native Tools Command Prompt for VS 2022"（或 VS 2019）启动
REM    2. 已安装 conda 且存在 py314 环境（含 Python 3.14 + Protobuf ≥ 7）
REM    3. 项目根目录为当前脚本所在目录的父目录
REM
REM  用法：
REM    scripts\run_p2b_regression.cmd              [默认: Debug 构建, 跑全部测试]
REM    scripts\run_p2b_regression.cmd Release       Release 构建
REM    scripts\run_p2b_regression.cmd Debug quick   Debug 构建, 跳过 C++ 测试
REM ============================================================================

setlocal enabledelayedexpansion

REM ── 切换到项目根目录 ──
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"
echo [INFO] Project root: %PROJECT_ROOT%

REM ── 解析参数 ──
set "BUILD_TYPE=Debug"
set "QUICK_MODE=0"
if not "%~1"=="" set "BUILD_TYPE=%~1"
if /i "%~2"=="quick" set "QUICK_MODE=1"
echo [INFO] Build type: %BUILD_TYPE%, Quick mode: %QUICK_MODE%

REM ── 步骤 0: 检测 VS Developer Command Prompt ──
echo.
echo ============================================================
echo [STEP 0] 检测 Visual Studio 构建环境
echo ============================================================
where cl.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未检测到 cl.exe，请从 "x64 Native Tools Command Prompt for VS" 启动本脚本
    echo         VS 2022 路径通常为:
    echo         C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
    exit /b 1
)
echo [OK] cl.exe 已找到
cl.exe 2>&1 | findstr "Version"

REM ── 步骤 1: 激活 conda py314 环境 ──
echo.
echo ============================================================
echo [STEP 1] 激活 conda py314 环境
echo ============================================================

REM 尝试多种 conda 激活方式
set "CONDA_ACTIVATED=0"

REM 方式1: 已在环境中（CONDA_PREFIX 已设置）
if defined CONDA_PREFIX (
    if /i "%CONDA_DEFAULT_ENV%"=="py314" (
        echo [OK] 已在 conda py314 环境中: %CONDA_PREFIX%
        set "CONDA_ACTIVATED=1"
    )
)

REM 方式2: 通过 conda activate
if %CONDA_ACTIVATED%==0 (
    where conda.bat >nul 2>&1
    if %errorlevel%==0 (
        REM 初始化 conda 到当前 shell
        for /f "delims=" %%i in ('conda.bat info --base 2^>nul') do set "CONDA_BASE=%%i"
        if exist "!CONDA_BASE!\condabin\conda_hook.bat" (
            call "!CONDA_BASE!\condabin\conda_hook.bat"
            call conda activate py314 2>nul
            if !errorlevel!==0 (
                echo [OK] 已通过 conda activate py314 激活
                set "CONDA_ACTIVATED=1"
            )
        )
    )
)

REM 方式3: 常见 conda 安装路径
if %CONDA_ACTIVATED%==0 (
    for %%P in (
        "%USERPROFILE%\anaconda3"
        "%USERPROFILE%\miniconda3"
        "D:\Users\%USERNAME%\anaconda3"
        "C:\ProgramData\anaconda3"
        "C:\ProgramData\miniconda3"
    ) do (
        if exist "%%~P\condabin\conda_hook.bat" (
            call "%%~P\condabin\conda_hook.bat"
            call conda activate py314 2>nul
            if !errorlevel!==0 (
                echo [OK] 已通过 %%~P 激活 py314 环境
                set "CONDA_ACTIVATED=1"
                goto :conda_done
            )
        )
        if exist "%%~P\envs\py314\python.exe" (
            set "PYTHON_EXE=%%~P\envs\py314\python.exe"
            echo [OK] 找到 py314 python.exe: !PYTHON_EXE!
            set "CONDA_ACTIVATED=2"
            goto :conda_done
        )
    )
)
:conda_done

if %CONDA_ACTIVATED%==0 (
    echo [WARN] 未找到 conda py314 环境，尝试使用系统 Python
    where python.exe >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] 未找到 Python，请安装 conda 并创建 py314 环境
        exit /b 1
    )
    set "PYTHON_EXE=python.exe"
) else (
    if %CONDA_ACTIVATED%==1 (
        set "PYTHON_EXE=python"
    )
)

echo [INFO] Python 可执行文件: %PYTHON_EXE%
%PYTHON_EXE% --version
if %errorlevel% neq 0 (
    echo [ERROR] Python 不可用
    exit /b 1
)

REM ── 步骤 2: CMake 配置与构建 ──
echo.
echo ============================================================
echo [STEP 2] CMake 配置与构建 (%BUILD_TYPE%)
echo ============================================================

set "BUILD_DIR=build\%BUILD_TYPE%"

if not exist "%BUILD_DIR%" (
    echo [INFO] 创建构建目录: %BUILD_DIR%
    mkdir "%BUILD_DIR%"
)

REM 检测 conda 前缀用于 CMAKE_PREFIX_PATH
if %CONDA_ACTIVATED%==1 (
    set "CMAKE_PREFIX_PATH_OPT=-DCMAKE_PREFIX_PATH=%CONDA_PREFIX%/Library"
) else if %CONDA_ACTIVATED%==2 (
    for %%P in ("%PYTHON_EXE%") do set "ENV_DIR=%%~dpP.."
    set "CMAKE_PREFIX_PATH_OPT=-DCMAKE_PREFIX_PATH=!ENV_DIR!\Library"
) else (
    set "CMAKE_PREFIX_PATH_OPT="
)

echo [INFO] 运行 CMake configure...
cmake -S . -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% %CMAKE_PREFIX_PATH_OPT% -DCAFFE_FFI_BUILD_TESTS=ON
if %errorlevel% neq 0 (
    echo [ERROR] CMake configure 失败
    exit /b 1
)
echo [OK] CMake configure 成功

echo [INFO] 构建...
cmake --build "%BUILD_DIR%" --config %BUILD_TYPE% --parallel
if %errorlevel% neq 0 (
    echo [ERROR] CMake build 失败
    exit /b 1
)
echo [OK] 构建成功

REM ── 步骤 3: 构建 Python 扩展（pip install -e . --no-build-isolation）──
echo.
echo ============================================================
echo [STEP 3] 安装 Python 扩展（editable mode）
echo ============================================================

set "KMP_DUPLICATE_LIB_OK=TRUE"
%PYTHON_EXE% -m pip install -e . --no-build-isolation 2>&1
if %errorlevel% neq 0 (
    echo [WARN] pip install -e 失败，尝试仅使用预编译 DLL
    set "PYTHONPATH=%PROJECT_ROOT%\%BUILD_DIR%;%PYTHONPATH%"
)

REM 复制 _caffe_ffi.dll 到 python 包目录
if exist "%BUILD_DIR%\caffe_ffi\_caffe_ffi.pyd" (
    echo [OK] _caffe_ffi.pyd 已在输出目录
) else if exist "%BUILD_DIR%\_caffe_ffi.dll" (
    echo [INFO] 复制 _caffe_ffi.dll 到 python 包目录...
    if not exist "python\caffe_ffi" mkdir "python\caffe_ffi"
    copy /y "%BUILD_DIR%\_caffe_ffi.dll" "python\caffe_ffi\_caffe_ffi.pyd" >nul 2>&1
)

REM ── 步骤 4: C++ 单元测试 ──
echo.
echo ============================================================
echo [STEP 4] C++ 单元测试
echo ============================================================

if %QUICK_MODE%==0 (
    if exist "%BUILD_DIR%\caffe_ffi_tests.exe" (
        echo [INFO] 运行 C++ 单元测试...
        cd /d "%PROJECT_ROOT%\%BUILD_DIR%"
        caffe_ffi_tests.exe
        set "CPP_TEST_RESULT=!errorlevel!"
        cd /d "%PROJECT_ROOT%"
        if !CPP_TEST_RESULT! neq 0 (
            echo [ERROR] C++ 单元测试失败（exit code: !CPP_TEST_RESULT!）
        ) else (
            echo [OK] C++ 单元测试全部通过
        )
    ) else (
        echo [WARN] 未找到 caffe_ffi_tests.exe，跳过 C++ 测试
    )
) else (
    echo [INFO] Quick 模式，跳过 C++ 单元测试
)

REM ── 步骤 5: Python P2-B 回归测试 ──
echo.
echo ============================================================
echo [STEP 5] Python P2-B 回归测试
echo ============================================================

set "P2B_TEST=tests\python\test_p2b_regression.py"
if exist "%P2B_TEST%" (
    echo [INFO] 运行 P2-B 回归测试（详细输出 + SPLIT-PERF 日志）...
    echo ------------------------------------------------------------
    %PYTHON_EXE% -m pytest "%P2B_TEST%" -v -s --tb=short
    set "PYTEST_RESULT=!errorlevel!"
    echo ------------------------------------------------------------
    if !PYTEST_RESULT! neq 0 (
        echo [ERROR] P2-B 回归测试失败（exit code: !PYTEST_RESULT!）
    ) else (
        echo [OK] P2-B 回归测试全部通过
    )
) else (
    echo [ERROR] 未找到测试文件: %P2B_TEST%
    set "PYTEST_RESULT=1"
)

REM ── 步骤 6: 性能日志汇总 ──
echo.
echo ============================================================
echo [STEP 6] 性能日志汇总
echo ============================================================

set "PERF_LOG_DIR=tests\python\.temp"
if exist "%PERF_LOG_DIR%" (
    echo [INFO] 查找最新的性能日志 CSV...
    set "LATEST_CSV="
    for /f "delims=" %%f in ('dir /b /o-d "%PERF_LOG_DIR%\perf_log_*.csv" 2^>nul') do (
        if not defined LATEST_CSV set "LATEST_CSV=%PERF_LOG_DIR%\%%f"
    )
    if defined LATEST_CSV (
        echo [OK] 最新性能日志: !LATEST_CSV!
        echo.
        echo ===== CSV 日志摘要 =====
        %PYTHON_EXE% -c "import csv,sys; rows=list(csv.reader(open(r'!LATEST_CSV!',encoding='utf-8'))); print(f'Total records: {len(rows)-1}'); [print(' | '.join(r[:7])) for r in rows[:20]]" 2>nul
        echo.
        echo [INFO] 完整日志文件: %PROJECT_ROOT%\!LATEST_CSV!
    ) else (
        echo [INFO] 未找到性能日志文件（测试可能未触达 perf_trace 记录点）
    )
) else (
    echo [INFO] 性能日志目录不存在: %PERF_LOG_DIR%
)

REM ── 步骤 7: 汇总结果 ──
echo.
echo ============================================================
echo [RESULT] P2-B 回归测试结果汇总
echo ============================================================
echo   Build type      : %BUILD_TYPE%
echo   C++ tests       : %QUICK_MODE:1=skipped%
echo   Python P2-B     : %PYTEST_RESULT:0=PASSED%
echo.

if %PYTEST_RESULT%==0 (
    if %QUICK_MODE%==1 (
        echo [DONE] P2-B 快速回归完成（跳过 C++ 测试）
        exit /b 0
    )
    if defined CPP_TEST_RESULT (
        if !CPP_TEST_RESULT!==0 (
            echo [DONE] 所有测试通过!
            exit /b 0
        )
    ) else (
        echo [DONE] Python 测试通过!
        exit /b 0
    )
)

echo [FAIL] 存在失败的测试，请查看上方日志
exit /b 1
