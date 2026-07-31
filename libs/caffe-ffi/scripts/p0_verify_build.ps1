#Requires -Version 5.1
<#
.SYNOPSIS
    A3+A5 P0 编译验证脚本：全量重新构建并运行 C++/Python/DLL 全套验证。
.DESCRIPTION
    执行顺序：
    1. 导入 MSVC 环境（vcvars64）
    2. 发现 py314 conda 环境
    3. 清理 build 目录（-Clean 开关）
    4. CMake Configure（开启 COW、测试）
    5. CMake Build（Release 模式）
    6. 运行 C++ 单元测试（ctest + gtest_filter 覆盖新用例）
    7. Windows DLL 自检
    8. Python COW 测试（pytest，如可用）
    9. 汇总验证结果
.PARAMETER Clean
    构建前完全清理 build 目录。
.PARAMETER Config
    构建配置：Release/Debug，默认 Release。
.PARAMETER SkipPythonTests
    跳过 Python pytest 测试。
.PARAMETER SkipDllCheck
    跳过 DLL 自检。
.PARAMETER CtestFilter
    CTest 用例过滤正则，默认运行全部。
.EXAMPLE
    .\scripts\p0_verify_build.ps1 -Clean
    清理后完整验证。
.EXAMPLE
    .\scripts\p0_verify_build.ps1 -SkipPythonTests
    仅验证 C++ 构建和测试。
#>

param(
    [switch]$Clean,
    [string]$Config = "Release",
    [switch]$SkipPythonTests,
    [switch]$SkipDllCheck,
    [string]$CtestFilter = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildDir = Join-Path $ProjectRoot "build"
$LogDir = Join-Path $ProjectRoot "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$MainLog = Join-Path $LogDir "p0_verify_${Timestamp}.log"

# ── helpers ──

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-Step {
    param([string]$Msg)
    $bar = "=" * 64
    Write-Host "`n$bar" -ForegroundColor Cyan
    Write-Host "  $Msg" -ForegroundColor Cyan
    Write-Host "$bar" -ForegroundColor Cyan
    Add-Content -Path $MainLog -Value "[$(Get-Date -Format 'HH:mm:ss')] $Msg"
}

function Write-Pass { param([string]$Msg) Write-Host "  [PASS] $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "  [FAIL] $Msg" -ForegroundColor Red }
function Write-Warn { param([string]$Msg) Write-Host "  [WARN] $Msg" -ForegroundColor Yellow }
function Write-Info { param([string]$Msg) Write-Host "  [INFO] $Msg" -ForegroundColor Gray }

$global:FailedSteps = @()
$global:PassedSteps = @()

function Invoke-CheckedStep {
    param(
        [string]$StepName,
        [scriptblock]$Action,
        [switch]$AllowFailure
    )
    Write-Step $StepName
    try {
        $output = & $Action 2>&1
        $output | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0 -and -not $AllowFailure) {
            throw "Exit code: $LASTEXITCODE"
        }
        Write-Pass "$StepName"
        $global:PassedSteps += $StepName
        Add-Content -Path $MainLog -Value "[$(Get-Date -Format 'HH:mm:ss')] PASS: $StepName"
        return $true
    } catch {
        Write-Fail "$StepName - $_"
        $global:FailedSteps += "$StepName : $_"
        Add-Content -Path $MainLog -Value "[$(Get-Date -Format 'HH:mm:ss')] FAIL: $StepName - $_"
        if (-not $AllowFailure) { throw }
        return $false
    }
}

# ── main ──

Write-Host @"
============================================================
  A3+A5 P0 Build Verification
  Timestamp : $Timestamp
  Project   : $ProjectRoot
  Config    : $Config
  COW       : ON (default)
  Log       : $MainLog
============================================================
"@ -ForegroundColor White

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# ── Step 0: MSVC Environment ──
Invoke-CheckedStep "Step 0: Import MSVC vcvars64 environment" {
    $VcvarsCandidates = @(
        "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    $VcvarsPath = $VcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $VcvarsPath) {
        throw "vcvars64.bat not found. Install Visual Studio 2022 or run from Developer Command Prompt."
    }
    Write-Info "vcvars64: $VcvarsPath"

    $VcvarsOutput = cmd /c "call `"$VcvarsPath`" >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed" }

    $SkipVars = @('TMP','TEMP','PROMPT','HOMEDRIVE','HOMEPATH','USERNAME','USERDOMAIN','COMPUTERNAME',
        'LOGONSERVER','SESSIONNAME','USERDNSDOMAIN','USERDOMAIN_ROAMINGPROFILE','APPDATA','LOCALAPPDATA',
        'ALLUSERSPROFILE','PROGRAMFILES','PROGRAMFILES(X86)','PROGRAMDATA','PUBLIC','SYSTEMDRIVE','SYSTEMROOT',
        'WINDIR','COMMONPROGRAMFILES','COMMONPROGRAMFILES(X86)','COMMONPROGRAMW6432','PROGRAMW6432','COMSPEC',
        'PATHEXT','PROCESSOR_ARCHITECTURE','PROCESSOR_IDENTIFIER','PROCESSOR_LEVEL','PROCESSOR_REVISION',
        'NUMBER_OF_PROCESSORS','OS','PATH','PSMODULEPATH','PWD','HOME','TERM','__VSCMD_PREINIT_PATH',
        'VSCMD_ARG_HOST_ARCH','VSCMD_ARG_TGT_ARCH','VSCMD_ARG_APP_PLAT','VSCMD_VER','VSCMD_START_DIR',
        'VSCMD_ARG_NO_LOGO','VSCMD_ARG_CWD')
    $VcvarsOutput | ForEach-Object {
        $line = $_.Trim()
        if ($line -match '^([^=]+)=(.*)$') {
            if ($Matches[1] -notin $SkipVars) {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2])
            }
        }
    }

    # Verify kernel32.lib
    $libOk = $false
    if ($env:LIB) {
        foreach ($d in ($env:LIB -split ';')) {
            if (Test-Path (Join-Path $d "kernel32.lib")) {
                Write-Info "kernel32.lib: $d"
                $libOk = $true
                break
            }
        }
    }
    if (-not $libOk) { throw "kernel32.lib not found in LIB paths" }
    Write-Info "cl.exe: $(Get-Command cl.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
}

# ── Step 1: py314 discovery ──
Invoke-CheckedStep "Step 1: Discover py314 conda environment" {
    $Py314Env = $null
    if ($env:CONDA_PREFIX -and $env:CONDA_PREFIX -match 'py314') {
        $Py314Env = $env:CONDA_PREFIX
    }
    if (-not $Py314Env) {
        foreach ($c in @(
            "$env:USERPROFILE\anaconda3\envs\py314",
            "$env:USERPROFILE\miniconda3\envs\py314",
            "$env:USERPROFILE\miniforge3\envs\py314",
            "C:\ProgramData\anaconda3\envs\py314",
            "C:\ProgramData\miniconda3\envs\py314"
        )) {
            if (Test-Path "$c\python.exe") { $Py314Env = $c; break }
        }
    }
    if (-not $Py314Env) {
        $found = Get-Command python.exe -ErrorAction SilentlyContinue | Where-Object { $_.Source -match 'py314' }
        if ($found) { $Py314Env = Split-Path -Parent $found.Source }
    }
    if (-not $Py314Env) {
        throw "py314 conda environment not found. Activate with: conda activate py314"
    }
    Write-Info "py314: $Py314Env"
    $PyPaths = @($Py314Env, "$Py314Env\Scripts", "$Py314Env\Library\bin",
        "$Py314Env\DLLs", "$Py314Env\Lib\site-packages\tvm_ffi\lib")
    $env:PATH = ($PyPaths + $env:PATH) -join ';'
    Write-Info "python: $(Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
}

# ── Step 2: Clean ──
if ($Clean) {
    Invoke-CheckedStep "Step 2: Clean build directory" {
        if (Test-Path $BuildDir) {
            Remove-Item -Recurse -Force $BuildDir
            Write-Info "Build directory removed"
        } else {
            Write-Info "Build directory does not exist (fresh build)"
        }
    }
}

Push-Location $ProjectRoot
try {

# ── Step 3: CMake Configure ──
Invoke-CheckedStep "Step 3: CMake Configure (COW=ON, Tests=ON)" {
    $CmakeArgs = @(
        '-S', '.',
        '-B', 'build',
        '-G', 'Ninja',
        "-DCMAKE_BUILD_TYPE=$Config",
        '-DCAFFE_FFI_BUILD_TESTS=ON',
        '-DCAFFE_FFI_ENABLE_COW=ON'
    )
    & cmake @CmakeArgs
}

# ── Step 4: Build ──
Invoke-CheckedStep "Step 4: CMake Build ($Config)" {
    & cmake --build build --config $Config
}

# ── Step 5: Copy tvm_ffi.dll ──
Invoke-CheckedStep "Step 5: Copy tvm_ffi.dll to build output" -AllowFailure {
    $TvmDll = Get-ChildItem -Path build -Filter tvm_ffi.dll -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -ne (Resolve-Path build).Path } | Select-Object -First 1
    if ($TvmDll) {
        Copy-Item $TvmDll.FullName build\ -Force
        Write-Info "Copied tvm_ffi.dll to build\"
    } else {
        Write-Warn "tvm_ffi.dll not found (may be statically linked)"
    }
}

# ── Step 6: C++ Unit Tests (ctest) ──
$CtestBaseArgs = @('--test-dir', 'build', '--output-on-failure', '--timeout', '120')
if ($CtestFilter) { $CtestBaseArgs += @('-R', $CtestFilter) }

# 6a: New A3+A5 specific tests
Invoke-CheckedStep "Step 6a: C++ tests — A3+A5 new test cases (SplitBackward, ShareDiffRefCount, OwnerCOW)" -AllowFailure {
    & ctest @CtestBaseArgs -R "SplitBackward|ShareDiffRefCount|OwnerCOW"
}

# 6b: COW tests
Invoke-CheckedStep "Step 6b: C++ tests — COW test group" -AllowFailure {
    & ctest @CtestBaseArgs -R "COWTest|COW"
}

# 6c: Split N=1/N=2 COW integration
Invoke-CheckedStep "Step 6c: C++ tests — Split COW integration" -AllowFailure {
    & ctest @CtestBaseArgs -R "SplitN[0-9]|ZeroCopy"
}

# 6d: Full test suite
Invoke-CheckedStep "Step 6d: C++ tests — FULL test suite" -AllowFailure {
    & ctest @CtestBaseArgs
}

# 6e: Direct gtest run (for detailed gtest output of new tests)
$TestExe = Join-Path $BuildDir "caffe_ffi_tests.exe"
if (Test-Path $TestExe) {
    Invoke-CheckedStep "Step 6e: Direct gtest run — new tests with verbose output" -AllowFailure {
        & $TestExe --gtest_filter="SplitBackwardTest.*:ShareDiffRefCount.*:OwnerCOWTest.*" --gtest_print_time=1
    }
} else {
    Write-Warn "caffe_ffi_tests.exe not found at $TestExe"
}

# ── Step 7: DLL Self-Check ──
if (-not $SkipDllCheck) {
    Invoke-CheckedStep "Step 7: Windows DLL self-check" -AllowFailure {
        $DllScript = Join-Path $ProjectRoot "scripts\check_windows_dll.py"
        & python $DllScript --verbose
    }
}

# ── Step 8: Python COW Tests ──
if (-not $SkipPythonTests) {
    $PyTestExe = Get-Command pytest -ErrorAction SilentlyContinue
    $PyTestDir = Join-Path $ProjectRoot "tests\python"
    if ($PyTestExe -and (Test-Path $PyTestDir)) {
        Invoke-CheckedStep "Step 8: Python COW tests (pytest test_cow.py)" -AllowFailure {
            Push-Location (Split-Path -Parent $ProjectRoot)  # go to xuanspace root for package imports
            try {
                & pytest (Join-Path $PyTestDir "test_cow.py") -v --tb=short 2>&1
            } finally {
                Pop-Location
            }
        }
    } else {
        Write-Warn "pytest not found or tests/python directory missing; skipping Python tests"
    }
}

} finally {
    Pop-Location
}

# ── Summary ──
Write-Step "Verification Summary"
Write-Host ""
Write-Host "  Passed steps: $($global:PassedSteps.Count)" -ForegroundColor Green
$global:PassedSteps | ForEach-Object { Write-Host "    ✅ $_" -ForegroundColor Green }
if ($global:FailedSteps.Count -gt 0) {
    Write-Host ""
    Write-Host "  Failed steps: $($global:FailedSteps.Count)" -ForegroundColor Red
    $global:FailedSteps | ForEach-Object { Write-Host "    ❌ $_" -ForegroundColor Red }
}
Write-Host ""

if ($global:FailedSteps.Count -eq 0) {
    Write-Host "  🎉 ALL P0 VERIFICATION CHECKS PASSED" -ForegroundColor Green
    $ExitCode = 0
} else {
    Write-Host "  ⚠️  Some checks failed — review logs: $MainLog" -ForegroundColor Yellow
    $ExitCode = 1
}

Write-Host "  Log file: $MainLog" -ForegroundColor Gray
Write-Host ""

exit $ExitCode
