#Requires -Version 5.1
<#
.SYNOPSIS
    caffe-ffi COW Phase 2 CI 脚本：自动构建、运行 COW 测试、DLL 自检，输出日志到文件。
.DESCRIPTION
    执行顺序：CMake 配置 → 构建 → CTest COW 测试 → DLL 自检
    所有输出捕获到 logs/ 目录下的带时间戳日志文件。
.PARAMETER Clean
    构建前清理 build 目录。
.PARAMETER SkipDllCheck
    跳过 DLL 自检步骤。
.PARAMETER SkipBuild
    跳过构建步骤（仅运行测试和自检）。
.EXAMPLE
    .\scripts\ci_cow_test.ps1
    完整 CI 流程。
.EXAMPLE
    .\scripts\ci_cow_test.ps1 -Clean
    清理后重新构建并测试。
.EXAMPLE
    .\scripts\ci_cow_test.ps1 -SkipBuild
    仅运行测试和 DLL 自检（适用于已构建场景）。
#>

param(
    [switch]$Clean,
    [switch]$SkipDllCheck,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectRoot "build"
$logDir = Join-Path $projectRoot "logs"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$mainLog = Join-Path $logDir "ci_cow_${timestamp}.log"

# Ensure log directory exists
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# ── helpers ──

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -Path $mainLog -Value $line
    Write-Host $line -ForegroundColor $Color
}

function Write-Step {
    param([string]$Message)
    $line = "=" * 60
    Write-Log $line "Cyan"
    Write-Log "  STEP: $Message" "Cyan"
    Write-Log $line "Cyan"
}

function Invoke-Step {
    param(
        [string]$StepName,
        [scriptblock]$ScriptBlock,
        [string]$LogFile,
        [switch]$AllowFailure
    )
    Write-Step $StepName
    try {
        & $ScriptBlock 2>&1 | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0 -and -not $AllowFailure) {
            throw "Exit code: $LASTEXITCODE"
        }
        Write-Log "  [PASS] $StepName completed" "Green"
        return $true
    } catch {
        Write-Log "  [FAIL] $StepName failed: $_" "Red"
        if (-not $AllowFailure) {
            throw
        }
        return $false
    }
}

# ── main ──

Write-Log "========================================" "Cyan"
Write-Log "  caffe-ffi COW Phase 2 CI Pipeline" "Cyan"
Write-Log "  Started: $timestamp" "Cyan"
Write-Log "  Project: $projectRoot" "Cyan"
Write-Log "  Log:     $mainLog" "Cyan"
Write-Log "========================================" "Cyan"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$overallSuccess = $true

# Step 1: Clean (optional)
if ($Clean) {
    Write-Step "Cleaning build directory"
    if (Test-Path $buildDir) {
        Remove-Item -Recurse -Force $buildDir
        Write-Log "  Build directory removed" "Yellow"
    }
}

# Step 2: CMake Configure
if (-not $SkipBuild) {
    $configureLog = Join-Path $logDir "cmake_configure_${timestamp}.log"
    $result = Invoke-Step -StepName "CMake Configure" -ScriptBlock {
        Push-Location $projectRoot
        try {
            cmake --preset default
        } finally {
            Pop-Location
        }
    } -LogFile $configureLog
    if (-not $result) { $overallSuccess = $false }

    # Step 3: CMake Build
    $buildLog = Join-Path $logDir "cmake_build_${timestamp}.log"
    $result = Invoke-Step -StepName "CMake Build" -ScriptBlock {
        Push-Location $projectRoot
        try {
            cmake --build --preset default
        } finally {
            Pop-Location
        }
    } -LogFile $buildLog
    if (-not $result) { $overallSuccess = $false }
}

# Step 4: Run COW unit tests (Blob-level)
$cowUnitLog = Join-Path $logDir "ctest_cow_unit_${timestamp}.log"
$result = Invoke-Step -StepName "CTest: COW Unit Tests (Blob-level)" -ScriptBlock {
    ctest --test-dir $buildDir -R "COWTest" --output-on-failure
} -LogFile $cowUnitLog -AllowFailure
if (-not $result) { $overallSuccess = $false }

# Step 5: Run COW integration tests (Split N=2 via Net)
$cowIntegLog = Join-Path $logDir "ctest_cow_integ_${timestamp}.log"
$result = Invoke-Step -StepName "CTest: COW Integration Tests (Split N=2)" -ScriptBlock {
    ctest --test-dir $buildDir -R "SplitN2COW|SplitN1ZeroCopy" --output-on-failure
} -LogFile $cowIntegLog -AllowFailure
if (-not $result) { $overallSuccess = $false }

# Step 6: Run full zero-copy test suite
$fullZcLog = Join-Path $logDir "ctest_zerocopy_full_${timestamp}.log"
$result = Invoke-Step -StepName "CTest: Full ZeroCopy Test Suite" -ScriptBlock {
    ctest --test-dir $buildDir -R "ZeroCopyTest" --output-on-failure
} -LogFile $fullZcLog -AllowFailure
if (-not $result) { $overallSuccess = $false }

# Step 7: DLL self-check
if (-not $SkipDllCheck) {
    $dllCheckLog = Join-Path $logDir "dll_check_${timestamp}.log"
    $result = Invoke-Step -StepName "DLL Self-Check" -ScriptBlock {
        python (Join-Path $projectRoot "scripts/check_windows_dll.py") --verbose
    } -LogFile $dllCheckLog -AllowFailure
    if (-not $result) { $overallSuccess = $false }
}

# ── summary ──

Write-Log "========================================" "Cyan"
Write-Log "  CI Pipeline Complete" "Cyan"
Write-Log "========================================" "Cyan"

$logFiles = @(
    @{Name="CMake Configure"; Path=$configureLog},
    @{Name="CMake Build"; Path=$buildLog},
    @{Name="COW Unit Tests"; Path=$cowUnitLog},
    @{Name="COW Integration Tests"; Path=$cowIntegLog},
    @{Name="Full ZeroCopy Tests"; Path=$fullZcLog},
    @{Name="DLL Self-Check"; Path=$dllCheckLog}
)

Write-Log ""
Write-Log "Log files:" "Cyan"
foreach ($lf in $logFiles) {
    if (Test-Path $lf.Path) {
        $size = (Get-Item $lf.Path).Length
        Write-Log "  $($lf.Name): $($lf.Path) ($size bytes)" "Gray"
    }
}

Write-Log ""
if ($overallSuccess) {
    Write-Log "  [PASS] All CI steps passed!" "Green"
} else {
    Write-Log "  [FAIL] Some CI steps failed. Check logs above." "Red"
}
Write-Log ""

exit $(if ($overallSuccess) { 0 } else { 1 })