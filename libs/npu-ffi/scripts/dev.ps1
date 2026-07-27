#Requires -Version 5.1
<#
.SYNOPSIS
    NPU-FFI local development environment setup script for Windows.
.DESCRIPTION
    Automates tvm-ffi dependency setup, CMake build, pip install, and verification.
.PARAMETER Build
    Only build C++ code.
.PARAMETER Install
    Only install pip package (editable mode).
.PARAMETER Test
    Run pytest.
.PARAMETER Clean
    Clean build directory.
.PARAMETER Rebuild
    Clean + rebuild + reinstall.
.EXAMPLE
    .\scripts\dev.ps1
    Build C++, install package, and verify import.
.EXAMPLE
    .\scripts\dev.ps1 -Build
    Only build C++ code.
.EXAMPLE
    .\scripts\dev.ps1 -Test
    Run pytest.
#>

param(
    [switch]$Build,
    [switch]$Install,
    [switch]$Test,
    [switch]$Clean,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$tvmFfiPath = Join-Path $projectRoot "..\..\vendor\tvm-ffi"
$buildDir = Join-Path $projectRoot "build"
$buildLibDir = Join-Path $buildDir "lib"
$buildVtaReleaseDir = Join-Path $buildDir "src\vta\Release"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Write-Step {
    param([string]$Message)
    Write-Host "==> " -ForegroundColor Cyan -NoNewline
    Write-Host $Message
}

function Write-Success {
    param([string]$Message)
    Write-Host "    [PASS] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Error2 {
    param([string]$Message)
    Write-Host "    [FAIL] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Warn {
    param([string]$Message)
    Write-Host "    [WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Test-PythonModule {
    param([string]$ModuleName)
    try {
        $null = python -c "import $ModuleName; print('ok')" 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Install-TvmFfi {
    Write-Step "Checking tvm-ffi installation..."
    if (Test-PythonModule "tvm_ffi") {
        Write-Success "tvm-ffi already installed"
    } else {
        if (-not (Test-Path $tvmFfiPath)) {
            Write-Error2 "tvm-ffi not found at $tvmFfiPath"
            Write-Host "    Please clone tvm-ffi to vendor/tvm-ffi first"
            exit 1
        }
        Write-Step "Installing tvm-ffi from $tvmFfiPath..."
        pip install --no-build-isolation -e $tvmFfiPath
        if ($LASTEXITCODE -ne 0) {
            Write-Error2 "Failed to install tvm-ffi"
            exit 1
        }
        Write-Success "tvm-ffi installed"
    }
}

function Build-Cpp {
    Write-Step "Configuring and building C++ code..."
    if (-not (Test-Path $buildDir)) {
        Write-Step "Running CMake configure..."
        Push-Location $projectRoot
        cmake -B build -G Ninja -DNPU_FFI_VTA_USE_STUB=ON
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            Write-Error2 "CMake configure failed"
            exit 1
        }
        Pop-Location
        Write-Success "CMake configured"
    }
    Write-Step "Running CMake build (Release)..."
    Push-Location $projectRoot
    cmake --build build --config Release
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error2 "CMake build failed"
        exit 1
    }
    Pop-Location
    Write-Success "C++ build complete"
}

function Install-Package {
    Write-Step "Checking if npu-ffi is already installed..."
    if (Test-PythonModule "npu_ffi") {
        Write-Success "npu-ffi already installed in editable mode"
    } else {
        Write-Step "Installing npu-ffi (editable, no-build-isolation)..."
        Push-Location $projectRoot
        pip install --no-build-isolation -e .
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            Write-Error2 "pip install failed"
            exit 1
        }
        Pop-Location
        Write-Success "npu-ffi installed"
    }
}

function Add-DllSearchPaths {
    Write-Step "Adding DLL search paths..."
    $dllPaths = @()
    if (Test-Path $buildLibDir) { $dllPaths += $buildLibDir }
    if (Test-Path $buildVtaReleaseDir) { $dllPaths += $buildVtaReleaseDir }
    if (Test-Path (Join-Path $buildDir "src\vta")) { $dllPaths += (Join-Path $buildDir "src\vta") }
    
    if ($dllPaths.Count -gt 0) {
        foreach ($path in $dllPaths) {
            $env:PATH = "$path;$env:PATH"
            Write-Success "Added: $path"
        }
        python -c @"
import os
import sys
build_lib = r'$buildLibDir'
build_vta_release = r'$buildVtaReleaseDir'
build_vta = r'$($buildDir)\src\vta'
for p in [build_lib, build_vta_release, build_vta]:
    if os.path.exists(p):
        try:
            os.add_dll_directory(p)
        except (OSError, AttributeError):
            pass
print('DLL paths configured')
"@
    } else {
        Write-Warn "No build directories found - skipping DLL path setup"
    }
}

function Verify-Import {
    Write-Step "Verifying npu_ffi import..."
    $verifyCode = @"
import os
import sys
build_lib = r'$buildLibDir'
build_vta_release = r'$buildVtaReleaseDir'
build_vta = r'$($buildDir)\src\vta'
for p in [build_lib, build_vta_release, build_vta]:
    if os.path.exists(p):
        try:
            os.add_dll_directory(p)
        except (OSError, AttributeError):
            pass
try:
    import npu_ffi
    from npu_ffi import vta
    print('npu_ffi imported successfully')
    print(f'vta module available: {vta is not None}')
    cmd = vta.tls_command_handle()
    print(f'tls_command_handle returned: {cmd}')
    vta.runtime_shutdown()
except Exception as e:
    print(f'Import failed: {e}', file=sys.stderr)
    sys.exit(1)
"@
    python -c $verifyCode
    if ($LASTEXITCODE -eq 0) {
        Write-Success "npu_ffi import and basic test passed"
    } else {
        Write-Error2 "npu_ffi import verification failed"
        exit 1
    }
}

function Run-Tests {
    Write-Step "Running pytest..."
    Push-Location $projectRoot
    pytest tests/python -v
    $testExit = $LASTEXITCODE
    Pop-Location
    if ($testExit -eq 0) {
        Write-Success "All tests passed"
    } else {
        Write-Error2 "Some tests failed"
        exit $testExit
    }
}

function Clean-Build {
    Write-Step "Cleaning build directory..."
    if (Test-Path $buildDir) {
        Remove-Item -Recurse -Force $buildDir
        Write-Success "Build directory removed"
    } else {
        Write-Warn "No build directory to clean"
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NPU-FFI Development Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project root: $projectRoot"
Write-Host ""

if ($Clean) {
    Clean-Build
    exit 0
}

if ($Rebuild) {
    Clean-Build
    Install-TvmFfi
    Build-Cpp
    Install-Package
    Add-DllSearchPaths
    Verify-Import
    exit 0
}

if ($Build) {
    Install-TvmFfi
    Build-Cpp
    exit 0
}

if ($Install) {
    Install-TvmFfi
    Install-Package
    exit 0
}

if ($Test) {
    Add-DllSearchPaths
    Run-Tests
    exit 0
}

Install-TvmFfi
Build-Cpp
Install-Package
Add-DllSearchPaths
Verify-Import

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor Yellow
Write-Host "  -Build    Only build C++"
Write-Host "  -Install  Only install pip package"
Write-Host "  -Test     Run pytest"
Write-Host "  -Clean    Clean build"
Write-Host "  -Rebuild  Clean + rebuild + install"
Write-Host ""
