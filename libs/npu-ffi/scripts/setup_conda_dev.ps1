# 设置Conda开发环境（Windows）
param([string]$EnvName = "npu-ffi-dev")

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VendorRoot = Resolve-Path (Join-Path $ProjectRoot "..\..")
$VendorRoot = Join-Path $VendorRoot "vendor"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up npu-ffi Conda dev environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Environment name: $EnvName"
Write-Host "Project root: $ProjectRoot"
Write-Host "Vendor root: $VendorRoot"

Set-Location $ProjectRoot

Write-Host ""
Write-Host "[1/4] Creating conda environment..." -ForegroundColor Yellow
conda env create -f environment.yml -n $EnvName --force

Write-Host ""
Write-Host "[2/4] Activating environment..." -ForegroundColor Yellow
conda activate $EnvName

Write-Host ""
Write-Host "[3/4] Installing tvm-ffi (editable)..." -ForegroundColor Yellow
$TvmFfiPath = Join-Path $VendorRoot "tvm-ffi"
if (Test-Path $TvmFfiPath) {
    Write-Host "Found local tvm-ffi at $TvmFfiPath"
    pip install --no-build-isolation -e $TvmFfiPath
} else {
    Write-Host "Local tvm-ffi not found, installing from PyPI..."
    pip install apache-tvm-ffi
}

Write-Host ""
Write-Host "[4/4] Installing npu-ffi (editable)..." -ForegroundColor Yellow
pip install --no-build-isolation -e .

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Development environment setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Activate with: conda activate $EnvName"
Write-Host ""
Write-Host "Run tests with:"
Write-Host "  `$env:KMP_DUPLICATE_LIB_OK='TRUE'  # Windows"
Write-Host "  pytest tests/python -v"
Write-Host ""
