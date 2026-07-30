# Verify Build Script (PowerShell version)
# Usage: .\scripts\verify_build.ps1

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# ============================================================
# Step 0: Import vcvars64 environment into PowerShell
# ============================================================
Write-Host "============================================================"
Write-Host " Step 0: Import vcvars64 environment"
Write-Host "============================================================"

$VcvarsPath = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $VcvarsPath)) {
    # Fallback: try VS 2022 Community
    $VcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
}
if (-not (Test-Path $VcvarsPath)) {
    # Fallback: try VS 2022 Professional
    $VcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
}
if (-not (Test-Path $VcvarsPath)) {
    Write-Host "[ERROR] vcvars64.bat not found at any known location"
    Write-Host "Please open Developer Command Prompt for VS instead"
    exit 1
}

Write-Host "vcvars64.bat: $VcvarsPath"

# Capture vcvars64 environment by running it in cmd and dumping 'set'
Write-Host "Importing MSVC environment variables..."
$VcvarsOutput = cmd /c "call `"$VcvarsPath`" >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] vcvars64.bat failed (exit code: $LASTEXITCODE)"
    Write-Host "Try running this script from Developer Command Prompt for VS"
    exit 1
}

# Parse cmd 'set' output and apply to PowerShell environment
$VcvarsOutput | ForEach-Object {
    $line = $_.Trim()
    if ($line -match '^([^=]+)=(.*)$') {
        $name = $Matches[1]
        $value = $Matches[2]
        # Skip internal cmd variables
        if ($name -notin @('TMP', 'TEMP', 'PROMPT', 'HOMEDRIVE', 'HOMEPATH', 'USERNAME', 'USERDOMAIN', 'COMPUTERNAME', 'LOGONSERVER', 'SESSIONNAME', 'USERDNSDOMAIN', 'USERDOMAIN_ROAMINGPROFILE', 'APPDATA', 'LOCALAPPDATA', 'ALLUSERSPROFILE', 'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMDATA', 'PUBLIC', 'SYSTEMDRIVE', 'SYSTEMROOT', 'WINDIR', 'COMMONPROGRAMFILES', 'COMMONPROGRAMFILES(X86)', 'COMMONPROGRAMW6432', 'PROGRAMW6432', 'COMSPEC', 'PATHEXT', 'PROCESSOR_ARCHITECTURE', 'PROCESSOR_IDENTIFIER', 'PROCESSOR_LEVEL', 'PROCESSOR_REVISION', 'NUMBER_OF_PROCESSORS', 'OS', 'PATH', 'PSMODULEPATH', 'PWD', 'HOME', 'TERM', '__VSCMD_PREINIT_PATH', 'VSCMD_ARG_HOST_ARCH', 'VSCMD_ARG_TGT_ARCH', 'VSCMD_ARG_APP_PLAT', 'VSCMD_VER', 'VSCMD_START_DIR', 'VSCMD_ARG_NO_LOGO', 'VSCMD_ARG_CWD')) {
            [Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

# Prepend py314 to PATH (for runtime DLL loading)
# Discover py314 conda environment (no hardcoded paths)
$Py314Env = $null
# 1) Already activated?
if ($env:CONDA_PREFIX -and $env:CONDA_PREFIX -match 'py314') {
    $Py314Env = $env:CONDA_PREFIX
}
# 2) Scan common conda install locations
if (-not $Py314Env) {
    $Candidates = @(
        "$env:USERPROFILE\anaconda3\envs\py314",
        "$env:USERPROFILE\miniconda3\envs\py314",
        "$env:USERPROFILE\miniforge3\envs\py314",
        "C:\ProgramData\anaconda3\envs\py314",
        "C:\ProgramData\miniconda3\envs\py314"
    )
    foreach ($c in $Candidates) {
        if (Test-Path "$c\python.exe") { $Py314Env = $c; break }
    }
}
# 3) Last resort: search PATH
if (-not $Py314Env) {
    $Found = Get-Command python.exe -ErrorAction SilentlyContinue | Where-Object { $_.Source -match 'py314' }
    if ($Found) { $Py314Env = Split-Path -Parent $Found.Source }
}
if (-not $Py314Env) {
    Write-Host "[ERROR] Cannot find py314 conda environment."
    Write-Host "  Please activate py314 first: conda activate py314"
    exit 1
}
Write-Host "[DISCOVER] py314 conda environment: $Py314Env"

$Py314Paths = @(
    "$Py314Env",
    "$Py314Env\Scripts",
    "$Py314Env\Library\bin",
    "$Py314Env\DLLs",
    "$Py314Env\Lib\site-packages\tvm_ffi\lib"
)
$env:PATH = ($Py314Paths + $env:PATH) -join ';'

# Verify critical env vars
Write-Host ""
Write-Host "Environment Diagnostics:"
Write-Host "  CMake: $(Get-Command cmake.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Write-Host "  LIB set: $(if ($env:LIB) { 'YES' } else { 'NO' })"
Write-Host "  INCLUDE set: $(if ($env:INCLUDE) { 'YES' } else { 'NO' })"

$LibOk = $false
if ($env:LIB) {
    foreach ($dir in $env:LIB -split ';') {
        if (Test-Path "$dir\kernel32.lib") {
            Write-Host "  [OK] $dir\kernel32.lib"
            $LibOk = $true
            break
        }
    }
}
if (-not $LibOk) {
    Write-Host "  [ERROR] kernel32.lib not found in any LIB path"
    exit 1
}
Write-Host "  cl.exe: $(Get-Command cl.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)"
Write-Host "[OK] Environment ready"

# ============================================================
# Step 1: Clean CMake cache
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " Step 1: Clean CMake cache (force reconfigure)"
Write-Host "============================================================"
Remove-Item -Path build\CMakeCache.txt -Force -ErrorAction SilentlyContinue
Remove-Item -Path build\CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "[OK] Cache cleared"

# ============================================================
# Step 2: CMake configure
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " Step 2: CMake configure"
Write-Host "============================================================"
$CmakeArgs = @(
    '-S', '.',
    '-B', 'build',
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCAFFE_FFI_BUILD_TESTS=ON'
)
& cmake $CmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] CMake configure failed"
    Write-Host "Check: TVM_FFI_USE_BUILTIN_TYPETRAITS is defined in CompilerConfig.cmake"
    Write-Host "Check: /WX is set in MSVC compile options"
    exit 1
}
Write-Host "[OK] CMake configure succeeded"

# ============================================================
# Step 3: Build
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " Step 3: Build"
Write-Host "============================================================"
& cmake --build build --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Build failed"
    Write-Host "If warnings are treated as errors, check the compile output"
    exit 1
}
Write-Host "[OK] Build succeeded"

# ============================================================
# Step 4: Copy tvm_ffi.dll to build output directory
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " Step 4: Copy tvm_ffi.dll to build output directory"
Write-Host "============================================================"
$TvmFfiDll = Get-ChildItem -Path build -Filter tvm_ffi.dll -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.DirectoryName -ne (Resolve-Path build).Path } | Select-Object -First 1
if ($TvmFfiDll) {
    Copy-Item -Path $TvmFfiDll.FullName -Destination build\ -Force
    Write-Host "[OK] Copied $($TvmFfiDll.FullName) to build\"
} else {
    Write-Host "[WARN] tvm_ffi.dll not found in build tree"
}

# ============================================================
# Step 5: Run C++ unit tests
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " Step 5: Run C++ unit tests"
Write-Host "============================================================"
$TestExe = "build\caffe_ffi_tests.exe"
if (Test-Path $TestExe) {
    & $TestExe
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] All C++ tests passed"
    } else {
        Write-Host "[WARN] C++ tests had failures (exit code: $LASTEXITCODE)"
    }
} else {
    Write-Host "[ERROR] $TestExe not found"
}

# ============================================================
# Summary
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " Verification complete"
Write-Host "============================================================"
Write-Host ""
Write-Host "New CMake config applied:"
Write-Host "  - TVM_FFI_USE_BUILTIN_TYPETRAITS: prevents custom TypeTraits conflicts"
Write-Host "    (cmake/CompilerConfig.cmake line 67)"
Write-Host "  - /WX: treats warnings as errors on MSVC"
Write-Host "    (cmake/CompilerConfig.cmake line 89)"
Write-Host "  - -fvisibility=hidden: hides WEAK symbols on GCC/Clang"
Write-Host "    (cmake/CompilerConfig.cmake line 94)"
Write-Host "  - -Wl,--exclude-libs,ALL: excludes static lib symbols on GNU"
Write-Host "    (cmake/CompilerConfig.cmake line 119)"
Write-Host "  - CAFFE_FFI_API: explicit __declspec(dllexport) for data symbols"
Write-Host "    (include/caffe_ffi/common.hpp lines 15-24)"
Write-Host "  - CAFFE_FFI_EXPORTS: compile definition for DLL target"
Write-Host "    (cmake/TargetBuild.cmake line 49)"
Write-Host "  - CAFFE_FFI_ENABLE_COW: Phase 2 COW switch (default ON)"
Write-Host "    (cmake/Options.cmake line 12)"
Write-Host ""
Write-Host "New test files:"
Write-Host "  - tests/cpp/test_objectptr_migration.cpp (12 test cases)"
Write-Host "  - tests/cpp/test_symbol_export.cpp (8 test cases)"
Write-Host ""
Write-Host "Run verification: .\scripts\verify_build.ps1"