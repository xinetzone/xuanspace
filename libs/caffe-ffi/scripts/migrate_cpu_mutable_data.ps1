#Requires -Version 5.1
<#
.SYNOPSIS
    cpu_mutable_data() 批量迁移脚本 (PowerShell 版)
    将 layer 源文件中所有 top[i]->cpu_data() 写入调用替换为 cpu_mutable_data()
.DESCRIPTION
    安全设计：
      - 仅替换 top[0]->cpu_data() 和 top[1]->cpu_data()，不触碰 bottom/内部 blob
      - 使用 -DryRun 预览变更，确认后去掉 -DryRun 实际执行
      - 支持 -Reverse 回滚
.PARAMETER DryRun
    预览变更，不实际修改文件。
.PARAMETER Reverse
    回滚迁移：cpu_mutable_data() → cpu_data()。
.EXAMPLE
    .\scripts\migrate_cpu_mutable_data.ps1 -DryRun
    预览 21 处变更。
.EXAMPLE
    .\scripts\migrate_cpu_mutable_data.ps1
    执行迁移。
.EXAMPLE
    .\scripts\migrate_cpu_mutable_data.ps1 -Reverse
    回滚迁移。
#>

param(
    [switch]$DryRun,
    [switch]$Reverse
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$layersDir = Join-Path $projectRoot "src\caffe_ffi\layers"

# 21 处迁移点（按文件分组）
$migrations = @(
    # ── In-place layers (9) — 高优先级 ──
    [PSCustomObject]@{File="relu_layer.cpp"; Line=36; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place ReLU"}
    [PSCustomObject]@{File="dropout_layer.cpp"; Line=36; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place Dropout"}
    [PSCustomObject]@{File="elu_layer.cpp"; Line=43; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place ELU"}
    [PSCustomObject]@{File="sigmoid_layer.cpp"; Line=37; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place Sigmoid"}
    [PSCustomObject]@{File="tanh_layer.cpp"; Line=37; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place Tanh"}
    [PSCustomObject]@{File="prelu_layer.cpp"; Line=91; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place PReLU"}
    [PSCustomObject]@{File="bias_layer.cpp"; Line=91; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place Bias"}
    [PSCustomObject]@{File="scale_layer.cpp"; Line=102; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place Scale"}
    [PSCustomObject]@{File="batch_norm_layer.cpp"; Line=89; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="In-place BatchNorm"}

    # ── Non-in-place layers (12) — 低优先级 ──
    [PSCustomObject]@{File="conv_layer.cpp"; Line=155; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Conv (non-in-place)"}
    [PSCustomObject]@{File="inner_product_layer.cpp"; Line=102; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="InnerProduct (non-in-place)"}
    [PSCustomObject]@{File="pooling_layer.cpp"; Line=123; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Pooling (non-in-place)"}
    [PSCustomObject]@{File="concat_layer.cpp"; Line=79; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Concat (non-in-place)"}
    [PSCustomObject]@{File="eltwise_layer.cpp"; Line=87; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Eltwise (non-in-place)"}
    [PSCustomObject]@{File="softmax_layer.cpp"; Line=57; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Softmax (non-in-place)"}
    [PSCustomObject]@{File="flatten_layer.cpp"; Line=53; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Flatten caffe_copy (non-in-place)"}
    [PSCustomObject]@{File="reshape_layer.cpp"; Line=121; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Reshape (non-in-place)"}
    [PSCustomObject]@{File="accuracy_layer.cpp"; Line=67; Old="top[1]->cpu_data()"; New="top[1]->cpu_mutable_data()"; Desc="Accuracy caffe_set top[1] (non-in-place)"}
    [PSCustomObject]@{File="accuracy_layer.cpp"; Line=82; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="Accuracy caffe_set top[0] (non-in-place)"}
    [PSCustomObject]@{File="softmax_loss_layer.cpp"; Line=120; Old="top[0]->cpu_data()"; New="top[0]->cpu_mutable_data()"; Desc="SoftmaxLoss top[0] (non-in-place)"}
    [PSCustomObject]@{File="softmax_loss_layer.cpp"; Line=184; Old="top[1]->cpu_data()"; New="top[1]->cpu_mutable_data()"; Desc="SoftmaxLoss caffe_copy top[1] (non-in-place)"}
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  cpu_mutable_data() Migration Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
$mode = if ($DryRun) { "DRY-RUN (preview)" } elseif ($Reverse) { "REVERSE (rollback)" } else { "EXECUTE" }
Write-Host "Project: $projectRoot"
Write-Host "Mode:    $mode"
Write-Host "Target:  $($migrations.Count) locations"
Write-Host ""

$changed = 0
$errors = 0

foreach ($m in $migrations) {
    $fpath = Join-Path $layersDir $m.File

    if (-not (Test-Path $fpath)) {
        Write-Host "[ERROR] File not found: $fpath" -ForegroundColor Red
        $errors++
        continue
    }

    $lines = Get-Content $fpath
    $actualLine = $lines[$m.Line - 1]

    if ($Reverse) {
        $pattern = $m.New
        $replacement = $m.Old
    } else {
        $pattern = $m.Old
        $replacement = $m.New
    }

    if ($actualLine -match [regex]::Escape($pattern)) {
        if ($DryRun) {
            Write-Host "[DRY-RUN] $($m.File):$($m.Line)" -ForegroundColor Yellow
            Write-Host "  OLD:  $actualLine"
            Write-Host "  NEW:  $($actualLine -replace [regex]::Escape($pattern), $replacement)"
            Write-Host "  DESC: $($m.Desc)"
            Write-Host ""
        } else {
            $lines[$m.Line - 1] = $actualLine -replace [regex]::Escape($pattern), $replacement
            Set-Content -Path $fpath -Value $lines -Encoding UTF8
            Write-Host "[OK] $($m.File):$($m.Line) — $($m.Desc)" -ForegroundColor Green
            $changed++
        }
    } elseif ($actualLine -match [regex]::Escape($replacement)) {
        if ($Reverse) {
            Write-Host "[SKIP] $($m.File):$($m.Line) — already reverted" -ForegroundColor Gray
        } else {
            Write-Host "[SKIP] $($m.File):$($m.Line) — already migrated" -ForegroundColor Gray
        }
    } else {
        Write-Host "[WARN] $($m.File):$($m.Line) — line content mismatch" -ForegroundColor Yellow
        Write-Host "  Expected: $pattern"
        Write-Host "  Actual:   $actualLine"
        $errors++
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Total:   $($migrations.Count)"
Write-Host "  Changed: $changed"
Write-Host "  Errors:  $errors"
Write-Host ""

if ($DryRun) {
    Write-Host "  DRY-RUN complete. Run without -DryRun to execute." -ForegroundColor Yellow
    Write-Host "  Rollback: .\scripts\migrate_cpu_mutable_data.ps1 -Reverse" -ForegroundColor Yellow
} elseif ($Reverse) {
    Write-Host "  Rollback complete." -ForegroundColor Green
} else {
    Write-Host "  Migration complete. Verify: git diff src/caffe_ffi/layers/" -ForegroundColor Green
    Write-Host "  Rollback: .\scripts\migrate_cpu_mutable_data.ps1 -Reverse" -ForegroundColor Yellow
}
Write-Host ""

exit $(if ($errors -gt 0) { 1 } else { 0 })