#!/usr/bin/env bash
# =============================================================================
# cpu_mutable_data() 批量迁移脚本
# 将 layer 源文件中所有 top[i]->cpu_data() 写入调用替换为 cpu_mutable_data()
#
# 安全设计：
#   - 仅替换 top[0]->cpu_data() 和 top[1]->cpu_data()，不触碰 bottom/内部 blob
#   - 使用 --dry-run 预览变更，确认后去掉 --dry-run 实际执行
#   - 支持反向回滚（见下方注释）
#
# 用法：
#   bash scripts/migrate_cpu_mutable_data.sh --dry-run    # 预览变更
#   bash scripts/migrate_cpu_mutable_data.sh               # 执行迁移
#   bash scripts/migrate_cpu_mutable_data.sh --reverse     # 回滚迁移
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LAYERS_DIR="$PROJECT_ROOT/src/caffe_ffi/layers"

DRY_RUN=false
REVERSE=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --reverse) REVERSE=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# 21 处迁移点（按文件分组）
# 格式: "文件:行号:旧模式:新模式:风险说明"
declare -a MIGRATIONS=(
    # ── In-place layers (9 处) — 高优先级，直接影响 COW 正确性 ──
    "relu_layer.cpp:36:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place ReLU 写入 top"
    "dropout_layer.cpp:36:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place Dropout 写入 top"
    "elu_layer.cpp:43:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place ELU 写入 top"
    "sigmoid_layer.cpp:37:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place Sigmoid 写入 top"
    "tanh_layer.cpp:37:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place Tanh 写入 top"
    "prelu_layer.cpp:91:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place PReLU 写入 top"
    "bias_layer.cpp:91:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place Bias 写入 top"
    "scale_layer.cpp:102:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place Scale 写入 top"
    "batch_norm_layer.cpp:89:top[0]->cpu_data():top[0]->cpu_mutable_data():In-place BatchNorm 写入 top"

    # ── Non-in-place layers (12 处) — 低优先级，use_count==1 时 cpu_mutable_data() 是空操作 ──
    "conv_layer.cpp:155:top[0]->cpu_data():top[0]->cpu_mutable_data():Conv 写入 top (non-in-place)"
    "inner_product_layer.cpp:102:top[0]->cpu_data():top[0]->cpu_mutable_data():InnerProduct 写入 top (non-in-place)"
    "pooling_layer.cpp:123:top[0]->cpu_data():top[0]->cpu_mutable_data():Pooling 写入 top (non-in-place)"
    "concat_layer.cpp:79:top[0]->cpu_data():top[0]->cpu_mutable_data():Concat 写入 top (non-in-place)"
    "eltwise_layer.cpp:87:top[0]->cpu_data():top[0]->cpu_mutable_data():Eltwise 写入 top (non-in-place)"
    "softmax_layer.cpp:57:top[0]->cpu_data():top[0]->cpu_mutable_data():Softmax 写入 top (non-in-place)"
    "flatten_layer.cpp:53:top[0]->cpu_data():top[0]->cpu_mutable_data():Flatten caffe_copy 写入 top (non-in-place)"
    "reshape_layer.cpp:121:top[0]->cpu_data():top[0]->cpu_mutable_data():Reshape 写入 top (non-in-place)"
    "accuracy_layer.cpp:67:top[1]->cpu_data():top[1]->cpu_mutable_data():Accuracy caffe_set 写入 top[1] (non-in-place)"
    "accuracy_layer.cpp:82:top[0]->cpu_data():top[0]->cpu_mutable_data():Accuracy caffe_set 写入 top[0] (non-in-place)"
    "softmax_loss_layer.cpp:120:top[0]->cpu_data():top[0]->cpu_mutable_data():SoftmaxLoss 写入 top[0] (non-in-place)"
    "softmax_loss_layer.cpp:184:top[1]->cpu_data():top[1]->cpu_mutable_data():SoftmaxLoss caffe_copy 写入 top[1] (non-in-place)"
)

echo "================================================"
echo "  cpu_mutable_data() Migration Script"
echo "================================================"
echo "Project: $PROJECT_ROOT"
echo "Mode:    $(if $DRY_RUN; then echo 'DRY-RUN (preview only)'; elif $REVERSE; then echo 'REVERSE (rollback)'; else echo 'EXECUTE'; fi)"
echo "Target:  ${#MIGRATIONS[@]} locations in $LAYERS_DIR"
echo ""

changed=0
errors=0

for entry in "${MIGRATIONS[@]}"; do
    IFS=':' read -r file line old new desc <<< "$entry"
    fpath="$LAYERS_DIR/$file"

    if [ ! -f "$fpath" ]; then
        echo "[ERROR] File not found: $fpath"
        ((errors++))
        continue
    fi

    if $REVERSE; then
        # Reverse: cpu_mutable_data() → cpu_data()
        pattern="$new"
        replacement="$old"
    else
        pattern="$old"
        replacement="$new"
    fi

    # Check if the line contains the expected pattern
    actual_line=$(sed -n "${line}p" "$fpath")
    if echo "$actual_line" | grep -qF "$old"; then
        if $DRY_RUN; then
            echo "[DRY-RUN] $file:$line"
            echo "  OLD: $actual_line"
            echo "  NEW: $(echo "$actual_line" | sed "s/$old/$replacement/")"
            echo "  DESC: $desc"
            echo ""
        else
            # Perform the replacement on the specific line
            sed -i "${line}s/$old/$replacement/" "$fpath"
            echo "[OK] $file:$line — $desc"
            ((changed++))
        fi
    elif echo "$actual_line" | grep -qF "$new"; then
        if $REVERSE; then
            # Already in original state
            echo "[SKIP] $file:$line — already reverted"
        else
            echo "[SKIP] $file:$line — already migrated"
        fi
    else
        echo "[WARN] $file:$line — line content doesn't match expected pattern"
        echo "  Expected to find: $old"
        echo "  Actual line:      $actual_line"
        ((errors++))
    fi
done

echo ""
echo "================================================"
echo "  Summary"
echo "================================================"
echo "  Total locations: ${#MIGRATIONS[@]}"
echo "  Changed:         $changed"
echo "  Errors:          $errors"
echo ""

if $DRY_RUN; then
    echo "  DRY-RUN complete. To execute, run without --dry-run."
    echo "  To rollback after execution, run with --reverse."
elif $REVERSE; then
    echo "  Rollback complete."
else
    echo "  Migration complete. Verify with: git diff src/caffe_ffi/layers/"
    echo "  To rollback, run: bash scripts/migrate_cpu_mutable_data.sh --reverse"
fi
echo ""