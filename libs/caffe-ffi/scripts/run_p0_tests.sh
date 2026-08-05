#!/usr/bin/env bash
# =============================================================================
# P0 环境测试自动化执行脚本（WSL Docker: caffe-ffi-jupyter）
#
# 用途：在 P0 环境（caffe-ffi-jupyter 容器）中一键执行剩余运行时测试，包含：
#   1. 环境预检        —— 容器在线、conda python、apache-tvm-ffi 可导入
#   2. TVM-FFI 依赖加载检查 —— scripts/p0_check_tvmffi.py（core 扩展 + libtvm_ffi.so 解析 + 全链路冒烟）
#   3. P2 数据/损失算子单元测试 —— tests/python/test_p2*.py（Task 7 阶段补充）
#   4. 全量回归测试    —— tests/python/ 整个 pytest 套件（可选，--full）
#
# 用法：
#   bash scripts/run_p0_tests.sh                 # 预检 + TVM-FFI 检查 + P2 测试
#   bash scripts/run_p0_tests.sh --full          # 追加全量回归
#   bash scripts/run_p0_tests.sh --smoke-only    # 仅 TVM-FFI 检查（含冒烟）
#   bash scripts/run_p0_tests.sh --no-preflight  # 跳过预检
#
# ⚠️ 运行位置：本脚本须在【宿主 WSL】（宿主机 `wsl` 环境）中运行，而非容器内部。
#   因为 `docker` 命令只在宿主可用；脚本通过 `docker exec` 进入容器执行测试。
#   宿主路径示例：`wsl -e sh -lc "cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi && bash scripts/run_p0_tests.sh"`
#
# 退出码：0 = 全部通过；1 = 任一阶段失败
# =============================================================================
set -u

# ── 可配置默认值 ────────────────────────────────────────────────────────────
CONTAINER="${P0_CONTAINER:-caffe-ffi-jupyter}"
PYTHON="${P0_PYTHON:-/opt/conda/envs/caffe-ffi/bin/python}"
CAFFE_FFI_DIR="/SpecWeave/projects/xuanspace/libs/caffe-ffi"

RUN_FULL=0
RUN_SMOKE=1
RUN_PREFLIGHT=1

for arg in "$@"; do
  case "$arg" in
    --full)       RUN_FULL=1 ;;
    --smoke-only) RUN_SMOKE=1; RUN_FULL=0 ;;
    --no-preflight) RUN_PREFLIGHT=0 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

PASS_CNT=0
FAIL_CNT=0

step_ok()  { PASS_CNT=$((PASS_CNT+1)); echo "  [PASS] $1"; }
step_fail(){ FAIL_CNT=$((FAIL_CNT+1)); echo "  [FAIL] $1"; }

# ── 在容器内执行一条命令（统一包装，避免引号/转义问题）──────────────────────
run_in_container() {
  docker exec "$CONTAINER" bash -lc "$1" 2>&1
}

echo "=============================================================="
echo " P0 环境测试自动化执行"
echo " 容器     : $CONTAINER"
echo " Python   : $PYTHON"
echo " caffe-ffi: $CAFFE_FFI_DIR"
echo "=============================================================="

# ── 1. 环境预检 ────────────────────────────────────────────────────────────
if [ "$RUN_PREFLIGHT" -eq 1 ]; then
  echo ""
  echo "[1] 环境预检"
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    step_ok "容器 $CONTAINER 在线"
  else
    step_fail "容器 $CONTAINER 不在线（请先启动容器）"
    echo "  提示: 参考 apps/caffe-ffi-jupyter/scripts/build.sh 构建并启动容器"
    exit 1
  fi

  if run_in_container "$PYTHON -c 'import sys; print(sys.version.split()[0])'" \
       | grep -qE '^3\.1[4-9]'; then
    step_ok "Python 版本 >= 3.14"
  else
    step_fail "Python 版本不符合要求（需 >=3.14）"
  fi

  if run_in_container "$PYTHON -c 'import tvm_ffi; print(tvm_ffi.__version__)'" \
       | grep -qE '^[0-9]'; then
    step_ok "apache-tvm-ffi 可导入（Cython core 扩展已构建）"
  else
    step_fail "apache-tvm-ffi 导入失败（Cython core 扩展缺失，需重建 tvm-ffi）"
  fi
fi

# ── 2. TVM-FFI 依赖加载检查（vendored libtvm_ffi.so 解析）─────────────────
if [ "$RUN_SMOKE" -eq 1 ]; then
  echo ""
  echo "[2] TVM-FFI 依赖加载检查 (scripts/p0_check_tvmffi.py)"
  if run_in_container "cd $CAFFE_FFI_DIR && $PYTHON scripts/p0_check_tvmffi.py"; then
    step_ok "TVM-FFI 依赖加载检查通过（core 扩展 + libtvm_ffi.so 解析 + 全链路冒烟）"
  else
    step_fail "TVM-FFI 依赖加载检查失败"
  fi
fi

# ── 3. P2 数据/损失算子单元测试（Task 7 阶段补充）──────────────────────────
echo ""
echo "[3] P2 算子单元测试 (tests/python/test_p2*.py)"
if run_in_container "ls $CAFFE_FFI_DIR/tests/python/test_p2"*.py >/dev/null 2>&1; then
  if run_in_container "cd $CAFFE_FFI_DIR && $PYTHON -m pytest tests/python/test_p2*.py"; then
    step_ok "P2 算子单元测试通过"
  else
    step_fail "P2 算子单元测试失败"
  fi
else
  echo "  [SKIP] 未发现 test_p2*.py（Task 7 单元测试待补充，见 .trae/specs/caffe-ffi-p2-ops-implementation/tasks.md）"
fi

# ── 4. 全量回归测试（可选）─────────────────────────────────────────────────
if [ "$RUN_FULL" -eq 1 ]; then
  echo ""
  echo "[4] 全量回归测试 (tests/python/)"
  if run_in_container "cd $CAFFE_FFI_DIR && $PYTHON -m pytest tests/python"; then
    step_ok "全量回归测试通过"
  else
    step_fail "全量回归测试失败（存在失败用例，需逐个排查）"
  fi
fi

# ── 汇总 ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================================="
echo " 汇总: PASS=$PASS_CNT  FAIL=$FAIL_CNT"
echo "=============================================================="
if [ "$FAIL_CNT" -gt 0 ]; then
  echo "存在失败项，请根据上方 [FAIL] 定位问题。"
  exit 1
fi
echo "全部通过。"
exit 0