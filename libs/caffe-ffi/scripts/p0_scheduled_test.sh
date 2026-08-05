#!/usr/bin/env bash
# =============================================================================
# P0 环境定时任务调度包装脚本（宿主 WSL 侧）
#
# 用途：以"无重叠、可日志、可失败上报"的方式调度 scripts/run_p0_tests.sh，
#       供 cron / systemd timer / CI 定时任务重复调用。
#
# 特性：
#   1. 锁文件防并发：上一次运行未结束时跳过本次（避免定时任务重叠压垮容器）。
#   2. 时间戳日志：每次运行输出写入 logs/p0_run_<ts>.log。
#   3. 退出码语义：0=通过；1=测试失败；2=参数错误；3=被锁跳过；4=环境错误。
#
# 用法：
#   bash scripts/p0_scheduled_test.sh [args...]   # args 透传给 run_p0_tests.sh
#   示例：
#     bash scripts/p0_scheduled_test.sh                 # 预检+依赖检查+P2测试
#     bash scripts/p0_scheduled_test.sh --full          # 追加全量回归
#
# 定时任务接入（两种方式任选其一）：
#   A) cron（宿主 WSL 内 `crontab -e` 添加）：
#        0 3 * * * cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi \
#          && bash scripts/p0_scheduled_test.sh >> logs/p0_cron.log 2>&1
#   B) systemd timer（见脚本末尾注释的单元文件示例）
#
# ⚠️ 运行位置：本脚本须在【宿主 WSL】运行（同 run_p0_tests.sh），
#    通过 `docker exec` 进入 caffe-ffi-jupyter 容器执行测试。
# =============================================================================
set -u

# ── 路径解析：脚本所在目录的上一级即为 caffe-ffi 根目录 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAFFE_FFI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCK_FILE="${P0_LOCK_FILE:-${CAFFE_FFI_DIR}/.p0_test.lock}"
LOG_DIR="${CAFFE_FFI_DIR}/logs"
mkdir -p "$LOG_DIR"

# ── 参数：--lock-timeout <秒> 可自定义等待被锁的时间（默认 0=直接跳过） ──
LOCK_TIMEOUT=0
RUN_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --lock-timeout)
      LOCK_TIMEOUT="${2:?missing value for --lock-timeout}"
      shift 2
      ;;
    *)
      RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

# ── 锁：flock 原子获取（-w 超时，默认 0=立即尝试一次），防止定时任务重叠 ──
exec 9>"$LOCK_FILE"
if ! flock -w "$LOCK_TIMEOUT" 9; then
  echo "[P0-SCHED] 已有 P0 测试在运行（锁文件 $LOCK_FILE），本次跳过。exit=3"
  exit 3
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/p0_run_${TS}.log"
echo "[P0-SCHED] 开始 P0 环境测试（$(date '+%F %T')）"
echo "[P0-SCHED] caffe-ffi : $CAFFE_FFI_DIR"
echo "[P0-SCHED] 日志文件 : $LOG_FILE"
echo "[P0-SCHED] 参数     : ${RUN_ARGS[*]:-（无）}"

# ── 派生独立进程执行并机外记录日志 ──
(
  cd "$CAFFE_FFI_DIR" || exit 4
  bash scripts/run_p0_tests.sh "${RUN_ARGS[@]}"
) >"$LOG_FILE" 2>&1
RC=$?

# ── 输出最近日志尾部，便于终端/CI 快速定位 ──
echo "----- 运行日志尾部 -----"
tail -n 30 "$LOG_FILE"
echo "------------------------"

if [ "$RC" -ne 0 ]; then
  echo "[P0-SCHED] 测试失败 exit=$RC，完整日志见 $LOG_FILE"
  exit "$RC"
fi
echo "[P0-SCHED] 测试通过 exit=0，日志 $LOG_FILE"
exit 0

# =============================================================================
# systemd timer 单元文件示例（可选方式 B）
#
# /etc/systemd/system/p0-test.service
#   [Unit]
#   Description=P0 caffe-ffi scheduled test
#
#   [Service]
#   ExecStart=/bin/bash /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi/scripts/p0_scheduled_test.sh --full
#   Type=oneshot
#
# /etc/systemd/system/p0-test.timer
#   [Unit]
#   Description=Run P0 caffe-ffi test daily at 03:00
#
#   [Timer]
#   OnCalendar=*-*-* 03:00:00
#   Persistent=true
#
#   [Install]
#   WantedBy=timers.target
#
# 启用：
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now p0-test.timer
#   systemctl list-timers | grep p0-test
# =============================================================================