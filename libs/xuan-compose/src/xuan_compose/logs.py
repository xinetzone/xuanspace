# SPDX-License-Identifier: GPL-2.0-only
"""日志流编排（翻译自 podman_compose.py 第 3959-3988、4383-4389 行）。

- ``create_format_logs_task``：为单个服务创建带颜色前缀的 ``podman logs``
  异步任务（上游位于命令层辅助区，按分层归属引擎日志编排）。
- ``_task_cancelled``：上游是 ``compose_up`` handler 内的无自由变量嵌套
  函数（仅依赖入参 ``task`` 与 ``sys``），分层后提为模块级函数，T7
  ``compose_up`` 移植时直接导入复用（T10 差异表登记）。
"""

import argparse
import asyncio
import sys
from typing import Any

from .translate.run_args import get_service_info

__all__ = ["create_format_logs_task", "_task_cancelled"]


def create_format_logs_task(
    compose: Any,
    args: argparse.Namespace,
    service: str,
    podman_args: list[Any],
    max_service_length: int,
) -> asyncio.Task[Any] | None:
    result = get_service_info(compose, service)
    if result is None:
        return None

    container, index = result
    # Add colored service prefix to output by piping output through sed
    if args.no_log_prefix:
        log_formatter = None
    else:
        color_idx = index % len(compose.console_colors)
        if args.no_color:  # monochrome output
            color = "\x1b[0m"
        else:
            color = compose.console_colors[color_idx]

        log_prefix = container["log_prefix"]
        space_suffix = " " * (max_service_length - len(log_prefix) + 1)
        log_formatter = f"{color}[{log_prefix}]{space_suffix}|\x1b[0m"

    target_service = compose.container_names_by_service[service]
    return asyncio.create_task(
        compose.podman.run([], "logs", podman_args + target_service, log_formatter=log_formatter)
    )


def _task_cancelled(task: asyncio.Task[Any]) -> bool:
    """等价于上游 compose_up 内的 ``_task_cancelled`` 判定。"""
    if task.cancelled():
        return True
    # Task.cancelling() is new in python 3.11
    if sys.version_info >= (3, 11) and task.cancelling():
        return True
    return False
