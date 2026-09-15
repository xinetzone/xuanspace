# SPDX-License-Identifier: GPL-2.0-only
"""``logs`` 命令（翻译自 podman_compose.py 第 4759-4792 行）。

日志流的任务构造/格式化在引擎层 ``xuan_compose.logs``（T6），本 handler
只负责 podman 参数组装与逐服务并发收集。注意上游 ``max(len(service)
for service in args.services)`` 对空序列会抛 ``ValueError``——逐行保留
该行为（正常 CLI 路径 args.services 必非空，T8 parser 保证）。
"""

import argparse
import asyncio
from typing import Any

from ..logs import create_format_logs_task

__all__ = ["compose_logs"]


async def compose_logs(compose: Any, args: argparse.Namespace) -> None:
    container_names_by_service = compose.container_names_by_service
    if not args.services and not args.latest:
        args.services = container_names_by_service.keys()
    compose.assert_services(args.services)

    podman_args = []
    if args.follow:
        podman_args.append("-f")
    if args.latest:
        podman_args.append("-l")
    if args.names:
        podman_args.append("-n")
    if not args.no_color:
        podman_args.append("--color")
    if args.since:
        podman_args.extend(["--since", args.since])
    # the default value is to print all logs which is in podman = 0 and not
    # needed to be passed
    if args.tail and args.tail != "all":
        podman_args.extend(["--tail", args.tail])
    if args.timestamps:
        podman_args.append("-t")
    if args.until:
        podman_args.extend(["--until", args.until])

    tasks: list[asyncio.Task[Any]] = []
    max_service_length = max(len(service) for service in args.services)
    for service in args.services:
        task = create_format_logs_task(compose, args, service, podman_args, max_service_length)
        if task:
            tasks.append(task)
    await asyncio.gather(*tasks)
