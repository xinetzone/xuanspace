# SPDX-License-Identifier: GPL-2.0-only
"""生命周期命令（翻译自 podman_compose.py 第 3400-3408、4708-4737、
4740-4755、4817-4868 行）：wait/start/stop/restart/pause/unpause/kill。

注意 ``compose_wait`` 上游调用的是同步 ``Podman.exec``（内部 ``os.execlp``
替换进程映像，不返回），因此逐行保留为无 ``await`` 调用。
"""

import argparse
import asyncio
import sys
from typing import Any

from ..compat import STOP_GRACE_PERIOD, str_to_seconds
from ..logging_utils import log
from .updown import wait_for_container_running_healthy

__all__ = [
    "compose_wait",
    "transfer_service_status",
    "compose_start",
    "compose_stop",
    "compose_restart",
    "compose_pause",
    "compose_unpause",
    "compose_kill",
]


async def compose_wait(
    compose: Any,
    args: argparse.Namespace,  # pylint: disable=unused-argument
) -> None:
    containers = [cnt["name"] for cnt in compose.containers]
    cmd_args = ["--"]
    cmd_args.extend(containers)
    compose.podman.exec([], "wait", cmd_args)


async def transfer_service_status(
    compose: Any, args: argparse.Namespace, action: str
) -> None:
    # TODO: handle dependencies, handle creations
    container_names_by_service = compose.container_names_by_service
    if not args.services:
        args.services = container_names_by_service.keys()
    compose.assert_services(args.services)
    targets = []
    for service in args.services:
        if service not in container_names_by_service:
            raise ValueError("unknown service: " + service)
        targets.extend(container_names_by_service[service])
    if action in ["stop", "restart"]:
        targets = list(reversed(targets))
    timeout_global = getattr(args, "timeout", None)
    tasks = []
    for target in targets:
        podman_args = []
        if action != "start":
            timeout = timeout_global
            if timeout is None:
                timeout_str = compose.container_by_name[target].get(
                    "stop_grace_period", STOP_GRACE_PERIOD
                )
                timeout = str_to_seconds(timeout_str)
            if timeout is not None:
                podman_args.extend(["-t", str(timeout)])
        tasks.append(asyncio.create_task(compose.podman.run([], action, podman_args + [target])))
    await asyncio.gather(*tasks)


async def compose_start(compose: Any, args: argparse.Namespace) -> None:
    await transfer_service_status(compose, args, "start")

    if args.wait:
        await wait_for_container_running_healthy(compose, args)


async def compose_stop(compose: Any, args: argparse.Namespace) -> None:
    await transfer_service_status(compose, args, "stop")


async def compose_restart(compose: Any, args: argparse.Namespace) -> None:
    await transfer_service_status(compose, args, "restart")


async def compose_pause(compose: Any, args: argparse.Namespace) -> None:
    container_names_by_service = compose.container_names_by_service
    if not args.services:
        args.services = container_names_by_service.keys()
    targets = []
    for service in args.services:
        targets.extend(container_names_by_service[service])
    await compose.podman.run([], "pause", targets)


async def compose_unpause(compose: Any, args: argparse.Namespace) -> None:
    container_names_by_service = compose.container_names_by_service
    if not args.services:
        args.services = container_names_by_service.keys()
    targets = []
    for service in args.services:
        targets.extend(container_names_by_service[service])
    await compose.podman.run([], "unpause", targets)


async def compose_kill(compose: Any, args: argparse.Namespace) -> None:
    # to ensure that the user did not execute the command by mistake
    if not args.services and not args.all:
        log.fatal(
            "Error: you must provide at least one service name or use (--all) to kill all services"
        )
        sys.exit()

    container_names_by_service = compose.container_names_by_service
    podman_args = []

    if args.signal:
        podman_args.extend(["--signal", args.signal])

    if args.all is True:
        services = container_names_by_service.keys()
        targets = []
        for service in services:
            targets.extend(container_names_by_service[service])
        for target in targets:
            podman_args.append(target)
        await compose.podman.run([], "kill", podman_args)
    elif args.services:
        targets = []
        for service in args.services:
            targets.extend(container_names_by_service[service])
        for target in targets:
            podman_args.append(target)
        await compose.podman.run([], "kill", podman_args)
