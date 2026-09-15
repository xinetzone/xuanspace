# SPDX-License-Identifier: GPL-2.0-only
"""``build`` 命令（翻译自 podman_compose.py 第 3688-3764 行）。

- ``build_one``：单个容器的镜像构建（含 ``--if-not-exists`` 预检与临时
  Dockerfile 清理回调）；
- ``compose_build``：按 ``build_deps``（additional_contexts 依赖）拓扑分批
  并发构建。``_add_build`` 是操作局部 ``pending_builds`` 的闭包，与上游
  一致保留在 handler 内部。
"""

import argparse
import asyncio
import os
from collections.abc import Callable
from typing import Any

from ..logging_utils import log
from ..runner import CalledProcessError
from ..translate.build import container_to_build_args

__all__ = ["build_one", "compose_build"]


async def build_one(compose: Any, args: argparse.Namespace, cnt: dict[str, Any]) -> int | None:
    if "build" not in cnt:
        return None
    if getattr(args, "if_not_exists", None):
        try:
            img_id = await compose.podman.output(
                [], "inspect", ["-t", "image", "-f", "{{.Id}}", cnt["image"]]
            )
        except CalledProcessError:
            img_id = None
        if img_id:
            return None

    cleanup_callbacks: list[Callable[[], object]] = []
    build_args = container_to_build_args(
        compose, cnt, args, os.path.exists, cleanup_callbacks=cleanup_callbacks
    )
    status = await compose.podman.run([], "build", build_args)
    for c in cleanup_callbacks:
        c()
    return status


async def compose_build(compose: Any, args: argparse.Namespace) -> int:
    pending_builds: dict[str, list[Any]] = {}

    def _add_build(cnt: dict[str, Any]) -> None:
        cur_builds = pending_builds.get(cnt["service_name"])
        if cur_builds:
            cur_builds.append(cnt)
        else:
            pending_builds[cnt["service_name"]] = [cnt]

    if args.services:
        container_names_by_service = compose.container_names_by_service
        compose.assert_services(args.services)
        for service in args.services:
            cnt = compose.container_by_name[container_names_by_service[service][0]]
            _add_build(cnt)
    else:
        for cnt in compose.containers:
            _add_build(cnt)

    # Continue building until there are no more pending tasks
    while pending_builds:
        # Find the tasks that are not waiting for any dependencies
        current_builds = []
        currently_built_services = []
        for service_name, containers in pending_builds.items():
            cur_srv = compose.services[service_name]
            build_deps = cur_srv.get("build", {}).get("build_deps", [])
            # Check if the task depends on any pending builds
            if set(build_deps).isdisjoint(pending_builds):
                for c in containers:
                    current_builds.append(asyncio.create_task(build_one(compose, args, c)))
                currently_built_services.append(service_name)

        # This should not happen because we check for circular references during the compose
        # file parsing. But just in case...
        if not current_builds:
            log.error("Found no buildable services due to additional_context dependencies")
            return 1

        status = 0
        for t in asyncio.as_completed(current_builds):
            s = await t
            if s is not None and s != 0:
                status = s
        if status != 0:
            return status

        for service_name in currently_built_services:
            # noinspection PyAsyncCall
            pending_builds.pop(service_name)

    return 0
