# SPDX-License-Identifier: GPL-2.0-only
"""``pull`` / ``push`` 命令（翻译自 podman_compose.py 第 3516-3546 行）。"""

import argparse
import asyncio
from typing import Any

from ..translate.run_args import is_local

__all__ = ["compose_pull", "compose_push"]


async def compose_pull(compose: Any, args: argparse.Namespace) -> int | None:
    img_containers = [cnt for cnt in compose.containers if "image" in cnt]
    if args.services:
        services = set(args.services)
        img_containers = [cnt for cnt in img_containers if cnt["_service"] in services]
    images = {cnt["image"] for cnt in img_containers}
    if not args.force_local:
        local_images = {cnt["image"] for cnt in img_containers if is_local(cnt)}
        images -= local_images
    status = 0
    statuses = await asyncio.gather(*[compose.podman.run([], "pull", [image]) for image in images])
    for s in statuses:
        if s is not None and s != 0:
            status = s
    return status


async def compose_push(compose: Any, args: argparse.Namespace) -> int | None:
    services = set(args.services)
    status = 0
    for cnt in compose.containers:
        if "build" not in cnt:
            continue
        if services and cnt["_service"] not in services:
            continue
        s = await compose.podman.run([], "push", [cnt["image"]])
        if s is not None and s != 0:
            status = s
    return status
