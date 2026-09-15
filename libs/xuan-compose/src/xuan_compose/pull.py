# SPDX-License-Identifier: GPL-2.0-only
"""镜像拉取（翻译自 podman_compose.py 第 4021-4072 行）。

蓝图中本模块随 T6 引擎层创建；因 T4 移植测试 ``test_pull_image.py`` 覆盖
``pull_image``/``pull_images``，提前在 T4 落地。T6 再补 ``prepare_images``
等引擎装配函数。
"""

import argparse
import asyncio
from typing import Any

from .logging_utils import log
from .model import PullImageSettings
from .runner import Podman  # noqa: F401  # 供 mock.patch("xuan_compose.pull.Podman") 与类型契约使用
from .translate.run_args import is_local

__all__ = [
    "pull_image",
    "pull_images",
    "settings_to_pull_args",
]


def settings_to_pull_args(settings: PullImageSettings) -> list[str]:
    args = ["--policy", settings.policy]
    if settings.quiet:
        args.append("--quiet")

    args.append(settings.image)
    return args


async def pull_image(podman: Any, settings: PullImageSettings) -> int | None:
    if settings.policy in ("never", "build"):
        log.debug("Skipping pull of image %s due to policy %s", settings.image, settings.policy)
        return 0

    ret = await podman.run([], "pull", settings_to_pull_args(settings))
    return ret if not settings.ignore_pull_error else 0


async def pull_images(
    podman: Any,
    args: argparse.Namespace,
    services: list[dict[str, Any]],
) -> int | None:
    pull_tasks = []
    settings: dict[str, PullImageSettings] = {}
    for pull_service in services:
        if not is_local(pull_service):
            image = str(pull_service.get("image", ""))
            policy = getattr(args, "pull", None) or pull_service.get("pull_policy", "missing")

            if image in settings:
                settings[image].update_policy(policy)
            else:
                settings[image] = PullImageSettings(
                    image, policy, getattr(args, "quiet_pull", False)
                )

            if "build" in pull_service:
                # From https://github.com/compose-spec/compose-spec/blob/main/build.md#using-build-and-image
                # When both image and build are specified,
                # we should try to pull the image first,
                # and then build it if it does not exist.
                # we should not stop here if pull fails.
                settings[image].ignore_pull_error = True

    for s in settings.values():
        pull_tasks.append(pull_image(podman, s))

    if pull_tasks:
        ret = await asyncio.gather(*pull_tasks)
        return next((r for r in ret if not r), 0)

    return 0
