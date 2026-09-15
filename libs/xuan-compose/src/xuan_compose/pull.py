# SPDX-License-Identifier: GPL-2.0-only
"""镜像拉取（翻译自 podman_compose.py 第 4021-4072 行）。

蓝图中本模块随 T6 引擎层创建；因 T4 移植测试 ``test_pull_image.py`` 覆盖
``pull_image``/``pull_images``，提前在 T4 落地。T6 再补 ``prepare_images``
等引擎装配函数。
"""

import argparse
import asyncio
from typing import Any

from .compat import strverscmp_lt
from .logging_utils import log
from .model import PullImageSettings
from .runner import Podman  # noqa: F401  # 供 mock.patch("xuan_compose.pull.Podman") 与类型契约使用
from .translate.run_args import is_local

__all__ = [
    "prepare_images",
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


async def prepare_images(
    compose: Any, args: argparse.Namespace, excluded: set[str]
) -> int | None:
    # 翻译自上游第 4076-4106 行。compose 为引擎鸭子类型（services/podman/
    # podman_version/commands），保持与 translate 层一致的 Any 解耦面。

    # When creating containers, podman create internally invokes podman pull with the default
    # policy of --pull=missing.
    # To minimize downtime during container up command, we explicitly run podman pull before
    # tearing down the old container, ensuring the image is already cached when we subsequently
    # call podman create.
    # However, the pull --policy flag was only introduced to podman in version 5.6.0, so we can
    # only perform this pre-teardown optimization when using podman >= 5.6.0.
    if compose.podman_version is not None and not strverscmp_lt(compose.podman_version, "5.6.0"):
        log.info("pulling images: ...")

        pull_services = [v for k, v in compose.services.items() if k not in excluded]
        err = await pull_images(compose.podman, args, pull_services)
        if err:
            log.error("Pull image failed")
            return err

    log.info("building images: ...")

    if not args.no_build:
        # `podman build` does not cache, so don't always build
        build_args = argparse.Namespace(if_not_exists=(not args.build), **args.__dict__)
        build_exit_code = await compose.commands["build"](compose, build_args)
        if build_exit_code != 0:
            log.error("Build command failed")
            return build_exit_code

    return 0
