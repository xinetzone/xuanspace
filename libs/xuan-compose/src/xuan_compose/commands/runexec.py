# SPDX-License-Identifier: GPL-2.0-only
"""``run`` / ``cp`` / ``exec`` 命令（翻译自 podman_compose.py 第 4549-4591、
4637-4650、4674-4681 行）。

纯参数组装（``compose_run_update_container_from_args``/``compose_cp_args``/
``compose_exec_args``）属翻译层，本模块只做服务解析与 podman 调用编排。
"""

import argparse
import sys
from typing import Any

from ..translate.container_args import container_to_args
from ..translate.run_args import (
    compose_cp_args,
    compose_exec_args,
    compose_run_update_container_from_args,
)
from .updown import create_pods

__all__ = ["compose_run", "compose_cp", "compose_exec"]


async def compose_run(compose: Any, args: argparse.Namespace) -> None:
    await create_pods(compose)
    compose.assert_services(args.service)
    container_names = compose.container_names_by_service[args.service]
    container_name = container_names[0]
    cnt = dict(compose.container_by_name[container_name])
    deps = cnt["_deps"]
    if deps and not args.no_deps:
        up_args = argparse.Namespace(
            **dict(
                args.__dict__,
                detach=True,
                services=[x.name for x in deps],
                # defaults
                no_build=False,
                build=None,
                force_recreate=False,
                no_recreate=False,
                no_start=False,
                no_cache=False,
                build_arg=[],
                parallel=1,
                remove_orphans=True,
                wait=False,
                no_attach=[],
            )
        )
        await compose.commands["up"](compose, up_args)

    build_args = argparse.Namespace(
        services=[args.service], if_not_exists=(not args.build), build_arg=[], **args.__dict__
    )
    await compose.commands["build"](compose, build_args)

    compose_run_update_container_from_args(compose, cnt, args)
    # run podman
    podman_args = await container_to_args(compose, cnt, args.detach, args.no_deps)
    if not args.detach:
        podman_args.insert(1, "-i")
        if args.rm:
            podman_args.insert(1, "--rm")
    p = await compose.podman.run([], "run", podman_args)
    sys.exit(p)


async def compose_cp(compose: Any, args: argparse.Namespace) -> None:
    if ":" in args.src and ":" not in args.dst:
        service = args.src.split(":", 1)[0]
    elif ":" in args.dst and ":" not in args.src:
        service = args.dst.split(":", 1)[0]
    else:
        raise ValueError(
            f"Invalid copy arguments format: source = {args.src}, destination = {args.dst}."
        )
    compose.assert_services(service)
    container_names = compose.container_names_by_service[service]
    podman_args = compose_cp_args(container_names[0], args)
    p = await compose.podman.run([], "cp", podman_args)
    sys.exit(p)


async def compose_exec(compose: Any, args: argparse.Namespace) -> None:
    compose.assert_services(args.service)
    container_names = compose.container_names_by_service[args.service]
    container_name = container_names[args.index - 1]
    cnt = compose.container_by_name[container_name]
    podman_args = compose_exec_args(cnt, container_name, args)
    p = await compose.podman.run([], "exec", podman_args)
    sys.exit(p)
