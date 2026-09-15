# SPDX-License-Identifier: GPL-2.0-only
"""运行期参数翻译（翻译自 podman_compose.py 第 3391-3397、3795-3809、3945-3956、
4428-4441、4594-4705 行）。

``is_local`` 随 T4 提前落地；T5 补齐 down/run/cp/exec 命令共享的纯翻译函数：
排除集计算、服务信息查找、卷名收集，以及 run/cp/exec 的参数组装。
这些函数不做子进程 I/O，``compose`` 形参为 ``Any`` 鸭子类型。
"""

import argparse
import random
from typing import Any

from ..logging_utils import log
from ..merge import clone
from ..types import DependField
from .mounts import fix_mount_dict, parse_short_mount
from .ports import norm_ports

__all__ = [
    "is_local",
    "get_excluded",
    "deps_from_container",
    "get_service_info",
    "get_volume_names",
    "compose_run_update_container_from_args",
    "compose_cp_args",
    "compose_exec_args",
]


def is_local(container: dict[str, Any]) -> bool:
    """Test if a container is local, i.e. if it is
    * prefixed with localhost/
    * has a build section and is not prefixed
    """
    image = container.get("image", "")
    return image.startswith("localhost/") or ("build" in container and "/" not in image)


def get_excluded(
    compose: Any,
    args: argparse.Namespace,
    dep_field: DependField = DependField.DEPENDENCIES,
) -> set[str]:
    excluded = set()
    if args.services:
        excluded = set(compose.services)
        for service in args.services:
            # we need 'getattr' as compose_down_parse does not configure 'no_deps'
            if service in compose.services and not getattr(args, "no_deps", False):
                excluded -= set(x.name for x in compose.services[service].get(dep_field, set()))
            excluded.discard(service)
    log.debug("** excluding: %s", excluded)
    return excluded


def deps_from_container(args: argparse.Namespace, cnt: dict[str, Any]) -> set[Any]:
    if args.no_deps:
        return set()
    return cnt["_deps"]


def get_service_info(compose: Any, service: str) -> tuple[dict[str, Any], int] | None:
    for index, cnt in enumerate(compose.containers):
        service_name = cnt["_service"]
        if service == service_name:
            return (cnt, index)
    return None


def get_volume_names(compose: Any, cnt: dict[str, Any]) -> list[str]:
    basedir = compose.dirname
    srv_name = cnt["_service"]
    ls = []
    for volume in cnt.get("volumes", []):
        if isinstance(volume, str):
            volume = parse_short_mount(volume, basedir)
        volume = fix_mount_dict(compose, volume, srv_name)
        mount_type = volume["type"]
        if mount_type != "volume":
            continue
        volume_name = volume.get("_vol", {}).get("name")
        ls.append(volume_name)
    return ls


def compose_run_update_container_from_args(
    compose: Any, cnt: dict[str, Any], args: argparse.Namespace
) -> None:
    # adjust one-off container options
    name0 = compose.format_name(args.service, f"tmp{random.randrange(0, 65536)}")
    cnt["name"] = args.name or cnt.get("container_name") or name0
    if args.entrypoint:
        cnt["entrypoint"] = args.entrypoint
    if args.user:
        cnt["user"] = args.user
    if args.workdir:
        cnt["working_dir"] = args.workdir
    env = dict(cnt.get("environment", {}))
    if args.env:
        additional_env_vars = dict(map(lambda each: each.split("=", maxsplit=1), args.env))
        env.update(additional_env_vars)
        cnt["environment"] = env
    if not args.service_ports:
        for k in ("expose", "publishall", "ports"):
            try:
                del cnt[k]
            except KeyError:
                pass
    if args.publish:
        ports = cnt.get("ports", [])
        ports.extend(norm_ports(args.publish))
        cnt["ports"] = ports
    if args.volume:
        # TODO: handle volumes
        volumes = clone(cnt.get("volumes", []))
        volumes.extend(args.volume)
        cnt["volumes"] = volumes
    cnt["tty"] = not args.T
    if args.cnt_command is not None and len(args.cnt_command) > 0:
        cnt["command"] = args.cnt_command
    # can't restart and --rm
    if args.rm and "restart" in cnt:
        del cnt["restart"]


def compose_cp_args(container_name: str, args: argparse.Namespace) -> list[str]:
    podman_args = []
    if args.archive:
        podman_args += ["--archive"]
    if args.overwrite:
        podman_args += ["--overwrite"]

    # Determine which argument has the colon so we know the direction
    if ":" in args.src:
        # container -> local
        cnt_path = args.src.split(":", 1)[1]
        podman_args += [container_name + ":" + cnt_path, args.dst]
    elif ":" in args.dst:
        # local -> container
        cnt_path = args.dst.split(":", 1)[1]
        podman_args += [args.src, container_name + ":" + cnt_path]

    return podman_args


def compose_exec_args(
    cnt: dict[str, Any], container_name: str, args: argparse.Namespace
) -> list[str]:
    podman_args = ["--interactive"]
    if args.privileged:
        podman_args += ["--privileged"]
    if args.user:
        podman_args += ["--user", args.user]
    if args.workdir:
        podman_args += ["--workdir", args.workdir]
    if not args.T:
        podman_args += ["--tty"]
    env = dict(cnt.get("environment", {}))
    if args.env:
        additional_env_vars = dict(
            map(lambda each: each.split("=", maxsplit=1) if "=" in each else (each, None), args.env)
        )
        env.update(additional_env_vars)
    for name, value in env.items():
        podman_args += ["--env", f"{name}" if value is None else f"{name}={value}"]
    podman_args += [container_name]
    if args.cnt_command is not None and len(args.cnt_command) > 0:
        podman_args += args.cnt_command
    return podman_args
