# SPDX-License-Identifier: GPL-2.0-only
"""构建参数翻译（翻译自 podman_compose.py 第 3549-3685 行）。

``is_context_git_url`` 的实现位于基础层 :mod:`xuan_compose.normalize`
（T3 已落地，规范化阶段也需要判定 git 上下文），本模块按蓝图符号面再导出。

注意：``_add_build`` 在上游是 ``compose_build`` handler 内的闭包
（依赖 handler 局部 ``pending_builds``），属 T7 命令层而非翻译层；
上游亦不存在独立的 ``can_merge_build`` 符号（同名测试实际覆盖的是
引擎 ``_parse_compose_file``，随 T6 交付）。
"""

import argparse
import os
import tempfile
from collections.abc import Callable
from typing import Any

from ..normalize import is_context_git_url, norm_as_list
from .resources import container_to_ulimit_build_args
from .secrets import get_secret_args

__all__ = [
    "is_context_git_url",
    "adjust_build_ssh_key_paths",
    "container_to_build_args",
]


def adjust_build_ssh_key_paths(compose: Any, agent_or_key: str) -> str:
    # when using a custom id for ssh property, path to a local SSH key is provided after "="
    parts = agent_or_key.split("=", 1)
    if len(parts) == 1:
        return agent_or_key
    name, path = parts
    path = os.path.expanduser(path)
    return name + "=" + os.path.join(compose.dirname, path)


def container_to_build_args(
    compose: Any,
    cnt: dict[str, Any],
    args: argparse.Namespace,
    path_exists: Callable[[str], bool],
    cleanup_callbacks: list[Callable[[], object]] | None = None,
) -> list[str]:
    build_desc = cnt["build"]
    if not hasattr(build_desc, "items"):
        build_desc = {"context": build_desc}
    ctx = build_desc.get("context", ".")
    dockerfile = build_desc.get("dockerfile", "")
    dockerfile_inline = build_desc.get("dockerfile_inline")
    if dockerfile_inline is not None:
        dockerfile_inline = str(dockerfile_inline)
        # Error if both `dockerfile_inline` and `dockerfile` are set
        if dockerfile and dockerfile_inline:
            raise OSError("dockerfile_inline and dockerfile can't be used simultaneously")
        dockerfile = tempfile.NamedTemporaryFile(delete=False, suffix=".containerfile")
        dockerfile.write(dockerfile_inline.encode())
        dockerfile.close()
        dockerfile = dockerfile.name

        def cleanup_temp_dockfile() -> None:
            if os.path.exists(dockerfile):
                os.remove(dockerfile)

        if cleanup_callbacks is not None:
            cleanup_callbacks.append(cleanup_temp_dockfile)

    build_args = []
    # if given context was not recognized as git url, try joining paths to get a file locally
    if not is_context_git_url(ctx):
        custom_dockerfile_given = False
        if dockerfile:
            dockerfile = os.path.join(ctx, dockerfile)
            custom_dockerfile_given = True
        else:
            dockerfile_alts = [
                "Containerfile",
                "ContainerFile",
                "containerfile",
                "Dockerfile",
                "DockerFile",
                "dockerfile",
            ]
            for dockerfile in dockerfile_alts:
                dockerfile = os.path.join(ctx, dockerfile)
                if path_exists(dockerfile):
                    break

        if path_exists(dockerfile):
            # normalize dockerfile path, as the user could have provided unpredictable file formats
            dockerfile = os.path.normpath(dockerfile)
            build_args.extend(["-f", dockerfile])
        else:
            if custom_dockerfile_given:
                # custom dockerfile name was also not found in the file system
                raise OSError(f"Dockerfile not found in {dockerfile}")
            raise OSError(f"Dockerfile not found in {ctx}")

    elif dockerfile:
        build_args.extend(["-f", dockerfile])

    build_args.extend(["-t", cnt["image"]])

    if "platform" in cnt:
        build_args.extend(["--platform", cnt["platform"]])
    for secret in build_desc.get("secrets", []):
        build_args.extend(get_secret_args(compose, cnt, secret, podman_is_building=True))
    for i in build_desc.get("extra_hosts", []):
        build_args.extend(["--add-host", i])
    for tag in build_desc.get("tags", []):
        build_args.extend(["-t", tag])
    labels = build_desc.get("labels", [])
    if isinstance(labels, dict):
        labels = [f"{k}={v}" for (k, v) in labels.items()]
    for label in labels:
        build_args.extend(["--label", label])
    for additional_ctx in build_desc.get("additional_contexts", {}):
        build_args.extend([f"--build-context={additional_ctx}"])
    if "target" in build_desc:
        build_args.extend(["--target", build_desc["target"]])
    for agent_or_key in norm_as_list(build_desc.get("ssh", {})):
        agent_or_key = adjust_build_ssh_key_paths(compose, agent_or_key)
        build_args.extend(["--ssh", agent_or_key])
    container_to_ulimit_build_args(cnt, build_args)
    if getattr(args, "no_cache", None):
        build_args.append("--no-cache")

    pull_policy = getattr(args, "pull", None)
    if pull_policy:
        build_args.append(f"--pull={pull_policy}")

    args_list = norm_as_list(build_desc.get("args", {}))
    for build_arg in args_list + args.build_arg:
        build_args.extend((
            "--build-arg",
            build_arg,
        ))
    for cache_img in build_desc.get("cache_from", []):
        build_args.extend(["--cache-from", cache_img])
    for cache_img in build_desc.get("cache_to", []):
        build_args.extend(["--cache-to", cache_img])
    build_args.append(ctx)
    return build_args
