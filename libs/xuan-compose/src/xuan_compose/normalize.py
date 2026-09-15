# SPDX-License-Identifier: GPL-2.0-only
"""compose 文档规范化：变量替换、短/长格式归一、最终构建上下文解析。

迁移自上游 1.6.0：

* ``rec_subs``（第 465-497 行）
* ``norm_as_list`` / ``norm_as_dict`` / ``norm_ulimit``（第 500-548 行）
* ``normalize_service`` / ``normalize``（第 2078-2198 行）
* ``normalize_service_final`` / ``normalize_final``（第 2201-2218 行）
* ``is_context_git_url``（第 3549-3567 行；按分层依赖置于本模块，
  T5 的 ``translate.build`` 将再导出以保持蓝图符号面）
"""

import os
import urllib.parse
from collections.abc import Iterable
from typing import Any, overload

from .compat import is_list, is_relative_ref, secondarypathisabs
from .errors import PodmanComposeError
from .interpolation import var_interpolate
from .merge import OverrideTag, ResetTag

__all__ = [
    "rec_subs",
    "norm_as_list",
    "norm_as_dict",
    "norm_ulimit",
    "normalize_service",
    "normalize",
    "normalize_service_final",
    "normalize_final",
    "is_context_git_url",
]


@overload
def rec_subs(value: dict, subs_dict: dict[str, Any]) -> dict: ...
@overload
def rec_subs(value: str, subs_dict: dict[str, Any]) -> str: ...
@overload
def rec_subs(value: Iterable, subs_dict: dict[str, Any]) -> Iterable: ...


def rec_subs(value: dict | str | Iterable, subs_dict: dict[str, Any]) -> dict | str | Iterable:
    """
    do bash-like substitution in value and if list of dictionary do that recursively
    """
    if isinstance(value, dict):
        if "environment" in value and isinstance(value["environment"], dict):
            # Load service's environment variables
            subs_dict = subs_dict.copy()
            svc_envs = {k: v for k, v in value["environment"].items() if k not in subs_dict}
            # we need to add `svc_envs` to the `subs_dict` so that it can evaluate the
            # service environment that references another service environment.
            svc_envs = rec_subs(svc_envs, subs_dict)
            subs_dict.update(svc_envs)

            # Resolve short-form environment variables (value is None) to their actual values
            for env_k, env_v in value["environment"].items():
                if env_v is None and env_k in subs_dict:
                    value["environment"][env_k] = subs_dict[env_k]

        value = {rec_subs(k, subs_dict): rec_subs(v, subs_dict) for k, v in value.items()}
    elif isinstance(value, str):
        value = var_interpolate(value, subs_dict)
    elif hasattr(value, "__iter__"):
        value = [rec_subs(i, subs_dict) for i in value]
    return value


def norm_as_list(src: dict[str, Any] | list[Any] | None) -> list[Any]:
    """
    given a dictionary {key1:value1, key2: None} or list
    return a list of ["key1=value1", "key2"]
    """
    if src is None:
        dst: list[Any] = []
    elif isinstance(src, dict):
        dst = [(f"{k}={v}" if v is not None else k) for k, v in src.items()]
    elif is_list(src):
        dst = list(src)
    else:
        dst = [src]
    return dst


def norm_as_dict(src: None | dict[str, str | None] | list[str] | str) -> dict[str, str | None]:
    """
    given a list ["key1=value1", "key2"]
    return a dictionary {key1:value1, key2: None}
    """
    if src is None:
        dst: dict[str, str | None] = {}
    elif isinstance(src, dict):
        dst = dict(src)
    elif is_list(src):
        dst = [i.split("=", 1) for i in src if i]  # type: ignore[assignment]
        dst = [(a if len(a) == 2 else (a[0], None)) for a in dst]  # type: ignore[assignment]
        dst = dict(dst)
    elif isinstance(src, str):
        key, value = src.split("=", 1) if "=" in src else (src, None)
        dst = {key: value}
    else:
        raise ValueError("dictionary or iterable is expected")
    return dst


def norm_ulimit(inner_value: dict | list | int | str) -> str | int:
    if isinstance(inner_value, dict):
        if not inner_value.keys() & {"soft", "hard"}:
            raise ValueError("expected at least one soft or hard limit")
        soft = inner_value.get("soft", inner_value.get("hard"))
        hard = inner_value.get("hard", inner_value.get("soft"))
        return f"{soft}:{hard}"
    if isinstance(inner_value, list):
        return norm_ulimit(norm_as_dict(inner_value))
    # if int or string return as is
    return inner_value


def is_context_git_url(path: str) -> bool:
    r = urllib.parse.urlparse(path)
    if r.scheme in ("git", "http", "https", "ssh", "file", "rsync"):
        return True
    # URL contains a ":" character, a hint of a valid URL
    # But also detects windows file paths (e.g. "C:\path\to\contextdir") as urls
    is_path_with_drive_letter = (
        (os.path.isabs(path) or secondarypathisabs(path))
        and len(path) > 2
        and path[1] == ":"
        and path[2] in ("\\", "/")
    )
    if r.scheme != "" and r.netloc == "" and r.path != "" and not is_path_with_drive_letter:
        return True
    if r.scheme == "":  # tweak path URL to get username from url parser
        r = urllib.parse.urlparse("ssh://" + path)
        if r.username is not None and r.username != "":
            return True
    return False


def normalize_service(service: dict[str, Any], sub_dir: str = "") -> dict[str, Any]:
    if isinstance(service, ResetTag):
        return service

    if isinstance(service, OverrideTag):
        # !override 作用于 service 时对应 YAML mapping 节点，value 必为 dict
        assert isinstance(service.value, dict)
        service = service.value

    if "build" in service:
        build = service["build"]
        if isinstance(build, str):
            service["build"] = {"context": build}
    if sub_dir and "build" in service:
        build = service["build"]
        context = build.get("context", "")
        if context or sub_dir:
            if context.startswith("./"):
                context = context[2:]
            if sub_dir:
                context = os.path.join(sub_dir, context)
            context = context.rstrip("/")
            if not context:
                context = "."
            service["build"]["context"] = context
    if "build" in service and "additional_contexts" in service["build"]:
        if isinstance(build["additional_contexts"], dict):
            new_additional_contexts = []
            for k, v in build["additional_contexts"].items():
                new_additional_contexts.append(f"{k}={v}")
            build["additional_contexts"] = new_additional_contexts
    if "build" in service and "args" in service["build"]:
        if isinstance(build["args"], dict):
            build["args"] = norm_as_list(build["args"])
    for key in ("env_file", "security_opt", "volumes"):
        if key not in service:
            continue
        if isinstance(service[key], str):
            service[key] = [service[key]]
    if "security_opt" in service:
        sec_ls = service["security_opt"]
        for ix, item in enumerate(sec_ls):
            if item in ("seccomp:unconfined", "apparmor:unconfined"):
                sec_ls[ix] = item.replace(":", "=")
    for key in ("environment", "labels"):
        if key not in service:
            continue
        service[key] = norm_as_dict(service[key])
    if "extends" in service:
        extends = service["extends"]
        if isinstance(extends, str):
            extends = {"service": extends}
            service["extends"] = extends
    if "depends_on" in service:
        # deps should become a dictionary of dependencies
        deps = service["depends_on"]
        if isinstance(deps, ResetTag):
            return service
        if isinstance(deps, str):
            deps = {deps: {}}
        elif is_list(deps):
            deps = {x: {} for x in deps}
        elif isinstance(deps, OverrideTag):
            assert isinstance(deps.value, list)
            deps.value = {x: {} for x in deps.value}

        # the dependency service_started is set by default
        # unless requested otherwise.
        if isinstance(deps, OverrideTag):
            # 上一分支已把 sequence 形态的 value 归一为 dict
            assert isinstance(deps.value, dict)
            dep_items: Any = deps.value.items()
        else:
            dep_items = deps.items()
        for k, v in dep_items:
            v.setdefault("condition", "service_started")
        service["depends_on"] = deps
    if "volumes" in service and sub_dir:
        new_volumes = []
        for v in service["volumes"]:
            if isinstance(v, str):
                if is_relative_ref(v):
                    v = os.path.join(sub_dir, v)
            elif isinstance(v, dict):
                source = v["source"]
                if is_relative_ref(source):
                    v["source"] = os.path.join(sub_dir, source)

            new_volumes.append(v)
        service["volumes"] = new_volumes
    if "env_file" in service and sub_dir:
        new_env_file = []
        for ef in service["env_file"]:
            if isinstance(ef, str):
                if is_relative_ref(ef):
                    ef = os.path.join(sub_dir, ef)
            elif isinstance(ef, dict):
                path = ef.get("path")
                if isinstance(path, str) and is_relative_ref(path):
                    ef["path"] = os.path.join(sub_dir, path)
            new_env_file.append(ef)
        service["env_file"] = new_env_file
    if "secrets" in service:
        secrets = service["secrets"]
        if isinstance(secrets, dict):
            raise PodmanComposeError("ERROR: secrets must be a list, not a dict")
        if isinstance(secrets, str):
            service["secrets"] = [secrets]
    if "build" in service and "secrets" in service["build"]:
        build_secrets = service["build"]["secrets"]
        if isinstance(build_secrets, dict):
            raise PodmanComposeError("ERROR: build.secrets must be a list, not a dict")
        if isinstance(build_secrets, str):
            service["build"]["secrets"] = [build_secrets]
    return service


def normalize(compose: dict[str, Any], sub_dir: str = "") -> dict[str, Any]:
    """
    convert compose dict of some keys from string or dicts into arrays

    If ``sub_dir`` is provided, relative paths in ``volumes``, ``env_file`` and
    ``build.context`` are rewritten to be relative to ``sub_dir`` (used when an
    included file lives in a different directory than the project root, per
    Compose Spec resolution of paths in ``include:``d files).
    """
    services = compose.get("services", {}) or {}
    for service in services.values():
        normalize_service(service, sub_dir)
    return compose


def normalize_service_final(service: dict[str, Any], project_dir: str) -> dict[str, Any]:
    if "build" in service:
        build = service["build"]
        context = build if isinstance(build, str) else build.get("context", ".")

        if not is_context_git_url(context):
            context = os.path.normpath(os.path.join(project_dir, context))
        if not isinstance(service["build"], dict):
            service["build"] = {}
        service["build"]["context"] = context
    return service


def normalize_final(compose: dict[str, Any], project_dir: str) -> dict[str, Any]:
    services = compose.get("services", {})
    for service in services.values():
        normalize_service_final(service, project_dir)
    return compose
