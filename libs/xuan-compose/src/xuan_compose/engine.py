# SPDX-License-Identifier: GPL-2.0-only
"""引擎层（翻译自 podman_compose.py 第 116、2374-2389、2419-3147 行）。

``ComposeEngine`` 是上游 ``PodmanCompose`` 的对等引擎，承载 compose 文件
解析、profiles 解析、x-podman 设置、命名/pod 决策、容器/卷/网络装配等
有状态逻辑。与上游的结构性差异（记 T10 差异表）：

- 类名 ``PodmanCompose`` → ``ComposeEngine``；
- 构造器显式接收可注入的 ``podman`` runner（默认 ``None``），实例化过程
  不创建 runner、不发起子进程；上游在 ``run()`` 中构造的 runner 装配逻辑
  随 T8 迁入 CLI 层；
- ``run()``/``_parse_args()`` 的命令分发与参数解析职责随 T8 迁入 CLI 层，
  本层只保留引擎自身的纯解析/装配方法；
- ``commands`` 保持为普通空字段，T7 命令层以显式注册表填充，构造期与
  导入期均不再有装饰器副作用；
- 嵌套枚举 ``XPodmanSettingKey`` 已在 T4/T5 提为 :mod:`xuan_compose.model`
  的模块级 ``StrEnum``，本模块以此为唯一事实源；
- 标签中的版本字符串使用派生包版本（``1.6.0+xuan.x``），非上游字面量。
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from typing import Any
from urllib.parse import quote

import yaml

from . import __version__
from .compat import try_int, try_parse_bool
from .dependencies import flat_deps
from .discovery import find_compose_files_recursively
from .envfile import dotenv_to_dict
from .logging_utils import log
from .merge import OverrideTag, ResetTag, load_yaml_or_die, rec_merge, resolve_extends
from .model import XPodmanSettingKey
from .normalize import norm_as_dict, norm_as_list, normalize, normalize_final, rec_subs
from .runner import Podman
from .translate.mounts import get_mnt_dict
from .translate.ports import norm_ports

__all__ = ["ComposeEngine", "COMPOSE_DEFAULT_LS"]

#: 项目名规范化：podman 要求 [a-zA-Z0-9][a-zA-Z0-9_.-]*（上游第 116 行）
norm_re = re.compile("[^-_a-z0-9]")

#: 未显式 -f/COMPOSE_FILE 时的默认 compose 文件名候选（上游第 2374-2389 行）
COMPOSE_DEFAULT_LS = [
    "compose.yaml",
    "compose.yml",
    "compose.override.yaml",
    "compose.override.yml",
    "podman-compose.yaml",
    "podman-compose.yml",
    "docker-compose.yml",
    "docker-compose.yaml",
    "docker-compose.override.yml",
    "docker-compose.override.yaml",
    "container-compose.yml",
    "container-compose.yaml",
    "container-compose.override.yml",
    "container-compose.override.yaml",
]


class ComposeEngine:
    """对等 ``podman_compose.PodmanCompose`` 的有状态引擎（不含 CLI 分发）。"""

    def __init__(self, podman: Podman | None = None) -> None:
        # 上游为无赋值的裸注解 ``self.podman: Podman``，在 run() 中构造；
        # 分层后改为显式注入，默认 None 保证实例化零子进程（TR-6.1）。
        self.podman: Podman | None = podman
        self.podman_version: str | None = None
        self.environ: dict[str, str] = {}
        self.exit_code = None
        self.commands: dict[str, Any] = {}
        self.global_args = argparse.Namespace()
        self.project_name: str | None = None
        self.dirname: str
        self.pods: list[Any]
        self.containers: list[Any] = []
        self.vols: dict[str, Any] | None = None
        self.networks: dict[str, Any] = {}
        self.default_net: str | None = "default"
        self.declared_secrets: dict[str, Any] | None = None
        self.container_names_by_service: dict[str, list[str]]
        self.container_by_name: dict[str, Any]
        self.services: dict[str, Any]
        self.all_services: set[Any] = set()
        self.prefer_volume_over_mount = True
        self.x_podman: dict[XPodmanSettingKey, Any] = {}
        self.merged_yaml: Any
        self.yaml_hash = ""
        self.console_colors = [
            "\x1b[1;32m",
            "\x1b[1;33m",
            "\x1b[1;34m",
            "\x1b[1;35m",
            "\x1b[1;36m",
        ]

    def assert_services(self, services: dict[str, Any]) -> None:
        if isinstance(services, str):
            services = [services]
        given = set(services or [])
        missing = given - self.all_services
        if missing:
            missing_csv = ",".join(missing)
            log.warning("missing services [%s]", missing_csv)
            sys.exit(1)

    def get_podman_args(self, cmd: str) -> list[str]:
        xargs = []
        for args in self.global_args.podman_args:
            xargs.extend(shlex.split(args))
        xargs += [cmd]
        cmd_norm = cmd if cmd != "create" else "run"
        cmd_args = self.global_args.__dict__.get(f"podman_{cmd_norm}_args", [])
        for args in cmd_args:
            xargs.extend(shlex.split(args))
        return xargs

    def config_hash(self, service: dict[str, Any]) -> str:
        """
        Returns a hash of the service configuration.
        This is used to detect changes in the service configuration.
        """
        if "_config_hash" in service:
            return service["_config_hash"]

        # Use a stable representation of the service configuration
        jsonable_service = self.original_configuration(service)
        config_str = json.dumps(jsonable_service, sort_keys=True)
        service["_config_hash"] = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
        return service["_config_hash"]

    def original_configuration(self, configuration: dict[Any, Any]) -> dict[str, Any]:
        """
        Returns the original configuration without any overrides or resets.
        This is used to get a stable representation of the configuration.
        (can be converted to a JSON string)
        """
        return {
            # recurse if the value is also a dictionary
            key: self.original_configuration(value) if isinstance(value, dict) else value
            # iterate over the configuration items
            for key, value in configuration.items()
            # filter
            if (
                # only string keys
                isinstance(key, str)
                # which do not start with an underscore
                and not key.startswith("_")
                # also exclude !override
                and not isinstance(value, OverrideTag)
                # and !reset tags
                and not isinstance(value, ResetTag)
            )
        }

    def resolve_pod_name(self) -> str | None:
        # Priorities:
        # - Command line --in-pod
        # - docker-compose.yml x-podman.in_pod
        # - Default value of true
        in_pod_arg = self.global_args.in_pod or self.x_podman.get(
            XPodmanSettingKey.IN_POD, True
        )

        in_pod_arg_parsed = try_parse_bool(in_pod_arg)
        if in_pod_arg_parsed is True:
            return f"pod_{self.project_name}"
        if in_pod_arg_parsed is False:
            return None

        assert isinstance(in_pod_arg, str) and in_pod_arg
        return in_pod_arg

    def resolve_pod_args(self) -> list[str]:
        # Priorities:
        # - Command line --pod-args
        # - docker-compose.yml x-podman.pod_args
        # - Default value
        if self.global_args.pod_args is not None:
            return shlex.split(self.global_args.pod_args)
        return self.x_podman.get(
            XPodmanSettingKey.POD_ARGS, ["--infra=false", "--share="]
        )

    def join_name_parts(self, *parts: str) -> str:
        setting = self.x_podman.get(XPodmanSettingKey.NAME_SEPARATOR_COMPAT, False)
        if try_parse_bool(setting):
            sep = "-"
        else:
            sep = "_"
        return sep.join(parts)

    def format_name(self, *parts: str) -> str:
        assert self.project_name is not None
        return self.join_name_parts(self.project_name, *parts)

    def _parse_x_podman_settings(self, compose: dict[str, Any], environ: dict[str, str]) -> None:
        known_keys = {s.value: s for s in XPodmanSettingKey}

        self.x_podman = {}

        for k, v in compose.get("x-podman", {}).items():
            known_key = known_keys.get(k)
            if known_key:
                self.x_podman[known_key] = v
            else:
                log.warning(
                    "unknown x-podman key [%s] in compose file, supported keys: %s",
                    k,
                    ", ".join(known_keys.keys()),
                )

        env = {
            key.removeprefix("PODMAN_COMPOSE_").lower(): value
            for key, value in environ.items()
            if key.startswith("PODMAN_COMPOSE_")
            and key not in {"PODMAN_COMPOSE_PROVIDER", "PODMAN_COMPOSE_WARNING_LOGS"}
        }

        for k, v in env.items():
            known_key = known_keys.get(k)
            if known_key:
                self.x_podman[known_key] = v
            else:
                log.warning(
                    "unknown PODMAN_COMPOSE_ key [%s] in environment, supported keys: %s",
                    k,
                    ", ".join(known_keys.keys()),
                )

        # If Docker Compose compatibility is enabled, set compatibility settings
        # that are not explicitly set already.
        if self.x_podman.get(XPodmanSettingKey.DOCKER_COMPOSE_COMPAT, False):

            def set_if_not_already_set(key: XPodmanSettingKey, value: bool) -> None:
                if key not in self.x_podman:
                    self.x_podman[key] = value

            set_if_not_already_set(
                XPodmanSettingKey.DEFAULT_NET_BEHAVIOR_COMPAT, True
            )
            set_if_not_already_set(XPodmanSettingKey.NAME_SEPARATOR_COMPAT, True)
            set_if_not_already_set(XPodmanSettingKey.IN_POD, False)

    def _parse_compose_file(self) -> None:
        args = self.global_args
        # cmd = args.command
        project_dir = os.environ.get("COMPOSE_PROJECT_DIR")
        if project_dir and os.path.isdir(project_dir):
            os.chdir(project_dir)
        pathsep = os.environ.get("COMPOSE_PATH_SEPARATOR", os.pathsep)

        # Load env files early to honor COMPOSE_FILE from .env
        early_dotenv: dict[str, str | None] = {}
        if not args.env_file:
            project_dotenv_file = os.path.realpath(os.path.join(os.getcwd(), ".env"))
            if os.path.exists(project_dotenv_file):
                early_dotenv.update(dotenv_to_dict(project_dotenv_file))
        else:
            for env_file in args.env_file:
                dotenv_path = os.path.realpath(env_file)
                if not os.path.exists(dotenv_path):
                    log.fatal("Couldn't find env file: %s", dotenv_path)
                    sys.exit(1)
                early_dotenv.update(dotenv_to_dict(dotenv_path))

        if not args.file:
            default_str = os.environ.get("COMPOSE_FILE")
            if not default_str:
                default_str = early_dotenv.get("COMPOSE_FILE")
            if default_str:
                default_ls = default_str.split(pathsep)
                args.file = list(filter(os.path.exists, default_ls))
            else:
                # Recursive search up the directory tree
                current_working_dir = os.getcwd()
                result = find_compose_files_recursively(current_working_dir, COMPOSE_DEFAULT_LS)

                if result:
                    found_files, base_dir = result
                    args.file = found_files
                    # Change to the directory where compose files were found
                    log.info("Found compose files in: %s", base_dir)
                    log.info("Changing working directory to: %s", base_dir)
                    os.chdir(base_dir)
                else:
                    # Fallback to original behavior if no files found
                    default_ls = COMPOSE_DEFAULT_LS
                    args.file = list(filter(os.path.exists, default_ls))

        files = args.file
        if not files:
            log.fatal(
                "no compose.yaml, docker-compose.yml or container-compose.yml file found, "
                "pass files with -f"
            )
            sys.exit(-1)
        # 上游为 map/zip 惰性对，此处等价收敛为列表推导（T10 差异表）
        missing = [fn for fn in files if not (fn == "-" or os.path.exists(fn))]
        if missing:
            log.fatal("missing files: %s", missing)
            sys.exit(1)
        # make absolute
        relative_files = files
        filename = files[0]
        project_name = args.project_name
        # no_ansi = args.no_ansi
        # no_cleanup = args.no_cleanup
        # dry_run = args.dry_run
        # host_env = None
        dirname: str = os.path.realpath(os.path.dirname(filename))
        dir_basename = os.path.basename(dirname)
        self.dirname = dirname

        dotenv_dict: dict[str, str | None] = {}
        if not args.env_file:
            # No --env-file specified: load the default .env from the
            # compose file's directory
            project_dotenv_file = os.path.realpath(os.path.join(dirname, ".env"))
            if os.path.exists(project_dotenv_file):
                dotenv_dict.update(dotenv_to_dict(project_dotenv_file))
        else:
            # User-specified env files are resolved relative to the CWD
            # Later files override earlier ones
            for env_file in args.env_file:
                dotenv_path = os.path.realpath(env_file)
                if not os.path.exists(dotenv_path):
                    log.fatal("Couldn't find env file: %s", dotenv_path)
                    sys.exit(1)
                dotenv_dict.update(dotenv_to_dict(dotenv_path))

        os.environ.update({
            key: value  # type: ignore[misc]
            for key, value in dotenv_dict.items()
            if key.startswith("PODMAN_")  # type: ignore[misc]
        })
        self.environ = dotenv_dict  # type: ignore[assignment]
        self.environ.update(dict(os.environ))
        # see: https://docs.docker.com/compose/reference/envvars/
        # see: https://docs.docker.com/compose/env-file/
        self.environ.update({
            "COMPOSE_PROJECT_DIR": dirname,
            "COMPOSE_FILE": pathsep.join(relative_files),
            "COMPOSE_PATH_SEPARATOR": pathsep,
        })

        if args and "env" in args and args.env:
            env_vars = norm_as_dict(args.env)
            self.environ.update(env_vars)  # type: ignore[arg-type]

        profiles_from_env = {
            p.strip() for p in self.environ.get("COMPOSE_PROFILES", "").split(",") if p.strip()
        }
        requested_profiles = set(args.profile).union(profiles_from_env)

        target_service = getattr(
            args, "service", None
        )  # example: command `run` can only have one service
        target_services = getattr(
            args, "services", None
        )  # example: command `build` can have several services

        target = [target_service] if target_service else target_services or []

        compose: dict[str, Any] = {}
        # Iterate over files primitively to allow appending to files in-loop
        files_iter = iter(files)
        # Track files appended by ``include:`` so we can resolve their
        # relative paths against the included file's directory per the
        # Compose Spec, without changing the legacy merge behavior of
        # files passed directly via ``-f``.
        include_origin_files: set[str] = set()

        while True:
            try:
                filename = next(files_iter)
            except StopIteration:
                break

            if filename.strip().split("/")[-1] == "-":
                content = load_yaml_or_die(filename, sys.stdin)
            else:
                with open(filename, encoding="utf-8") as f:
                    content = load_yaml_or_die(filename, f)
                # log(filename, json.dumps(content, indent = 2))
            if not isinstance(content, dict):
                log.fatal("Compose file does not contain a top level object: %s", filename)
                sys.exit(1)
            if "version" in content:
                log.warning(
                    "%s: the attribute `version` is obsolete, it will be ignored, "
                    "please remove it to avoid potential confusion",
                    filename,
                )
            # For files arriving via ``include:``, paths inside the file must
            # resolve against the included file's directory rather than the
            # project root (Compose Spec, ``include`` section). Pass that as
            # sub_dir so volumes / env_file / build.context get rewritten.
            file_sub_dir = ""
            if filename in include_origin_files:
                file_dir = os.path.dirname(os.path.abspath(filename))
                file_sub_dir = os.path.relpath(file_dir, self.dirname)
                if file_sub_dir == ".":
                    file_sub_dir = ""
                elif not file_sub_dir.startswith((".", "/")):
                    # Prefix with "./" so rewritten paths remain recognizable
                    # as relative refs (is_relative_ref checks for "./"/".."
                    # prefixes).
                    file_sub_dir = "./" + file_sub_dir
            content = normalize(content, file_sub_dir)
            # log(filename, json.dumps(content, indent = 2))

            # See also https://docs.docker.com/compose/how-tos/project-name/#set-a-project-name
            # **project_name** is initialized to the argument of the `-p` command line flag.
            if not project_name:
                project_name = self.environ.get("COMPOSE_PROJECT_NAME")
                if not project_name:
                    project_name = content.get("name")
                if not project_name:
                    project_name = dir_basename.lower()
                project_name = rec_subs(project_name, self.environ)
                # More strict then actually needed for simplicity:
                # podman requires [a-zA-Z0-9][a-zA-Z0-9_.-]*
                project_name_normalized = norm_re.sub("", project_name)
                if not project_name_normalized:
                    raise RuntimeError(f"Project name [{project_name}] normalized to empty")
                project_name = project_name_normalized

            self.project_name = project_name
            assert self.project_name is not None
            self.environ.update({"COMPOSE_PROJECT_NAME": self.project_name})

            content = rec_subs(content, self.environ)
            if isinstance(content_services := content.get("services"), dict):
                for service in content_services.values():
                    if not isinstance(service, OverrideTag) and not isinstance(service, ResetTag):
                        if "extends" in service and (
                            service_file := service["extends"].get("file")
                        ):
                            service["extends"]["file"] = os.path.join(
                                os.path.dirname(filename), service_file
                            )

            rec_merge(compose, content)
            # If `include` is used, append included files to files
            include = compose.get("include")
            if include is not None:
                # Validate that `include` is a list. If it were a dict, iterating it
                # would yield its keys (strings), causing bogus file paths
                if not isinstance(include, list):
                    raise RuntimeError("`include` must be a list")

                new_includes: list[str] = []
                for item in include:
                    if isinstance(item, str):
                        new_includes.append(os.path.join(os.path.dirname(filename), item))
                    elif isinstance(item, dict):
                        if "path" not in item:
                            raise RuntimeError("Missing required 'path' key in `include` block")
                        path = item["path"]
                        if isinstance(path, str):
                            new_includes.append(os.path.join(os.path.dirname(filename), path))
                        elif isinstance(path, list):
                            new_includes.extend(
                                os.path.join(os.path.dirname(filename), p) for p in path
                            )
                        else:
                            raise RuntimeError("'path' must be a string or a list of strings")
                    else:
                        raise RuntimeError(
                            "Items in `include` must be strings or dictionaries with a 'path' key"
                        )
                files.extend(new_includes)
                include_origin_files.update(new_includes)
                # As compose obj is updated and tested with every loop, not deleting `include`
                # from it, results in it being tested again and again, original values for
                # `include` be appended to `files`, and, included files be processed for ever.
                # Solution is to remove 'include' key from compose obj. This doesn't break
                # having `include` present and correctly processed in included files
                del compose["include"]
        resolved_services = self._resolve_profiles(
            compose.get("services") or {}, target, requested_profiles
        )
        compose["services"] = resolved_services
        if not getattr(args, "no_normalize", None):
            compose = normalize_final(compose, self.dirname)
        compose.pop("version", None)
        self.merged_yaml = yaml.safe_dump(compose)
        merged_json_b = json.dumps(
            self.original_configuration(compose), separators=(",", ":")
        ).encode("utf-8")
        self.yaml_hash = hashlib.sha256(merged_json_b).hexdigest()
        compose["_dirname"] = dirname
        # debug mode
        if len(files) > 1:
            log.debug(" ** merged:\n%s", json.dumps(self.original_configuration(compose), indent=2))
        # ver = compose.get('version')

        self._parse_x_podman_settings(compose, self.environ)

        pod_name = self.resolve_pod_name()

        services: dict = compose.get("services", {})
        if not services:
            log.warning("WARNING: No services defined")
        # include services with no profile defined or the selected profiles
        services = self._resolve_profiles(services, target, requested_profiles)

        # NOTE: maybe add "extends.service" to _deps at this stage
        flat_deps(services, with_extends=True)
        service_names = sorted([(len(srv["_deps"]), name) for name, srv in services.items()])
        resolve_extends(services, [name for _, name in service_names], self.environ)
        flat_deps(services)

        # networks: [...]
        nets = compose.get("networks") or {}
        if not nets:
            nets["default"] = None

        # Resolve the inter-service build dependencies in additional contexts
        self._resolve_context_dependencies(services)

        self.networks = nets
        if self.x_podman.get(XPodmanSettingKey.DEFAULT_NET_BEHAVIOR_COMPAT, False):
            # If there is no network_mode and networks in service,
            # docker-compose will create default network named '<project_name>_default'
            # and add the service to the default network.
            # So we always set `default_net = 'default'` for compatibility
            if "default" not in self.networks:
                self.networks["default"] = None
        else:
            if len(self.networks) == 1:
                self.default_net = list(nets.keys())[0]
            elif "default" in nets:
                self.default_net = "default"
            else:
                self.default_net = None

        allnets = set()
        for name, srv in services.items():
            srv_nets = srv.get("networks", self.default_net)
            srv_nets = (
                list(srv_nets.keys()) if isinstance(srv_nets, dict) else norm_as_list(srv_nets)
            )
            allnets.update(srv_nets)
        given_nets = set(nets.keys())
        missing_nets = allnets - given_nets
        unused_nets = given_nets - allnets - set(["default"])
        if len(unused_nets):
            unused_nets_str = ",".join(unused_nets)
            log.warning("WARNING: unused networks: %s", unused_nets_str)
        if len(missing_nets):
            missing_nets_str = ",".join(missing_nets)
            raise RuntimeError(f"missing networks: {missing_nets_str}")
        # volumes: [...]
        self.vols = compose.get("volumes", {}) or {}
        podman_compose_labels = [
            "io.podman.compose.project=" + project_name,
            "io.podman.compose.version=" + __version__,
            f"PODMAN_SYSTEMD_UNIT=podman-compose@{project_name}.service",
            "com.docker.compose.project=" + project_name,
            "com.docker.compose.project.working_dir=" + dirname,
            "com.docker.compose.project.config_files=" + ",".join(relative_files),
        ]
        # other top-levels:
        # configs: {...}
        self.declared_secrets = compose.get("secrets", {})
        given_containers = []
        container_names_by_service: dict[str, list[str]] = {}
        self.services = services
        for service_name, service_desc in services.items():
            replicas = 1
            if "scale" in args and args.scale is not None:
                # Check `--scale` args from CLI command
                scale_args = args.scale.split("=")
                if service_name == scale_args[0]:
                    replicas = try_int(scale_args[1], fallback=1)
            elif "scale" in service_desc:
                # Check `scale` value from compose yaml file
                replicas = try_int(service_desc.get("scale"), fallback=1)
            elif (
                "deploy" in service_desc
                and "replicas" in service_desc.get("deploy", {})
                and "replicated" == service_desc.get("deploy", {}).get("mode", "")
            ):
                # Check `deploy: replicas:` value from compose yaml file
                # Note: All conditions are necessary to handle case
                replicas = try_int(service_desc.get("deploy", {}).get("replicas"), fallback=1)

            container_names_by_service[service_name] = []
            for num in range(1, replicas + 1):
                name0 = self.format_name(service_name, str(num))
                if num == 1:
                    name = service_desc.get("container_name", name0)
                else:
                    name = name0

                if service_desc.get("container_name", False):
                    log_prefix = name
                else:
                    log_prefix = f"{service_name}_{num}"
                container_names_by_service[service_name].append(name)
                # log(service_name,service_desc)
                cnt = {
                    "pod": pod_name,
                    "name": name,
                    "num": num,
                    "service_name": service_name,
                    "log_prefix": log_prefix,
                    **service_desc,
                }
                x_podman = service_desc.get("x-podman")
                rootfs_mode = x_podman is not None and x_podman.get("rootfs") is not None
                if "image" not in cnt and not rootfs_mode:
                    cnt["image"] = self.format_name(service_name)
                labels = norm_as_list(cnt.get("labels"))
                cnt["ports"] = norm_ports(cnt.get("ports"))
                labels.extend(podman_compose_labels)
                labels.extend([
                    f"io.podman.compose.config-hash={self.config_hash(service_desc)}",
                    f"com.docker.compose.container-number={num}",
                    f"io.podman.compose.service={service_name}",
                    f"com.docker.compose.service={service_name}",
                ])
                cnt["labels"] = labels
                cnt["_service"] = service_name
                cnt["_project"] = project_name
                given_containers.append(cnt)
                volumes = cnt.get("volumes", [])
                for volume in volumes:
                    mnt_dict = get_mnt_dict(self, cnt, volume)
                    if (
                        mnt_dict.get("type") == "volume"
                        and mnt_dict["source"]
                        and mnt_dict["source"] not in self.vols  # type: ignore[operator]
                    ):
                        vol_name = mnt_dict["source"]
                        raise RuntimeError(f"volume [{vol_name}] not defined in top level")
        self.container_names_by_service = container_names_by_service
        self.all_services = set(container_names_by_service.keys())
        container_by_name = {c["name"]: c for c in given_containers}
        # log("deps:", [(c["name"], c["_deps"]) for c in given_containers])
        given_containers = list(container_by_name.values())
        given_containers.sort(key=lambda c: len(c.get("_deps", [])))
        # log("sorted:", [c["name"] for c in given_containers])

        self.pods = [{"name": pod_name}] if pod_name else []
        self.containers = given_containers
        self.container_by_name = {c["name"]: c for c in given_containers}

    def _resolve_profiles(
        self,
        defined_services: dict[str, Any],
        target: list[str],
        requested_profiles: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Returns a service dictionary (key = service name, value = service config) compatible with
        the requested_profiles list and target service or services.

        The returned service dictionary contains all services which do not include/reference a
        profile, match the requested_profiles, and match profiles of an explicitly targeted service
        or services.

        :param defined_services: The service dictionary
        :param requested_profiles: The profiles requested using the --profile arg.
        :param target: Name of service or services targeted by the current command
        """
        if requested_profiles is None:
            requested_profiles = set()

        for service in target:
            requested_profiles.update(defined_services.get(service, {}).get("profiles", []))

        services = {}

        for name, config in defined_services.items():
            service_profiles = set(config.get("profiles", []))
            if not service_profiles or requested_profiles.intersection(service_profiles):
                services[name] = config
        return services

    # Docker Compose specifies that services can have build-time dependencies on each other
    # through the "additional_contexts" that can refer to the other services' images.
    def _resolve_context_dependencies(self, services: dict[str, Any]) -> None:
        for name, service in services.items():
            additional_build_contexts = service.get("build", {}).get("additional_contexts")
            if not additional_build_contexts:
                continue

            deps = set()
            processed = []
            for context in additional_build_contexts:
                parts = str(context).split("=", 1)
                if len(parts) != 2:
                    processed.append(context)  # Something unknown. Just ignore.
                    continue

                context_name, val = parts
                if not val.startswith("service:"):
                    processed.append(context)  # Not a service reference
                    continue
                # This is a reference in the form of "service:target_service"
                target_service_name = val.removeprefix("service:")
                target_service = services[target_service_name]
                if target_service is None:
                    raise ValueError(
                        f"Service '{name}' references non-existent service "
                        f"'{val.removeprefix('service:')}' in the "
                        f"additional context '{context_name}'"
                    )
                # Get the image name for that service
                target_image = target_service.get("image")
                if not target_image:
                    target_image = "localhost/" + self.format_name(target_service_name)

                # Replace the context with the docker image reference
                image_url = "docker://" + quote(target_image)
                processed.append(f"{context_name}={image_url}")
                deps.add(target_service_name)

            service["build"]["additional_contexts"] = processed
            service["build"]["build_deps"] = sorted(list(deps))

        # Verify that there are no (possibly recursive) circular dependencies between services
        def check_circular(current: str, path: list[str]) -> None:
            if current in path:
                cycle = " -> ".join(path + [current])
                raise ValueError(f"Circular dependency in additional build contexts: {cycle}")
            path.append(current)

            # Retrieve dependencies stored during the resolution phase
            service_deps = services[current].get("build", {}).get("build_deps", [])
            for dep in service_deps:
                check_circular(dep, path)

            path.pop()

        for service_name in services:
            check_circular(service_name, [])
