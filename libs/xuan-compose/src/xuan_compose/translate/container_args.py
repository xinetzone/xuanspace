# SPDX-License-Identifier: GPL-2.0-only
"""容器创建/运行参数总装（翻译自 podman_compose.py 第 1344-1622 行）。

``container_to_args`` 把规范化后的服务描述翻译为完整的
``podman run`` 参数序列，编排 mounts/networks/secrets/ports/resources
等各子翻译器；其中卷与网络的存在性断言（async I/O）就地等待，
与上游调用顺序逐行一致。``compose`` 形参为 ``Any`` 鸭子类型。
"""

import json
import os
import shlex
from typing import Any

from ..compat import is_list, str_to_seconds
from ..envfile import dotenv_to_dict
from ..logging_utils import log
from ..normalize import norm_as_list
from .mounts import get_mount_args
from .networks import assert_cnt_nets, get_net_args
from .ports import port_dict_to_str
from .resources import container_to_res_args, container_to_ulimit_args
from .secrets import get_secret_args

__all__ = [
    "container_to_args",
]


async def container_to_args(
    compose: Any, cnt: dict[str, Any], detached: bool = True, no_deps: bool = False
) -> list[str]:
    # TODO: double check -e , --add-host, -v, --read-only
    dirname = compose.dirname
    name = cnt["name"]
    podman_args = [f"--name={name}"]

    if detached:
        podman_args.append("-d")

    pod = cnt.get("pod", "")
    if pod:
        podman_args.append(f"--pod={pod}")
    deps = []
    for dep_srv in cnt.get("_deps", []):
        deps.extend(compose.container_names_by_service.get(dep_srv.name, []))
    if deps and not no_deps:
        deps_csv = ",".join(deps)
        podman_args.append(f"--requires={deps_csv}")
    sec = norm_as_list(cnt.get("security_opt"))
    for sec_item in sec:
        podman_args.extend(["--security-opt", sec_item])
    ann = norm_as_list(cnt.get("annotations"))
    for a in ann:
        podman_args.extend(["--annotation", a])
    if cnt.get("read_only"):
        podman_args.append("--read-only")
    if cnt.get("http_proxy") is False:
        podman_args.append("--http-proxy=false")
    for i in cnt.get("labels", []):
        podman_args.extend(["--label", i])
    for c in cnt.get("cap_add", []):
        podman_args.extend(["--cap-add", c])
    for c in cnt.get("cap_drop", []):
        podman_args.extend(["--cap-drop", c])
    for item in cnt.get("group_add", []):
        podman_args.extend(["--group-add", item])
    for item in cnt.get("devices", []):
        podman_args.extend(["--device", item])
    for item in cnt.get("device_cgroup_rules", []):
        podman_args.extend(["--device-cgroup-rule", item])
    for item in norm_as_list(cnt.get("dns")):
        podman_args.extend(["--dns", item])
    for item in norm_as_list(cnt.get("dns_opt")):
        podman_args.extend(["--dns-opt", item])
    for item in norm_as_list(cnt.get("dns_search")):
        podman_args.extend(["--dns-search", item])
    env_file = cnt.get("env_file", [])
    if isinstance(env_file, (dict, str)):
        env_file = [env_file]
    for i in env_file:
        if isinstance(i, str):
            i = {"path": i}
        path = i["path"]
        required = i.get("required", True)
        i = os.path.realpath(os.path.join(dirname, path))
        if not os.path.exists(i):
            if not required:
                continue
            raise ValueError(f"Env file at {i} does not exist")
        dotenv_dict = {}
        dotenv_dict = dotenv_to_dict(i)
        env = norm_as_list(dotenv_dict)
        for e in env:
            podman_args.extend(["-e", e])
    env = norm_as_list(cnt.get("environment", {}))
    for e in env:
        # new environment variable is set
        if "=" in e:
            podman_args.extend(["-e", e])
        else:
            # environment variable already exists in environment so pass its value
            if e in compose.environ.keys():
                podman_args.extend(["-e", f"{e}={compose.environ[e]}"])

    tmpfs_ls = cnt.get("tmpfs", [])
    if isinstance(tmpfs_ls, str):
        tmpfs_ls = [tmpfs_ls]
    for i in tmpfs_ls:
        podman_args.extend(["--tmpfs", i])
    for volume in cnt.get("volumes", []):
        podman_args.extend(await get_mount_args(compose, cnt, volume))

    await assert_cnt_nets(compose, cnt)
    podman_args.extend(get_net_args(compose, cnt))

    log_config = cnt.get("logging")
    if log_config is not None:
        podman_args.append(f'--log-driver={log_config.get("driver", "k8s-file")}')
        log_opts = log_config.get("options", {})
        podman_args += [f"--log-opt={name}={value}" for name, value in log_opts.items()]
    for secret in cnt.get("secrets", []):
        podman_args.extend(get_secret_args(compose, cnt, secret))
    for i in cnt.get("extra_hosts", []):
        podman_args.extend(["--add-host", i])
    for i in cnt.get("expose", []):
        podman_args.extend(["--expose", i])
    if cnt.get("publishall"):
        podman_args.append("-P")
    ports = cnt.get("ports", [])
    if isinstance(ports, str):
        ports = [ports]
    for port in ports:
        if isinstance(port, dict):
            port = port_dict_to_str(port)
        elif not isinstance(port, str):
            raise TypeError("port should be either string or dict")
        podman_args.extend(["-p", port])

    userns_mode = cnt.get("userns_mode")
    if userns_mode is not None:
        podman_args.extend(["--userns", userns_mode])

    user = cnt.get("user")
    if user is not None:
        podman_args.extend(["-u", user])
    if cnt.get("working_dir") is not None:
        podman_args.extend(["-w", cnt["working_dir"]])
    if cnt.get("hostname"):
        podman_args.extend(["--hostname", cnt["hostname"]])
    if cnt.get("shm_size"):
        podman_args.extend(["--shm-size", str(cnt["shm_size"])])
    if cnt.get("stdin_open"):
        podman_args.append("-i")
    if cnt.get("stop_signal"):
        podman_args.extend(["--stop-signal", cnt["stop_signal"]])
    stop_grace = cnt.get("stop_grace_period")
    if stop_grace:
        timeout = str_to_seconds(stop_grace)
        if timeout is not None:
            podman_args.extend(["--stop-timeout", str(timeout)])

    sysctls = cnt.get("sysctls")
    if sysctls is not None:
        if isinstance(sysctls, dict):
            for sysctl, value in sysctls.items():
                podman_args.extend(["--sysctl", f"{sysctl}={value}"])
        elif isinstance(sysctls, list):
            for i in sysctls:
                podman_args.extend(["--sysctl", i])
        else:
            raise TypeError("sysctls should be either dict or list")

    if cnt.get("tty"):
        podman_args.append("--tty")
    if cnt.get("privileged"):
        podman_args.append("--privileged")
    if cnt.get("pid"):
        podman_args.extend(["--pid", cnt["pid"]])
    pull_policy = cnt.get("pull_policy")
    if pull_policy is not None and pull_policy != "build":
        podman_args.append(f"--pull={pull_policy}")
    if cnt.get("restart") is not None:
        podman_args.extend(["--restart", cnt["restart"]])
    container_to_ulimit_args(cnt, podman_args)
    container_to_res_args(cnt, podman_args)
    # currently podman shipped by fedora does not package this
    if cnt.get("init"):
        podman_args.append("--init")
    if cnt.get("init-path"):
        podman_args.extend(["--init-path", cnt["init-path"]])

    ipc = cnt.get("ipc")
    if ipc is not None:
        if not isinstance(ipc, str):
            raise ValueError(f"invalid ipc mode [{ipc}]")

        mode, colon, param = ipc.partition(":")

        if (mode in ("", "host", "none", "private", "shareable") and not colon) or (
            mode in ("container", "ns") and param
        ):
            podman_args.extend(["--ipc", ipc])
        elif mode == "service" and param:
            if param not in compose.container_names_by_service:
                raise ValueError(f"invalid ipc mode [{ipc}], service [{param}] does not exist.")
            other_cnt = compose.container_names_by_service[param][0]
            podman_args.extend(["--ipc", "container:" + other_cnt])
        else:
            raise ValueError(f"invalid ipc mode [{ipc}]")

    entrypoint = cnt.get("entrypoint")
    if entrypoint is not None:
        if isinstance(entrypoint, str):
            entrypoint = shlex.split(entrypoint)
        podman_args.extend(["--entrypoint", json.dumps(entrypoint)])
    platform = cnt.get("platform")
    if platform is not None:
        podman_args.extend(["--platform", platform])
    if cnt.get("runtime"):
        podman_args.extend(["--runtime", cnt["runtime"]])

    cpuset = cnt.get("cpuset")
    if cpuset is not None:
        podman_args.extend(["--cpuset-cpus", cpuset])

    # WIP: healthchecks are still work in progress
    healthcheck = cnt.get("healthcheck", {})
    if not isinstance(healthcheck, dict):
        raise ValueError("'healthcheck' must be a key-value mapping")
    healthcheck_disable = healthcheck.get("disable", False)
    healthcheck_test = healthcheck.get("test")
    if healthcheck_disable:
        healthcheck_test = ["NONE"]
    if healthcheck_test:
        # If it's a string, it's equivalent to specifying CMD-SHELL
        if isinstance(healthcheck_test, str):
            # podman does not add shell to handle command with whitespace
            podman_args.extend([
                "--health-cmd",
                json.dumps(["CMD-SHELL", healthcheck_test]),
            ])
        elif is_list(healthcheck_test):
            healthcheck_test = healthcheck_test.copy()
            # If it's a list, first item is either NONE, CMD or CMD-SHELL.
            healthcheck_type = healthcheck_test.pop(0)
            if healthcheck_type == "NONE":
                podman_args.append("--no-healthcheck")
            elif healthcheck_type == "CMD":
                podman_args.extend(["--health-cmd", json.dumps(healthcheck_test)])
            elif healthcheck_type == "CMD-SHELL":
                if len(healthcheck_test) != 1:
                    raise ValueError("'CMD-SHELL' takes a single string after it")
                podman_args.extend(["--health-cmd", json.dumps(healthcheck_test)])
            else:
                raise ValueError(
                    f"unknown healthcheck test type [{healthcheck_type}], "
                    "expecting NONE, CMD or CMD-SHELL."
                )
        else:
            raise ValueError("'healthcheck.test' either a string or a list")

    # interval, timeout, start_period, and start_interval are specified as durations.
    if "interval" in healthcheck:
        podman_args.extend(["--health-interval", healthcheck["interval"]])
    if "timeout" in healthcheck:
        podman_args.extend(["--health-timeout", healthcheck["timeout"]])
    if "start_period" in healthcheck:
        podman_args.extend(["--health-start-period", healthcheck["start_period"]])
    if "start_interval" in healthcheck:
        podman_args.extend(["--health-startup-interval", healthcheck["start_interval"]])

    # convert other parameters to string
    if "retries" in healthcheck:
        podman_args.extend(["--health-retries", str(healthcheck["retries"])])

    # handle podman extension
    if "x-podman" in cnt:
        raise ValueError(
            "Configuration under x-podman has been migrated to x-podman.uidmaps and "
            "x-podman.gidmaps fields"
        )

    rootfs_mode = False
    for uidmap in cnt.get("x-podman.uidmaps", []):
        podman_args.extend(["--uidmap", uidmap])
    for gidmap in cnt.get("x-podman.gidmaps", []):
        podman_args.extend(["--gidmap", gidmap])
    if cnt.get("x-podman.no_hosts", False):
        podman_args.extend(["--no-hosts"])
    if "x-podman.passwd" in cnt:
        # --passwd defaults to true
        podman_args.extend([f"--passwd={'false' if not cnt['x-podman.passwd'] else 'true'}"])
    rootfs = cnt.get("x-podman.rootfs")
    if rootfs is not None:
        rootfs_mode = True
        podman_args.extend(["--rootfs", rootfs])
        log.warning("WARNING: x-podman.rootfs and image both specified, image field ignored")

    if not rootfs_mode:
        podman_args.append(cnt["image"])  # command, ..etc.
    command = cnt.get("command")
    if command is not None:
        if isinstance(command, str):
            podman_args.extend(shlex.split(command))
        else:
            podman_args.extend([str(i) for i in command])
    return podman_args
