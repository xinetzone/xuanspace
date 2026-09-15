# SPDX-License-Identifier: GPL-2.0-only
"""资源限制翻译：ulimit / CPU / 内存 / GPU（翻译自 podman_compose.py 第 692-716、957-1070 行）。"""

from typing import Any

from ..compat import try_float, try_int
from ..normalize import norm_as_dict, norm_ulimit

__all__ = [
    "ulimit_to_ulimit_args",
    "container_to_ulimit_args",
    "container_to_ulimit_build_args",
    "container_to_res_args",
    "container_to_gpu_res_args",
    "container_to_cpu_res_args",
]


def ulimit_to_ulimit_args(ulimit: str | dict[str, Any] | list[Any], podman_args: list[str]) -> None:
    if ulimit is not None:
        # ulimit can be a single value, i.e. ulimit: host
        if isinstance(ulimit, str):
            podman_args.extend(["--ulimit", ulimit])
        # or a dictionary or list:
        else:
            ulimit_dict = norm_as_dict(ulimit)
            ulimit_ls = [
                f"{ulimit_key}={norm_ulimit(inner_value)}"  # type: ignore[arg-type]
                for ulimit_key, inner_value in ulimit_dict.items()  # type: ignore[union-attr]
            ]
            for i in ulimit_ls:
                podman_args.extend(["--ulimit", i])


def container_to_ulimit_args(cnt: dict[str, Any], podman_args: list[str]) -> None:
    ulimit_to_ulimit_args(cnt.get("ulimits", []), podman_args)


def container_to_ulimit_build_args(cnt: dict[str, Any], podman_args: list[str]) -> None:
    build = cnt.get("build")

    if build is not None:
        ulimit_to_ulimit_args(build.get("ulimits", []), podman_args)


def container_to_res_args(cnt: dict[str, Any], podman_args: list[str]) -> None:
    container_to_cpu_res_args(cnt, podman_args)
    container_to_gpu_res_args(cnt, podman_args)


def container_to_gpu_res_args(cnt: dict[str, Any], podman_args: list[str]) -> None:
    # https://docs.docker.com/compose/gpu-support/
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html

    deploy = cnt.get("deploy", {})
    res = deploy.get("resources", {})
    reservations = res.get("reservations", {})
    devices = reservations.get("devices", [])
    gpu_on = False
    for device in devices:
        driver = device.get("driver")
        if driver is None:
            continue

        capabilities = device.get("capabilities")
        if capabilities is None:
            continue

        if driver != "nvidia" or "gpu" not in capabilities:
            continue

        count = device.get("count", "all")
        device_ids = device.get("device_ids", "all")
        if device_ids != "all" and len(device_ids) > 0:
            for device_id in device_ids:
                podman_args.extend((
                    "--device",
                    f"nvidia.com/gpu={device_id}",
                ))
            gpu_on = True
            continue

        if count != "all":
            for device_id in range(count):
                podman_args.extend((
                    "--device",
                    f"nvidia.com/gpu={device_id}",
                ))
            gpu_on = True
            continue

        podman_args.extend((
            "--device",
            "nvidia.com/gpu=all",
        ))
        gpu_on = True

    if gpu_on:
        podman_args.append("--security-opt=label=disable")


def container_to_cpu_res_args(cnt: dict[str, Any], podman_args: list[str]) -> None:
    # v2: https://docs.docker.com/compose/compose-file/compose-file-v2/#cpu-and-other-resources
    # cpus, cpu_shares, mem_limit, mem_reservation
    cpus_limit_v2 = try_float(cnt.get("cpus"), None)  # type: ignore[arg-type]
    cpu_shares_v2 = try_int(cnt.get("cpu_shares"), None)  # type: ignore[arg-type]
    mem_limit_v2 = cnt.get("mem_limit")
    mem_res_v2 = cnt.get("mem_reservation")
    # v3: https://docs.docker.com/compose/compose-file/compose-file-v3/#resources
    # spec: https://github.com/compose-spec/compose-spec/blob/master/deploy.md#resources
    # deploy.resources.{limits,reservations}.{cpus, memory}
    deploy = cnt.get("deploy", {})
    res = deploy.get("resources", {})
    limits = res.get("limits", {})
    cpus_limit_v3 = try_float(limits.get("cpus"), None)
    mem_limit_v3 = limits.get("memory")
    reservations = res.get("reservations", {})
    # cpus_res_v3 = try_float(reservations.get('cpus', None), None)
    mem_res_v3 = reservations.get("memory")
    # add args
    cpus = cpus_limit_v3 or cpus_limit_v2
    if cpus:
        podman_args.extend((
            "--cpus",
            str(cpus),
        ))
    if cpu_shares_v2:
        podman_args.extend((
            "--cpu-shares",
            str(cpu_shares_v2),
        ))
    mem = mem_limit_v3 or mem_limit_v2
    if mem:
        podman_args.extend((
            "-m",
            str(mem).lower(),
        ))
    mem_res = mem_res_v3 or mem_res_v2
    if mem_res:
        podman_args.extend((
            "--memory-reservation",
            str(mem_res).lower(),
        ))

    # Handle pids limit from both container level and deploy section
    pids_limit = cnt.get("pids_limit")
    deploy_pids = limits.get("pids")

    # Ensure consistency between pids_limit and deploy.resources.limits.pids
    if pids_limit is not None and deploy_pids is not None:
        if str(pids_limit) != str(deploy_pids):
            raise ValueError(
                f"Inconsistent PIDs limit: pids_limit ({pids_limit}) and "
                f"deploy.resources.limits.pids ({deploy_pids}) must be the same"
            )

    final_pids_limit = pids_limit if pids_limit is not None else deploy_pids
    if final_pids_limit is not None:
        podman_args.extend(["--pids-limit", str(final_pids_limit)])
