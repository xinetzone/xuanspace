# SPDX-License-Identifier: GPL-2.0-only
"""服务依赖图与条件等待（翻译自 podman_compose.py 第 1685-1756、3812-3920 行）。

- ``rec_deps``：递归展开某服务的全部传递依赖（带回环规避）。
- ``calc_dependents``：反向计算 ``_dependents`` 集合。
- ``flat_deps``：规范化后服务表的依赖图构建入口。
- ``check_dep_conditions`` / ``_validate_completed_successfully``：
  上游位于 up 命令辅助区，仅依赖鸭子类型的引擎句柄（``podman``、
  ``podman_version``、``container_names_by_service``），逻辑归属依赖就绪编排，
  按 T4 测试 ``test_depends_on.py`` 的要求置于本模块。

``compose`` 形参一律为 ``Any``：引擎层（T6）不得被本层导入。
"""

import asyncio
import json
from typing import Any

from .compat import is_list, strverscmp_lt
from .logging_utils import log
from .model import ServiceDependency, ServiceDependencyCondition
from .runner import CalledProcessError
from .types import DependField

__all__ = [
    "calc_dependents",
    "check_dep_conditions",
    "flat_deps",
    "rec_deps",
]


def rec_deps(
    services: dict[str, Any], service_name: str, start_point: str | None = None
) -> set[ServiceDependency]:
    """
    return all dependencies of service_name recursively
    """
    if not start_point:
        start_point = service_name
    deps = services[service_name]["_deps"]
    for dep_name in deps.copy():
        # avoid A depends on A
        if dep_name.name == service_name:
            continue
        dep_srv = services.get(dep_name.name)
        if not dep_srv:
            continue
        # NOTE: avoid creating loops, A->B->A
        if any(start_point == x.name for x in dep_srv["_deps"]):
            continue
        new_deps = rec_deps(services, dep_name.name, start_point)
        deps.update(new_deps)
    return deps


def calc_dependents(services: dict[str, Any]) -> None:
    for name, srv in services.items():
        deps: set[ServiceDependency] = srv.get("_deps", set())
        for dep in deps:
            if dep.name in services:
                services[dep.name].setdefault(DependField.DEPENDENTS, set()).add(
                    ServiceDependency(name, dep.condition.value)
                )


def flat_deps(services: dict[str, Any], with_extends: bool = False) -> None:
    """
    create dependencies "_deps" or update it recursively for all services
    """
    for name, srv in services.items():
        # parse dependencies for each service
        deps: set[ServiceDependency] = set()
        srv["_deps"] = deps
        # TODO: manage properly the dependencies coming from base services when extended
        if with_extends:
            ext = srv.get("extends", {}).get("service")
            if ext:
                if ext != name:
                    deps.add(ServiceDependency(ext, "service_started"))
                continue

        # the compose file has been normalized. depends_on, if exists, can only be a dictionary
        # the normalization adds a "service_started" condition by default
        deps_ls = srv.get("depends_on", {})
        deps_ls = [ServiceDependency(k, v["condition"]) for k, v in deps_ls.items()]
        deps.update(deps_ls)
        # parse link to get service name and remove alias
        links_ls = srv.get("links", [])
        if not is_list(links_ls):
            links_ls = [links_ls]
        deps.update([ServiceDependency(c.split(":")[0], "service_started") for c in links_ls])
        for c in links_ls:
            if ":" in c:
                dep_name, dep_alias = c.split(":")
                if "_aliases" not in services[dep_name]:
                    services[dep_name]["_aliases"] = set()
                services[dep_name]["_aliases"].add(dep_alias)

    # expand the dependencies on each service
    for name, srv in services.items():
        rec_deps(services, name)

    calc_dependents(services)


async def _validate_completed_successfully(
    compose: Any, container_names: list[str]
) -> None:
    # Poll until all containers have left the 'created' state
    # This prevents podman wait from racing against container startup
    last_log_time = 0.0
    while True:
        try:
            statuses_raw = await compose.podman.output(
                [], "inspect", ["--format={{.State.Status}}"] + container_names
            )
            statuses = statuses_raw.decode().split()
            if all(s != "created" for s in statuses if s):
                break
        except CalledProcessError as exc:
            log.debug(
                "podman inspect failed while polling for created states: %s",
                exc,
            )

        now = asyncio.get_event_loop().time()
        if now - last_log_time >= 1.0:
            log.debug(
                "Waiting for dependency containers to leave 'created' state: %s",
                ", ".join(container_names),
            )
            last_log_time = now
        await asyncio.sleep(0.05)

    # podman does not actually support value "service_completed_successfully"
    # default value "stopped" is sent instead
    await compose.podman.output([], "wait", ["--condition=stopped"] + container_names)

    for container_name in container_names:
        try:
            inspect_output = await compose.podman.output([], "inspect", [container_name])
        except CalledProcessError as exc:
            raise RuntimeError(
                f"Container {container_name} disappeared after waiting for stop"
            ) from exc
        container_info = json.loads(inspect_output)[0]

        exit_code = container_info.get("State", {}).get("ExitCode", -1)
        if exit_code != 0:
            error_msg = (
                f"Container {container_name} didn't complete successfully: exit code {exit_code}"
            )
            log.error(error_msg)
            raise RuntimeError(error_msg)


async def check_dep_conditions(compose: Any, deps: set) -> None:
    """Enforce that all specified conditions in deps are met"""
    if not deps:
        return

    for condition in ServiceDependencyCondition:
        deps_cd = []
        for d in deps:
            if d.condition == condition:
                if (
                    d.condition
                    in (ServiceDependencyCondition.HEALTHY, ServiceDependencyCondition.UNHEALTHY)
                ) and (
                    compose.podman_version is not None
                    and strverscmp_lt(compose.podman_version, "4.6.0")
                ):
                    log.warning(
                        "Ignored %s condition check due to podman %s doesn't support %s!",
                        d.name,
                        compose.podman_version,
                        condition.value,
                    )
                    continue

                deps_cd.extend(compose.container_names_by_service[d.name])

        if deps_cd:

            async def wait_one(
                d_cnt: str, condition: ServiceDependencyCondition = condition
            ) -> None:
                while True:
                    try:
                        if condition == ServiceDependencyCondition.SERVICE_COMPLETED_SUCCESSFULLY:
                            await _validate_completed_successfully(compose, [d_cnt])
                        else:
                            await compose.podman.output(
                                [], "wait", [f"--condition={condition.value}", d_cnt]
                            )
                        log.debug(
                            "dependency for condition %s has been fulfilled on container %s",
                            condition.value,
                            d_cnt,
                        )
                        break
                    except CalledProcessError as _exc:
                        output = list(
                            ((_exc.stdout or b"") + (_exc.stderr or b"")).decode().split("\n")
                        )
                        log.debug(
                            'Podman wait returned an error (%d) when executing "%s": %s',
                            _exc.returncode,
                            _exc.cmd,
                            output,
                        )
                    await asyncio.sleep(1)

            await asyncio.gather(*(wait_one(cnt) for cnt in deps_cd))
