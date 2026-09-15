# SPDX-License-Identifier: GPL-2.0-only
"""``up`` / ``down`` 命令及其编排辅助（翻译自 podman_compose.py 第 802-835、
3767-3787、3812-3942、4109-4154、4158-4425、4444-4526 行）。

本模块承载命令层最重的运行时编排：pod 创建、依赖条件等待、容器启停、
镜像预拉/构建装配、存量容器重建判定、attach 任务监督与 SIGINT 收口。
``compose`` 形参为 ``Any`` 鸭子类型（与翻译层惯例一致，T10 差异表登记）。

依赖条件检查（check_dep_conditions/_validate_completed_successfully）上游位于
up 命令辅助区，T4 已按 test_depends_on.py 归属迁入执行层 dependencies.py；
本模块从那里再导入 check_dep_conditions 保持符号面（handler 经
compose.commands 互调，测试亦从本模块导入），不重复实现（单一事实源）。
"""

import argparse
import asyncio
import os
import signal
import sys
from typing import Any

from ..compat import STOP_GRACE_PERIOD, str_to_seconds, strverscmp_lt
from ..dependencies import check_dep_conditions
from ..logging_utils import log
from ..logs import _task_cancelled
from ..pull import prepare_images
from ..runner import CalledProcessError, wait_with_timeout
from ..translate.container_args import container_to_args
from ..translate.run_args import deps_from_container, get_excluded, get_volume_names, is_local
from ..types import DependField

__all__ = [
    "pod_exists",
    "create_pods",
    "create_secrets_from_environment",
    "check_dep_conditions",
    "run_container",
    "wait_for_container_running_healthy",
    "compose_up",
    "compose_down",
]


async def pod_exists(compose: Any, name: str) -> bool:
    exit_code = await compose.podman.run([], "pod", ["exists", name])
    return exit_code == 0


async def create_pods(compose: Any) -> None:
    for pod in compose.pods:
        if await pod_exists(compose, pod["name"]):
            continue

        podman_args = [
            "create",
            "--name=" + pod["name"],
        ] + compose.resolve_pod_args()

        ports = pod.get("ports", [])
        if isinstance(ports, str):
            ports = [ports]
        for i in ports:
            podman_args.extend(["-p", str(i)])
        await compose.podman.run([], "pod", podman_args)


async def create_secrets_from_environment(compose: Any) -> None:
    if not compose.declared_secrets:
        return
    for secret_name in compose.declared_secrets.keys():
        secret_environment = compose.declared_secrets[secret_name].get("environment")
        if secret_environment:
            secret_environment_value = os.getenv(secret_environment)

            if secret_environment_value is None:
                raise ValueError(
                    f"Environment variable '{secret_environment}' required"
                    + " by secret '{secret_name}' is not set in the process environment."
                )

            log.debug(
                "attempting creation of secret '%s' set to '%s'",
                secret_name,
                secret_environment_value,
            )

            assert compose.project_name is not None

            await compose.podman.run(
                [],
                "secret",
                [
                    "create",
                    "--label",
                    "io.podman.compose.project=" + compose.project_name,
                    "--env",
                    f"{compose.project_name}_{secret_name}",
                    secret_environment,
                ],
            )


async def run_container(
    compose: Any,
    name: str,
    deps: set[Any],
    command: tuple[Any, ...],
    log_formatter: str | None = None,
    suppress_output: bool = False,
) -> int | None:
    """runs a container after waiting for its dependencies to be fulfilled"""

    # wait for the dependencies to be fulfilled
    if "start" in command:
        log.debug("Checking dependencies prior to container %s start", name)
        await check_dep_conditions(compose, deps)

    # start the container
    log.debug("Starting task for container %s", name)
    return await compose.podman.run(  # type: ignore[misc]
        *command, log_formatter=log_formatter, suppress_output=suppress_output
    )


async def wait_for_container_running_healthy(
    compose: Any, args: argparse.Namespace
) -> None:
    if compose.podman_version is not None and strverscmp_lt(compose.podman_version, "4.6.0"):
        log.warning("Ignore --wait due to podman %s doesn't support it!", compose.podman_version)
        return

    log.info("waiting for all containers to be running|healthy")

    # distinguish between containers that have a healthcheck and those that don't
    cnt_with_healthcheck = []
    cnt_without_healthcheck = []
    for cnt in compose.containers:
        if "healthcheck" in cnt:
            cnt_with_healthcheck.append(cnt["name"])
        else:
            cnt_without_healthcheck.append(cnt["name"])

    async def run_podman_wait() -> None:
        # wait for running state of containers without a healthcheck
        if cnt_without_healthcheck:
            await compose.podman.run(
                [],
                "wait",
                [
                    "--condition=running",
                    "--ignore",
                    *cnt_without_healthcheck,
                ],
            )
        # wait for healthy state of containers with a healthcheck
        if cnt_with_healthcheck:
            await compose.podman.run(
                [],
                "wait",
                [
                    "--condition=healthy",
                    "--ignore",
                    *cnt_with_healthcheck,
                ],
            )

    # if --wait-timeout is not set None is used, which means no timeout
    # https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
    # the CancelledError is handled in the compose.podman.run() method
    await wait_with_timeout(run_podman_wait(), timeout=args.wait_timeout)


async def compose_up(compose: Any, args: argparse.Namespace) -> int | None:
    excluded = get_excluded(compose, args)
    no_attach_services = set(args.no_attach)
    unknown_no_attach_services = no_attach_services - set(compose.services)
    if unknown_no_attach_services:
        log.error("no such service: %s", sorted(unknown_no_attach_services)[0])
        return 1

    exit_code = await prepare_images(compose, args, excluded)
    if exit_code != 0:
        log.error("Prepare images failed")
        if not args.dry_run:
            return exit_code

    # if needed, tear down existing containers

    assert compose.project_name is not None, "Project name must be set before running up command"
    existing_containers = await compose.podman.existing_containers(compose.project_name)
    recreate_services: set[str] = set()
    running_services = {c.service_name for c in existing_containers.values() if not c.exited}

    await create_secrets_from_environment(compose)

    if existing_containers:
        if args.force_recreate and args.no_recreate:
            log.error(
                "Cannot use --force-recreate and --no-recreate at the same time, "
                "please remove one of them"
            )
            return 1

        if not args.no_recreate:
            requested_services = set(args.services) if args.services else set()
            always_recreate_deps = getattr(args, "always_recreate_deps", False)

            # resolve current local image IDs for services with running containers
            current_image_ids: dict[str, str] = {}
            for c in existing_containers.values():
                if (
                    c.service_name in excluded
                    or c.service_name not in compose.services
                    or not c.image_id
                ):
                    continue
                service = compose.services[c.service_name]
                image = service.get("image")
                if image and image not in current_image_ids:
                    try:
                        img_id = await compose.podman.output(
                            [], "inspect", ["-t", "image", "-f", "{{.Id}}", image]
                        )
                        current_image_ids[image] = img_id.decode().strip()
                    except CalledProcessError:
                        pass

            for c in existing_containers.values():
                if (
                    c.service_name in excluded
                    or c.service_name not in compose.services  # orphaned container
                ):
                    continue

                service = compose.services[c.service_name]
                force_this = args.force_recreate and (
                    not requested_services
                    or c.service_name in requested_services
                    or always_recreate_deps
                )

                image_changed = False
                image = service.get("image")
                if image and c.image_id:
                    local_id = current_image_ids.get(image, "")
                    if local_id and local_id != c.image_id:
                        log.info(
                            "Image changed for service %s (%s), will recreate",
                            c.service_name,
                            image,
                        )
                        image_changed = True

                if force_this or image_changed or c.config_hash != compose.config_hash(service):
                    recreate_services.add(c.service_name)

                    # Running dependents of service are removed by down command
                    # so we need to recreate and start them too
                    dependents = {
                        dep.name
                        for dep in service.get(DependField.DEPENDENTS, [])
                        if dep.name in running_services
                    }
                    if dependents:
                        log.debug(
                            "Service %s's dependents should be recreated and running again: %s",
                            c.service_name,
                            dependents,
                        )
                        recreate_services.update(dependents)
                        excluded = excluded - dependents

        log.debug("** excluding update: %s", excluded)
        log.debug("Prepare to recreate services: %s", recreate_services)

        teardown_needed = bool(recreate_services)

        if teardown_needed:
            log.info("tearing down existing containers: ...")
            down_args = argparse.Namespace(
                **dict(args.__dict__, volumes=False, rmi=None, services=recreate_services)
            )
            await compose.commands["down"](compose, down_args)
            log.info("tearing down existing containers: done\n\n")

    await create_pods(compose)

    log.info("creating missing containers: ...")

    create_error_codes: list[int | None] = []
    for cnt in compose.containers:
        if cnt["_service"] in excluded or (
            cnt["name"] in existing_containers and cnt["_service"] not in recreate_services
        ):
            log.debug("** skipping create: %s", cnt["name"])
            continue
        if getattr(args, "no_hosts", False):
            cnt["x-podman.no_hosts"] = True
        podman_args = await container_to_args(compose, cnt, detached=False, no_deps=args.no_deps)
        exit_code = await compose.podman.run([], "create", podman_args)
        create_error_codes.append(exit_code)

    if args.dry_run:
        return None

    if args.no_start:
        # return first error code from create calls, if any
        return next((code for code in create_error_codes if code is not None and code != 0), 0)

    if args.detach:
        log.info("starting containers (detached): ...")
        start_error_codes: list[int | None] = []
        for cnt in compose.containers:
            if cnt["_service"] in excluded:
                log.debug("** skipping start: %s", cnt["name"])
                continue
            exit_code = await run_container(
                compose, cnt["name"], deps_from_container(args, cnt), ([], "start", [cnt["name"]])
            )
            start_error_codes.append(exit_code)

        if args.wait:
            await wait_for_container_running_healthy(compose, args)

        # return first error code from start calls, if any
        return next((code for code in start_error_codes if code is not None and code != 0), 0)

    log.info("starting containers (attached): ...")

    # TODO: handle already existing
    # TODO: if error creating do not enter loop
    # TODO: colors if sys.stdout.isatty()
    exit_code_from = args.__dict__.get("exit_code_from")
    if exit_code_from:
        args.abort_on_container_exit = True

    max_service_length = 0
    for cnt in compose.containers:
        curr_length = len(cnt["_service"])
        max_service_length = curr_length if curr_length > max_service_length else max_service_length

    tasks: set[asyncio.Task[Any]] = set()

    async def handle_sigint() -> None:
        log.info("Caught SIGINT or Ctrl+C, shutting down...")
        try:
            log.info("Shutting down gracefully, please wait...")
            down_args = argparse.Namespace(**dict(args.__dict__, volumes=False, rmi=None))
            await compose.commands["down"](compose, down_args)
        except Exception as e:
            log.error("Error during shutdown: %s", e)
        finally:
            for task in tasks:
                task.cancel()

    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(handle_sigint()))

    for i, cnt in enumerate(compose.containers):
        # Add colored service prefix to output by piping output through sed
        color_idx = i % len(compose.console_colors)
        color = compose.console_colors[color_idx]
        space_suffix = " " * (max_service_length - len(cnt["_service"]) + 1)
        log_formatter = "{}[{}]{}|\x1b[0m".format(color, cnt["_service"], space_suffix)
        if cnt["_service"] in excluded:
            log.debug("** skipping: %s", cnt["name"])
            continue

        if cnt["_service"] in no_attach_services:
            tasks.add(
                asyncio.create_task(
                    run_container(
                        compose,
                        cnt["name"],
                        deps_from_container(args, cnt),
                        ([], "start", ["-a", cnt["name"]]),
                        suppress_output=True,
                    ),
                    name=cnt["_service"],
                )
            )
            continue

        tasks.add(
            asyncio.create_task(
                run_container(
                    compose,
                    cnt["name"],
                    deps_from_container(args, cnt),
                    ([], "start", ["-a", cnt["name"]]),
                    log_formatter=log_formatter,
                ),
                name=cnt["_service"],
            )
        )

    exit_code = 0
    exiting = False
    first_failed_task = None

    while tasks:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        if args.abort_on_container_failure and first_failed_task is None:
            # Generally a single returned item when using asyncio.FIRST_COMPLETED, but that's not
            # guaranteed. If multiple tasks finish at the exact same time the choice of which
            # finished "first" is arbitrary
            for t in done:
                if t.result() != 0:
                    first_failed_task = t

        if args.abort_on_container_exit or first_failed_task:
            if not exiting:
                # If 2 containers exit at the exact same time, the cancellation of the other ones
                # cause the status to overwrite. Sleeping for 1 seems to fix this and make it match
                # docker-compose
                await asyncio.sleep(1)
                for t in tasks:
                    if not _task_cancelled(t):
                        t.cancel()
            t_: asyncio.Task[Any]
            exiting = True
            if first_failed_task:
                # Matches docker-compose behaviour, where the exit code of the task that triggered
                # the cancellation is always propagated when aborting on failure
                exit_code = first_failed_task.result()
            else:
                for t_ in done:
                    if t_.get_name() == exit_code_from:
                        exit_code = t_.result()
    return exit_code


async def compose_down(compose: Any, args: argparse.Namespace) -> None:
    excluded = get_excluded(compose, args, DependField.DEPENDENTS)
    podman_args: list[str] = []
    timeout_global = getattr(args, "timeout", None)
    containers = list(reversed(compose.containers))

    down_tasks = []
    for cnt in containers:
        if cnt["_service"] in excluded:
            continue
        podman_stop_args = [*podman_args]
        timeout = timeout_global
        if timeout is None:
            timeout_str = cnt.get("stop_grace_period", STOP_GRACE_PERIOD)
            timeout = str_to_seconds(timeout_str)
        if timeout is not None:
            podman_stop_args.extend(["-t", str(timeout)])
        down_tasks.append(
            asyncio.create_task(
                compose.podman.run([], "stop", [*podman_stop_args, cnt["name"]]), name=cnt["name"]
            )
        )
    await asyncio.gather(*down_tasks)
    for cnt in containers:
        if cnt["_service"] in excluded:
            continue
        await compose.podman.run([], "rm", [cnt["name"]])

    orphaned_images = set()
    if args.remove_orphans:
        orphaned_containers = (
            (
                await compose.podman.output(
                    [],
                    "ps",
                    [
                        "--filter",
                        f"label=io.podman.compose.project={compose.project_name}",
                        "-a",
                        "--format",
                        "{{ .Image }} {{ .Names }}",
                    ],
                )
            )
            .decode("utf-8")
            .splitlines()
        )
        orphaned_images = {item.split()[0] for item in orphaned_containers}
        names = {item.split()[1] for item in orphaned_containers}
        for name in names:
            await compose.podman.run([], "stop", [*podman_args, name])
        for name in names:
            await compose.podman.run([], "rm", [name])
    if args.volumes:
        vol_names_to_keep = set()
        for cnt in containers:
            if cnt["_service"] not in excluded:
                continue
            vol_names_to_keep.update(get_volume_names(compose, cnt))
        log.debug("keep %s", vol_names_to_keep)
        for volume_name in await compose.podman.volume_ls():
            if volume_name in vol_names_to_keep:
                continue
            await compose.podman.run([], "volume", ["rm", volume_name])
    if args.rmi:
        images_to_remove = set()
        for cnt in containers:
            if cnt["_service"] in excluded:
                continue
            if args.rmi == "local" and not is_local(cnt):
                continue
            images_to_remove.add(cnt["image"])
        images_to_remove.update(orphaned_images)
        log.debug("images to remove: %s", images_to_remove)
        await compose.podman.run([], "rmi", ["--ignore", "--force"] + list(images_to_remove))

    if excluded:
        return
    for pod in compose.pods:
        await compose.podman.run([], "pod", ["rm", pod["name"]])
    for network in await compose.podman.network_ls():
        await compose.podman.run([], "network", ["rm", network])
