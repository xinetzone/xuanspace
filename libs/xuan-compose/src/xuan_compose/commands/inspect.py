# SPDX-License-Identifier: GPL-2.0-only
"""查询/检查类命令（翻译自 podman_compose.py 第 3324-3375、4530-4541、
4795-4814、4876-4935 行）：ls/ps/config/port/stats/images。
"""

import argparse
import json
import os
from typing import Any

__all__ = [
    "list_running_projects",
    "compose_ps",
    "compose_config",
    "compose_port",
    "compose_stats",
    "compose_images",
]


async def list_running_projects(compose: Any, args: argparse.Namespace) -> None:
    img_containers = [cnt for cnt in compose.containers if "image" in cnt]
    parsed_args = vars(args)
    _format = parsed_args.get("format", "table")
    data: list[Any] = []
    if _format == "table":
        data.append(["NAME", "STATUS", "CONFIG_FILES"])

    for img in img_containers:
        try:
            name = img["name"]
            output = await compose.podman.output(
                [],
                "inspect",
                [
                    name,
                    "--format",
                    '''
                    {{ .State.Status }}
                    {{ .State.Running }}
                    {{ index .Config.Labels "com.docker.compose.project.working_dir" }}
                    {{ index .Config.Labels "com.docker.compose.project.config_files" }}
                    ''',
                ],
            )
            command_output = output.decode().split()
            running = bool(json.loads(command_output[1]))
            status = f"{command_output[0]}({1 if running else 0})"
            path = os.path.join(command_output[2], command_output[3])

            if _format == "table":
                if isinstance(command_output, list):
                    data.append([name, status, path])

            elif _format == "json":
                # Replicate how docker compose returns the list
                json_obj = {"Name": name, "Status": status, "ConfigFiles": path}
                data.append(json_obj)
        except Exception:
            break

    if _format == "table":
        column_widths = [max(map(len, column)) for column in zip(*data)]

        for row in data:
            formatted_row = [cell.ljust(width) for cell, width in zip(row, column_widths)]
            formatted_row[-2:] = ["\t".join(formatted_row[-2:]).strip()]
            print("\t".join(formatted_row))

    elif _format == "json":
        print(data)


async def compose_ps(compose: Any, args: argparse.Namespace) -> None:
    ps_args = ["-a", "--filter", f"label=io.podman.compose.project={compose.project_name}"]
    if args.quiet is True:
        ps_args.extend(["--format", "{{.ID}}"])
    elif args.format:
        ps_args.extend(["--format", args.format])

    await compose.podman.run(
        [],
        "ps",
        ps_args,
    )


async def compose_config(compose: Any, args: argparse.Namespace) -> None:
    if args.services:
        for service in compose.services:
            if not args.quiet:
                print(service)
        return
    if not args.quiet:
        print(compose.merged_yaml)


async def compose_port(compose: Any, args: argparse.Namespace) -> None:
    compose.assert_services(args.service)
    containers = compose.container_names_by_service[args.service]
    output = await compose.podman.output([], "inspect", [containers[args.index - 1]])
    inspect_json = json.loads(output.decode("utf-8"))
    private_port = str(args.private_port) + "/" + args.protocol
    host_port = inspect_json[0]["NetworkSettings"]["Ports"][private_port][0]["HostPort"]
    print(host_port)


async def compose_stats(compose: Any, args: argparse.Namespace) -> None:
    container_names_by_service = compose.container_names_by_service
    if not args.services:
        args.services = container_names_by_service.keys()
    targets = []
    podman_args = []
    if args.interval:
        podman_args.extend(["--interval", args.interval])
    if args.format:
        podman_args.extend(["--format", args.format])
    if args.no_reset:
        podman_args.append("--no-reset")
    if args.no_stream:
        podman_args.append("--no-stream")

    for service in args.services:
        targets.extend(container_names_by_service[service])
    for target in targets:
        podman_args.append(target)

    try:
        await compose.podman.run([], "stats", podman_args)
    except KeyboardInterrupt:
        pass


async def compose_images(compose: Any, args: argparse.Namespace) -> None:
    img_containers = [cnt for cnt in compose.containers if "image" in cnt]
    data = []
    if args.quiet is True:
        for img in img_containers:
            name = img["name"]
            output = await compose.podman.output([], "images", ["--quiet", img["image"]])
            data.append(output.decode("utf-8").split())
    else:
        data.append(["CONTAINER", "REPOSITORY", "TAG", "IMAGE ID", "SIZE", ""])
        for img in img_containers:
            name = img["name"]
            output = await compose.podman.output(
                [],
                "images",
                [
                    "--format",
                    "table " + name + " {{.Repository}} {{.Tag}} {{.ID}} {{.Size}}",
                    "-n",
                    img["image"],
                ],
            )
            data.append(output.decode("utf-8").split())

    # Determine the maximum length of each column
    column_widths = [max(map(len, column)) for column in zip(*data)]

    # Print each row
    for row in data:
        # Format each cell using the maximum column width
        formatted_row = [cell.ljust(width) for cell, width in zip(row, column_widths)]
        formatted_row[-2:] = ["".join(formatted_row[-2:]).strip()]
        print("\t".join(formatted_row))
