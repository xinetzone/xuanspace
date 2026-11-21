# SPDX-License-Identifier: GPL-2.0-only
"""CLI 参数解析层（翻译自 podman_compose.py 第 119-129、3149-3267、
4943-5525 行）。

上游用 ``@cmd_parse`` 装饰器在 import 期向模块级单例的
``commands[name]._parse_args`` 列表**追加** parser 函数（第 3305-3316 行）；
分层后改为无状态函数：

- 本模块保留 23 个 parser 函数本体（含 ``PullPolicyAction``），签名统一为
  ``fn(parser) -> None``，与上游逐行对等（option strings、action、default、
  nargs、choices、help 文本、注册顺序全部保留）；
- 命令名到 parser 函数元组的映射（挂载顺序精确复刻 append 顺序，
  ``wait`` 为空元组）由 P2-2 收敛到单一清单
  :data:`xuan_compose.cli.command_specs.COMMAND_SPECS`；
- ``build_parser(registry=...)``：纯函数构造全局 parser，不触碰引擎/argv
  （prog 由 argparse 内部按调用进程推导，属 FR-4 允许的唯一 prog 差异）；
- ``parse_args(engine, argv, registry=...)``：解析后的回填/分流逻辑
  （COMPOSE_ENV_FILES、--version、无子命令 help 退出、日志级别），并把
  结果写入 ``engine.global_args`` 供引擎运行时读取。
"""

import argparse
import os
import re
import sys
from collections.abc import Sequence
from typing import Any

from ..commands.registry import CommandRegistry
from ..commands.systemd import compose_systemd
from ..errors import PodmanComposeError
from ..logging_utils import configure_logging

__all__ = [
    "PODMAN_CMDS",
    "PullPolicyAction",
    "SYSTEMD_DESC",
    "SYSTEMD_HELP",
    "build_parser",
    "parse_args",
]

#: 上游第 119-129 行：为每个 podman 子命令生成全局 ``--podman-<cmd>-args``。
PODMAN_CMDS = (
    "pull",
    "push",
    "build",
    "inspect",
    "run",
    "start",
    "stop",
    "rm",
    "volume",
)

#: 环境变量布尔开关的真值集合（大小写不敏感）。
_ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str) -> bool:
    """读取布尔型环境变量开关（如 XC_NO_MACHINE_START）。"""
    return os.environ.get(name, "").strip().lower() in _ENV_TRUE_VALUES


class PullPolicyAction(argparse.Action):
    """翻译自上游第 5338-5352 行：--pull/--pull-always 的归一化动作。"""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        if option_string == "--pull-always":
            if values in (None, "true"):
                namespace.pull = "always"

            return

        namespace.pull = "newer" if values is None else values


def init_global_parser(parser: argparse.ArgumentParser) -> None:
    """翻译自上游 ``_init_global_parser``（第 3174-3267 行，静态方法）。"""
    parser.add_argument("-v", "--version", help="show version", action="store_true")
    parser.add_argument(
        "--in-pod",
        help=(
            "Specify pod usage:\n"
            "  'true'   - create/use a pod named pod_<project name>\n"
            "  'false'  - do not use a pod\n"
            "  '<name>' - create/use a custom pod with the given name"
        ),
        metavar="in_pod",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pod-args",
        help="custom arguments to be passed to `podman pod`",
        metavar="pod_args",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--env-file",
        help="Specify an alternate environment file (can be specified multiple times)",
        metavar="env_file",
        action="append",
        default=[],
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Specify an compose file (default: docker-compose.yml) or '-' to read from stdin.",
        metavar="file",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--profile",
        help="Specify a profile to enable",
        metavar="profile",
        action="append",
        default=[],
    )
    parser.add_argument(
        "-p",
        "--project-name",
        help="Specify an alternate project name (default: directory name)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--podman-path",
        help="Specify an alternate path to podman (default: use location in $PATH variable)",
        type=str,
        default="podman",
    )
    parser.add_argument(
        "--podman-args",
        help="custom global arguments to be passed to `podman`",
        metavar="args",
        action="append",
        default=[],
    )
    for podman_cmd in PODMAN_CMDS:
        parser.add_argument(
            f"--podman-{podman_cmd}-args",
            help=f"custom arguments to be passed to `podman {podman_cmd}`",
            metavar="args",
            action="append",
            default=[],
        )
    parser.add_argument(
        "--no-ansi",
        help="Do not print ANSI control characters",
        action="store_true",
    )
    parser.add_argument(
        "--no-cleanup",
        help="Do not stop and remove existing pod & containers",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        help="No action; perform a simulation of commands",
        action="store_true",
    )
    parser.add_argument("--parallel", type=int, default=os.environ.get("COMPOSE_PARALLEL_LIMIT", sys.maxsize))
    parser.add_argument(
        "--no-machine-start",
        help=(
            "Do not auto-start the default podman machine when the Podman service "
            "is unreachable (also enabled by XC_NO_MACHINE_START=1)"
        ),
        action="store_true",
        default=_env_flag("XC_NO_MACHINE_START"),
    )
    parser.add_argument(
        "--verbose",
        help="Print debugging output",
        action="store_true",
    )


# ---------------------------------------------------------------------------
# 23 个命令 parser 函数（翻译自上游第 4943-5525 行，顺序与装饰器出现序一致）
# ---------------------------------------------------------------------------


def compose_version_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="Format the output",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Shows only Podman Compose's version number",
    )


def compose_up_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Detached mode: Run container in the background, print new container name. \
            Incompatible with --abort-on-container-exit and --abort-on-container-failure.",
    )
    parser.add_argument("--no-color", action="store_true", help="Produce monochrome output.")
    parser.add_argument(
        "--quiet-pull",
        action="store_true",
        help="Pull without printing progress information.",
    )
    parser.add_argument("--no-deps", action="store_true", help="Don't start linked services.")
    parser.add_argument(
        "--no-attach",
        action="append",
        default=[],
        metavar="SERVICE",
        help="Do not attach to SERVICE.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Recreate containers even if their configuration and image haven't changed.",
    )
    parser.add_argument(
        "--always-recreate-deps",
        action="store_true",
        help="Recreate dependent containers. Incompatible with --no-recreate.",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="If containers already exist, don't recreate them. Incompatible with --force-recreate and -V.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Don't build an image, even if it's missing.",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Don't start the services after creating them.",
    )
    parser.add_argument("--build", action="store_true", help="Build images before starting containers.")
    parser.add_argument(
        "--abort-on-container-exit",
        action="store_true",
        help="Stops all containers if any container was stopped. Incompatible with -d and "
        "--abort-on-container-failure.",
    )
    parser.add_argument(
        "--abort-on-container-failure",
        action="store_true",
        help="Stops all containers if any container stops with a non-zero exit code. Incompatible "
        "with -d and --abort-on-container-exit.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=None,
        help="Use this timeout in seconds for container shutdown when attached or when containers "
        "are already running. (default: 10)",
    )
    parser.add_argument(
        "-V",
        "--renew-anon-volumes",
        action="store_true",
        help="Recreate anonymous volumes instead of retrieving data from the previous containers.",
    )
    parser.add_argument(
        "--remove-orphans",
        action="store_true",
        help="Remove containers for services not defined in the Compose file.",
    )
    # `--scale` argument needs to store as single value and not append,
    # as multiple scale values could be confusing.
    parser.add_argument(
        "--scale",
        metavar="SERVICE=NUM",
        help="Scale SERVICE to NUM instances. Overrides the `scale` setting in the Compose file if present.",
    )
    parser.add_argument(
        "--exit-code-from",
        metavar="SERVICE",
        type=str,
        default=None,
        help="Return the exit code of the selected service container. Implies --abort-on-container-exit.",
    )
    parser.add_argument(
        "--no-hosts",
        action="store_true",
        help="Do not modify the /etc/hosts file in the container.",
    )


def compose_down_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--volumes",
        action="store_true",
        default=False,
        help="Remove named volumes declared in the `volumes` section of the Compose file and "
        "anonymous volumes attached to containers.",
    )
    parser.add_argument(
        "--remove-orphans",
        action="store_true",
        help="Remove containers for services not defined in the Compose file.",
    )
    parser.add_argument(
        "--rmi",
        type=str,
        nargs="?",
        const="all",
        choices=["local", "all"],
        help="Remove images used by services. `local` remove only images that don't have a "
        "custom tag. (`local` or `all`)",
    )


def compose_run_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--build", action="store_true", help="Build images before starting containers.")
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Detached mode: Run container in the background, print new container name.",
    )
    parser.add_argument("--name", type=str, default=None, help="Assign a name to the container")
    parser.add_argument(
        "--entrypoint",
        type=str,
        default=None,
        help="Override the entrypoint of the image.",
    )
    parser.add_argument(
        "-e",
        "--env",
        metavar="KEY=VAL",
        action="append",
        help="Set an environment variable (can be used multiple times)",
    )
    parser.add_argument(
        "-l",
        "--label",
        metavar="KEY=VAL",
        action="append",
        help="Add or override a label (can be used multiple times)",
    )
    parser.add_argument("-u", "--user", type=str, default=None, help="Run as specified username or uid")
    parser.add_argument("--no-deps", action="store_true", help="Don't start linked services")
    parser.add_argument(
        "--rm",
        action="store_true",
        help="Remove container after run. Ignored in detached mode.",
    )
    parser.add_argument(
        "-p",
        "--publish",
        action="append",
        help="Publish a container's port(s) to the host (can be used multiple times)",
    )
    parser.add_argument(
        "--service-ports",
        action="store_true",
        help="Run command with the service's ports enabled and mapped to the host.",
    )
    parser.add_argument(
        "-v",
        "--volume",
        action="append",
        help="Bind mount a volume (can be used multiple times)",
    )
    parser.add_argument(
        "-T",
        action="store_true",
        help="Disable pseudo-tty allocation. By default `podman-compose run` allocates a TTY.",
    )
    parser.add_argument(
        "-w",
        "--workdir",
        type=str,
        default=None,
        help="Working directory inside the container",
    )
    parser.add_argument("service", metavar="service", nargs=None, help="service name")
    parser.add_argument(
        "cnt_command",
        metavar="command",
        nargs=argparse.REMAINDER,
        help="command and its arguments",
    )


def compose_exec_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Detached mode: Run container in the background, print new container name.",
    )
    parser.add_argument(
        "--privileged",
        action="store_true",
        default=False,
        help="Give the process extended Linux capabilities inside the container",
    )
    parser.add_argument("-u", "--user", type=str, default=None, help="Run as specified username or uid")
    parser.add_argument(
        "-T",
        action="store_true",
        help="Disable pseudo-tty allocation. By default `podman-compose run` allocates a TTY.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="Index of the container if there are multiple instances of a service",
    )
    parser.add_argument(
        "-e",
        "--env",
        metavar="KEY=VAL",
        action="append",
        help="Set an environment variable (can be used multiple times)",
    )
    parser.add_argument(
        "-w",
        "--workdir",
        type=str,
        default=None,
        help="Working directory inside the container",
    )
    parser.add_argument("service", metavar="service", nargs=None, help="service name")
    parser.add_argument(
        "cnt_command",
        metavar="command",
        nargs=argparse.REMAINDER,
        help="command and its arguments",
    )


def compose_parse_cp(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-a",
        "--archive",
        help=("Chown copied files to the primary uid/gid of the destination container (default=True)"),
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        help="Allow to overwrite directories with non-directories and vice versa (default=None)",
        default=None,
    )
    parser.add_argument(
        "src",
        metavar="SOURCE",
        help="Source path. Use SERVICE:PATH for container paths, or just PATH for local paths",
        default=None,
    )
    parser.add_argument(
        "dst",
        metavar="DESTINATION",
        help="Destination path. Use SERVICE:PATH for container paths, or just PATH for local paths",
        default=None,
    )


def compose_parse_timeout(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--timeout",
        help="Specify a shutdown timeout in seconds. ",
        type=int,
        default=None,
    )


def compose_logs_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Follow log output. The default is false",
    )
    parser.add_argument(
        "-l",
        "--latest",
        action="store_true",
        help="Act on the latest container podman is aware of",
    )
    parser.add_argument(
        "-n",
        "--names",
        action="store_true",
        help="Output the container name in the log",
    )
    parser.add_argument("--no-color", action="store_true", help="Produce monochrome output")
    parser.add_argument(
        "--no-log-prefix",
        action="store_true",
        help="Don't print prefix with container identifier in logs",
    )
    parser.add_argument("--since", help="Show logs since TIMESTAMP", type=str, default=None)
    parser.add_argument("-t", "--timestamps", action="store_true", help="Show timestamps.")
    parser.add_argument(
        "--tail",
        help="Number of lines to show from the end of the logs for each container.",
        type=str,
        default="all",
    )
    parser.add_argument("--until", help="Show logs until TIMESTAMP", type=str, default=None)
    parser.add_argument("services", metavar="services", nargs="*", default=None, help="service names")


def compose_systemd_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-a",
        "--action",
        choices=["register", "unregister", "create-unit", "list", "ls"],
        default="register",
        help="create systemd unit file or register compose stack to it",
    )


def compose_pull_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force-local",
        action="store_true",
        default=False,
        help="Also pull unprefixed images for services which have a build section",
    )
    parser.add_argument("services", metavar="services", nargs="*", help="services to pull")


def compose_push_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ignore-push-failures",
        action="store_true",
        help="Push what it can and ignores images with push failures. (not implemented)",
    )
    parser.add_argument("services", metavar="services", nargs="*", help="services to push")


def compose_ps_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-q", "--quiet", help="Only display container IDs", action="store_true")


def compose_build_up_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pull",
        help="Pull image policy (always|missing|never|newer)."
        " Set to 'newer' if specify --pull without a value."
        " default (pull_policy in compose file).",
        action=PullPolicyAction,
        nargs="?",
        choices=["always", "missing", "never", "newer"],
    )
    parser.add_argument(
        "--pull-always",
        help="Deprecated, use --pull=always instead",
        action=PullPolicyAction,
        nargs="?",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--build-arg",
        metavar="key=val",
        action="append",
        default=[],
        help="Set build-time variables for services.",
    )
    parser.add_argument(
        "--no-cache",
        help="Do not use cache when building the image.",
        action="store_true",
    )


def compose_build_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "services",
        metavar="services",
        nargs="*",
        default=None,
        help="affected services",
    )


def compose_up_start_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for services to be running|healthy. Implies detached mode.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=None,
        help="Maximum duration in seconds to wait for the project to be running|healthy",
    )


def compose_config_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-normalize", help="Don't normalize compose model.", action="store_true")
    parser.add_argument("--services", help="Print the service names, one per line.", action="store_true")
    parser.add_argument(
        "-q",
        "--quiet",
        help="Do not print config, only parse.",
        action="store_true",
    )


def compose_port_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="index of the container if there are multiple instances of a service",
    )
    parser.add_argument(
        "--protocol",
        choices=["tcp", "udp"],
        default="tcp",
        help="tcp or udp",
    )
    parser.add_argument("service", metavar="service", nargs=None, help="service name")
    parser.add_argument(
        "private_port",
        metavar="private_port",
        nargs=None,
        type=int,
        help="private port",
    )


def compose_pause_unpause_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("services", metavar="services", nargs="*", default=None, help="service names")


def compose_kill_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("services", metavar="services", nargs="*", default=None, help="service names")
    parser.add_argument(
        "-s",
        "--signal",
        type=str,
        help="Signal to send to the container (default 'KILL')",
    )
    parser.add_argument(
        "-a",
        "--all",
        help="Signal all running containers",
        action="store_true",
    )


def compose_images_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-q", "--quiet", help="Only display images IDs", action="store_true")


def compose_stats_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("services", metavar="services", nargs="*", default=None, help="service names")
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        help="Time in seconds between stats reports (default 5)",
    )
    parser.add_argument(
        "--no-reset",
        help="Disable resetting the screen between intervals",
        action="store_true",
    )
    parser.add_argument(
        "--no-stream",
        help="Disable streaming stats and only pull the first result",
        action="store_true",
    )


def compose_format_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        help="Pretty-print container statistics to JSON or using a Go template",
    )


def compose_ls_parse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--format",
        choices=["table", "json"],
        default="table",
        help="Format the output",
    )


def _help_desc_from_docstring(doc: str) -> tuple[str, str | None]:
    """复刻上游 ``cmd_run`` 装饰器的 help/desc 切分（第 3293-3299 行）。

    ``re.sub(r"^\\s+", "", doc)`` 只剥离整个字符串**开头**的空白（无
    MULTILINE），随后按第一个换行切两段：首行为 help、其余为 description。
    """
    help_desc = re.sub(r"^\s+", "", doc)
    if "\n" in help_desc:
        help_text, desc = help_desc.split("\n", 1)
        return help_text, desc
    return help_desc, None


# systemd 是唯一未在 ``@cmd_run`` 传 cmd_desc 的命令：help/desc 来自
# handler docstring，此处复用同一算法，保证与上游逐字对等。
SYSTEMD_HELP, SYSTEMD_DESC = _help_desc_from_docstring(compose_systemd.__doc__ or "")


def build_parser(
    command_descriptions: dict[str, str | None] | None = None,
    registry: CommandRegistry | None = None,
) -> argparse.ArgumentParser:
    """构造全局 argparse parser（翻译自 ``_parse_args`` 的前半段）。

    纯函数：不读 argv、不触碰引擎。subparser 顺序与登记序一致（即上游
    ``@cmd_run`` 出现序），另在首个位置插入上游的 ``help`` 伪子命令。
    ``registry`` 省略时按唯一清单
    :func:`~xuan_compose.cli.command_specs.build_registry` 装配；显式传入
    ``command_descriptions`` 时以该映射覆盖清单中的 description。
    """
    if registry is None:
        from .command_specs import build_registry

        registry = build_registry()
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    init_global_parser(parser)
    subparsers = parser.add_subparsers(title="command", dest="command")
    _ = subparsers.add_parser("help", help="show help")
    descriptions = (
        command_descriptions if command_descriptions is not None else {spec.name: spec.description for spec in registry}
    )
    for spec in registry:
        subparser = subparsers.add_parser(
            spec.name,
            help=spec.help,
            description=descriptions.get(spec.name),
        )
        for cmd_parser in spec.parsers:
            cmd_parser(subparser)
    return parser


def parse_args(
    engine: Any,
    argv: list[str] | None = None,
    registry: CommandRegistry | None = None,
) -> argparse.Namespace:
    """解析 argv 并完成上游 ``_parse_args`` 的全部回填/分流副作用。

    解析结果同时写入 ``engine.global_args``（引擎运行时经
    ``get_podman_args``/``resolve_pod_name`` 等读取），返回同一 Namespace。
    """
    parser = build_parser(registry=registry)
    global_args = parser.parse_args(argv)

    if global_args.command in ("up", "start"):
        wait_timeout = global_args.wait_timeout
        if wait_timeout is not None and wait_timeout < 0:
            raise PodmanComposeError("--wait-timeout must be a non-negative integer")

    if global_args.command == "up" and global_args.wait:
        incompatible = []
        if global_args.abort_on_container_exit:
            incompatible.append("--abort-on-container-exit")
        if global_args.abort_on_container_failure:
            incompatible.append("--abort-on-container-failure")
        if global_args.exit_code_from:
            incompatible.append("--exit-code-from")
        if incompatible:
            if len(incompatible) > 1:
                incompatible_options = ", ".join(incompatible[:-1]) + " or " + incompatible[-1]
            else:
                incompatible_options = incompatible[0]
            raise PodmanComposeError(f"--wait cannot be combined with {incompatible_options}")
        global_args.detach = True

    compose_env_files = os.environ.get("COMPOSE_ENV_FILES")
    if not global_args.env_file and compose_env_files:
        global_args.env_file = compose_env_files.split(",")

    if global_args.version:
        global_args.command = "version"
    if not global_args.command or global_args.command == "help":
        parser.print_help()
        sys.exit(-1)

    configure_logging(global_args.verbose)
    engine.global_args = global_args
    return global_args
