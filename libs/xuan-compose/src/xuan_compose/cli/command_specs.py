# SPDX-License-Identifier: GPL-2.0-only
"""指令清单（P2-2）——24 条命令的单一登记点。

此前一条指令的声明散落三处：``commands.COMMAND_HANDLERS``（handler）、
``cli.parser.COMMAND_PARSERS``（parser 挂载序）、``cli.parser.COMMAND_HELP``
（help 文本）。三者靠人工同步，新增指令须同时改两个模块。

现在收敛为 :data:`COMMAND_SPECS`：每条 :class:`CommandSpec` 同时携带
handler、parser 函数元组与 help/description。新增指令只需在下方清单追加
一条登记，parser 树与引擎查表均由 :func:`build_registry` 派生：

- :func:`build_registry` → 供 CLI 装配（``build_parser`` 建 subparser、
  ``install_handlers`` 灌入 ``compose.commands``）；
- ``COMMAND_HANDLERS`` / ``COMMAND_PARSERS`` / ``COMMAND_HELP`` 是供断言
  与快照对拍使用的只读派生视图，不参与装配。

顺序约束：清单顺序 = 上游 ``@cmd_run`` 出现序 = subparser 创建序，
help 输出顺序依赖它，不可重排。
"""

from types import MappingProxyType

from ..commands.build import compose_build
from ..commands.inspect import (
    compose_config,
    compose_images,
    compose_port,
    compose_ps,
    compose_stats,
    list_running_projects,
)
from ..commands.lifecycle import (
    compose_kill,
    compose_pause,
    compose_restart,
    compose_start,
    compose_stop,
    compose_unpause,
    compose_wait,
)
from ..commands.logs import compose_logs
from ..commands.pullpush import compose_pull, compose_push
from ..commands.registry import CommandRegistry, CommandSpec, Handler, ParserFn
from ..commands.runexec import compose_cp, compose_exec, compose_run
from ..commands.systemd import compose_systemd
from ..commands.updown import compose_down, compose_up
from ..commands.version import compose_version
from .parser import (
    SYSTEMD_DESC,
    SYSTEMD_HELP,
    compose_build_parse,
    compose_build_up_parse,
    compose_config_parse,
    compose_down_parse,
    compose_exec_parse,
    compose_format_parse,
    compose_images_parse,
    compose_kill_parse,
    compose_logs_parse,
    compose_ls_parse,
    compose_parse_cp,
    compose_parse_timeout,
    compose_pause_unpause_parse,
    compose_port_parse,
    compose_ps_parse,
    compose_pull_parse,
    compose_push_parse,
    compose_run_parse,
    compose_stats_parse,
    compose_systemd_parse,
    compose_up_parse,
    compose_up_start_parse,
    compose_version_parse,
)

__all__ = [
    "COMMAND_HELP",
    "COMMAND_HANDLERS",
    "COMMAND_PARSERS",
    "COMMAND_SPECS",
    "build_registry",
]

#: 24 条指令的单一事实源（handler + parser 挂载序 + help/description）。
COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("ls", list_running_projects, (compose_ls_parse,), "List running compose projects"),
    CommandSpec("version", compose_version, (compose_version_parse,), "show version"),
    CommandSpec("wait", compose_wait, (), "wait running containers to stop"),
    CommandSpec(
        "systemd",
        compose_systemd,
        (compose_systemd_parse,),
        SYSTEMD_HELP,
        SYSTEMD_DESC,
    ),
    CommandSpec("pull", compose_pull, (compose_pull_parse,), "pull stack images"),
    CommandSpec("push", compose_push, (compose_push_parse,), "push stack images"),
    CommandSpec(
        "build",
        compose_build,
        (compose_build_up_parse, compose_build_parse),
        "build stack images",
    ),
    CommandSpec(
        "up",
        compose_up,
        (
            compose_up_parse,
            compose_build_up_parse,
            compose_build_parse,
            compose_up_start_parse,
        ),
        "Create and start the entire stack or some of its services",
    ),
    CommandSpec(
        "down",
        compose_down,
        (compose_down_parse, compose_parse_timeout, compose_build_parse),
        "tear down entire stack",
    ),
    CommandSpec("ps", compose_ps, (compose_ps_parse, compose_format_parse), "show status of containers"),
    CommandSpec(
        "run",
        compose_run,
        (compose_run_parse,),
        "create a container similar to a service to run a one-off command",
    ),
    CommandSpec(
        "cp",
        compose_cp,
        (compose_parse_cp,),
        "copy files/folders between a service container and the local filesystem",
    ),
    CommandSpec("exec", compose_exec, (compose_exec_parse,), "execute a command in a running container"),
    CommandSpec(
        "start",
        compose_start,
        (compose_build_parse, compose_up_start_parse),
        "start specific services",
    ),
    CommandSpec(
        "stop",
        compose_stop,
        (compose_parse_timeout, compose_build_parse),
        "stop specific services",
    ),
    CommandSpec(
        "restart",
        compose_restart,
        (compose_parse_timeout, compose_build_parse),
        "restart specific services",
    ),
    CommandSpec("logs", compose_logs, (compose_logs_parse,), "show logs from services"),
    CommandSpec("config", compose_config, (compose_config_parse,), "displays the compose file"),
    CommandSpec("port", compose_port, (compose_port_parse,), "Prints the public port for a port binding."),
    CommandSpec(
        "pause",
        compose_pause,
        (compose_pause_unpause_parse,),
        "Pause all running containers",
    ),
    CommandSpec(
        "unpause",
        compose_unpause,
        (compose_pause_unpause_parse,),
        "Unpause all running containers",
    ),
    CommandSpec(
        "kill",
        compose_kill,
        (compose_kill_parse,),
        "Kill one or more running containers with a specific signal",
    ),
    CommandSpec(
        "stats",
        compose_stats,
        (compose_stats_parse, compose_format_parse),
        "Display percentage of CPU, memory, network I/O, block I/O and PIDs for services.",
    ),
    CommandSpec(
        "images",
        compose_images,
        (compose_images_parse,),
        "List images used by the created containers",
    ),
)


def build_registry(specs: tuple[CommandSpec, ...] = COMMAND_SPECS) -> CommandRegistry:
    """按清单装配一个新注册表（每次返回独立实例，登记可撤销）。"""
    registry = CommandRegistry()
    registry.register_all(specs)
    return registry


#: 只读派生视图：handler 名表（断言/测试用，装配请走 build_registry）。
COMMAND_HANDLERS: MappingProxyType[str, Handler] = MappingProxyType({spec.name: spec.handler for spec in COMMAND_SPECS})

#: 只读派生视图：命令名 → parser 挂载序（``wait`` 上游无 parser，为空元组）。
COMMAND_PARSERS: MappingProxyType[str, tuple[ParserFn, ...]] = MappingProxyType(
    {spec.name: spec.parsers for spec in COMMAND_SPECS}
)

#: 只读派生视图：命令名 → 单行 help（systemd 由 handler docstring 派生）。
COMMAND_HELP: MappingProxyType[str, str] = MappingProxyType({spec.name: spec.help for spec in COMMAND_SPECS})
