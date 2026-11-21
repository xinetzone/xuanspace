"""内置命令插件 ``xs.commands``（P1-3）。

命令来源为一张静态声明表：``cli.py`` 不再逐个 import 子命令，而是装配本插件、
从 :class:`~xs.plugins.registry.CommandRegistry` 取出登记项后构建 Typer 应用。

两条通道共用同一份清单：

* 静态通道：``tools/xs/pyproject.toml`` 的 ``[tool.xuan.plugin]``；
* 入口点通道：本模块的 :data:`MANIFEST`，由 ``[project.entry-points."xuan.plugins"]``
  指向本模块后被发现。

两者契约必须完全一致，由 ``tests/test_plugin_cmd.py`` 机检。
"""

from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import cast

from xuan_plugin.context import Context
from xuan_plugin.manifest import PluginManifest, ProvideDecl
from xuan_plugin.plugin import Plugin

from .registry import CommandEntry, CommandRegistry, CommandTarget


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """一条命令声明：能力键后缀、目标对象与登记类别。"""

    name: str
    target: str
    kind: str = "command"

    @property
    def key(self) -> str:
        return f"xuan:cli/commands#{self.name}"


COMMAND_SPECS = (
    CommandSpec("list", "xs.commands.list_cmd:list_projects"),
    CommandSpec("doctor", "xs.commands.doctor_cmd:doctor"),
    CommandSpec("init", "xs.commands.init_cmd:init_workspace"),
    CommandSpec("build", "xs.commands.build_cmd:build_project"),
    CommandSpec("new", "xs.commands.new_cmd:create_new_project"),
    CommandSpec("update", "xs.commands.update_cmd:update_cmd"),
    CommandSpec("py-compat", "xs.commands.py_compat_cmd:py_compat"),
    CommandSpec("affected", "xs.commands.affected_cmd:affected"),
    CommandSpec("toolchain", "xs.commands.toolchain_cmd:app", kind="group"),
    CommandSpec("deps", "xs.commands.deps_cmd:app", kind="group"),
    CommandSpec("doc", "xs.commands.doc_cmd:app", kind="group"),
    CommandSpec("version", "xs.commands.version_cmd:app", kind="group"),
    CommandSpec("meta", "xs.commands.meta_cmd:app", kind="group"),
    CommandSpec("archive", "xs.commands.archive_cmd:app", kind="group"),
    CommandSpec("lfs", "xs.commands.lfs_cmd:app", kind="group"),
    CommandSpec("plugin", "xs.commands.plugin_cmd:app", kind="group"),
)

MANIFEST = PluginManifest(
    id="xs.commands",
    version="0.1.0",
    provide=tuple(ProvideDecl(key=spec.key) for spec in COMMAND_SPECS),
    reversible=True,
    priority=100,
    lifecycle="reloadable",
    source="entry-point:xs.plugins.commands",
)


def _load_target(spec: CommandSpec) -> CommandTarget:
    """按 ``模块:属性`` 导入命令目标；导入失败即装配失败。"""
    module_name, _, attribute = spec.target.partition(":")
    return cast(CommandTarget, getattr(import_module(module_name), attribute))


def apply(ctx: Context, config: dict) -> None:
    """登记全部命令与能力键；每一项都经 ``ctx.effect`` 可撤销。"""
    registry = config.get("registry")
    if not isinstance(registry, CommandRegistry):
        raise TypeError("xs.commands 需要 config['registry'] 为 CommandRegistry")

    for spec in COMMAND_SPECS:
        target = _load_target(spec)
        entry = CommandEntry(name=spec.name, target=target, kind=spec.kind)
        ctx.effect(partial(registry.register, entry), label=f"command:{spec.name}")
        ctx.effect(partial(ctx.provide, spec.key, target), label=f"provide:{spec.key}")


PLUGIN = Plugin(
    name=MANIFEST.id,
    apply=apply,
    provide=[spec.key for spec in COMMAND_SPECS],
)
