"""``xuan-build``：构建后端分派能力层（P2-1）。

能力键 ``xuan:build/backend#dispatch`` 的提供者。消费者只需拿到
:class:`~xuan_build.dispatch.BuildDispatcher`，不再按项目类型 if/elif。
"""

from xuan_plugin.context import Context
from xuan_plugin.manifest import PluginManifest, ProvideDecl
from xuan_plugin.plugin import Plugin

from .backends import BuildBackend, NativeBackend, OtherBackend, PythonBackend, Reporter, StaticBackend
from .dispatch import DEFAULT_DISPATCHER, BuildDispatcher, dispatch

DISPATCH_KEY = "xuan:build/backend#dispatch"

MANIFEST = PluginManifest(
    id="xuan.build",
    version="0.1.0",
    provide=(ProvideDecl(key=DISPATCH_KEY),),
    reversible=True,
    priority=50,
    lifecycle="reloadable",
    source="entry-point:xuan_build",
)


def apply(ctx: Context, config: dict) -> None:
    """登记分派器；撤销时移除能力键。"""
    ctx.effect(lambda: ctx.provide(DISPATCH_KEY, DEFAULT_DISPATCHER), label=f"provide:{DISPATCH_KEY}")


PLUGIN = Plugin(
    name=MANIFEST.id,
    apply=apply,
    provide=[DISPATCH_KEY],
)

__all__ = [
    "DEFAULT_DISPATCHER",
    "DISPATCH_KEY",
    "MANIFEST",
    "PLUGIN",
    "BuildBackend",
    "BuildDispatcher",
    "NativeBackend",
    "OtherBackend",
    "PythonBackend",
    "Reporter",
    "StaticBackend",
    "dispatch",
]
