"""``xuan-ffi``：FFI 能力键与子进程隔离模式（P2-3 试点 C）。

能力键形如 ``xuan:ffi/caffe#api``；消费者只注入键并调用
:meth:`~xuan_ffi.api.FFIAPI.invoke`，不感知具体发行包与传输模式，
因此「换后端」与「换传输」都不改调用方代码。

原生插件声明 ``lifecycle="static"``：``close()`` 只做逻辑退役（撤销能力
句柄、结束子进程），不承诺卸载已映射的 ``.so``（ADR-03）。
"""

from xuan_plugin.context import Context
from xuan_plugin.disposable import Disposable
from xuan_plugin.manifest import PluginManifest, ProvideDecl
from xuan_plugin.plugin import Plugin

from .api import FFIAPI, IN_PROCESS, SUBPROCESS, FFIError, FFIInfo, FFIUnavailableError
from .inprocess import InProcessFFI
from .keys import FFIKeyError, ffi_key, ffi_name
from .providers import BUILTIN_FFI, create_ffi
from .proxy import SubprocessFFI

PLUGIN_ID = "xuan.ffi"
_NAMES = tuple(sorted(BUILTIN_FFI))

MANIFEST = PluginManifest(
    id=PLUGIN_ID,
    version="0.1.0",
    provide=tuple(ProvideDecl(key=ffi_key(name)) for name in _NAMES),
    reversible=True,
    priority=40,
    lifecycle="static",
    source="entry-point:xuan_ffi",
)


def apply(ctx: Context, config: dict) -> None:
    """按配置为每个内置 FFI 登记能力键；撤销时关闭提供者并移除键。

    配置项：``mode``（``in-process`` / ``subprocess``）、``modules``
    （名称 → 模块覆盖）、``options``（透传给子进程提供者）。
    """
    settings = config or {}
    mode = settings.get("mode", IN_PROCESS)
    modules = settings.get("modules", {})
    options = settings.get("options", {})
    for name in _NAMES:
        provider = create_ffi(name, module=modules.get(name), mode=mode, **options)
        _provide(ctx, name, provider)


def _provide(ctx: Context, name: str, provider: FFIAPI) -> None:
    """登记一条能力键，撤销时先关闭提供者再移除键。"""

    def execute() -> list[Disposable]:
        return [ctx.provide(ffi_key(name), provider), provider.close]

    ctx.effect(execute, label=f"provide:{ffi_key(name)}")


PLUGIN = Plugin(
    name=PLUGIN_ID,
    apply=apply,
    provide=[ffi_key(name) for name in _NAMES],
)

__all__ = [
    "BUILTIN_FFI",
    "FFIAPI",
    "FFIError",
    "FFIInfo",
    "FFIKeyError",
    "FFIUnavailableError",
    "IN_PROCESS",
    "MANIFEST",
    "PLUGIN",
    "PLUGIN_ID",
    "SUBPROCESS",
    "InProcessFFI",
    "SubprocessFFI",
    "create_ffi",
    "ffi_key",
    "ffi_name",
]
