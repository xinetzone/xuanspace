"""进程内提供者：直接 import 目标模块并调用其属性。

``close`` 只丢弃本对象持有的模块引用，**不**承诺卸载已映射的 ``.so``
（ADR-03：CPython 不做 extension module 的 ``dlclose``）。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

from .api import FFIAPI, IN_PROCESS, FFIError, FFIInfo, FFIUnavailableError


class InProcessFFI(FFIAPI):
    """在同进程内直接调用目标模块的提供者。"""

    transport = IN_PROCESS

    def __init__(self, name: str, module: str) -> None:
        self.name = name
        self.module = module
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module)
            except ImportError as exc:
                raise FFIUnavailableError(f"无法导入 FFI 模块 {self.module!r}：{exc}") from exc
        return self._module

    def describe(self) -> FFIInfo:
        try:
            module = self._load()
        except FFIError as exc:
            return FFIInfo(self.name, self.module, self.transport, available=False, error=str(exc))

        version_getter = getattr(module, "version", None)
        available_getter = getattr(module, "is_available", None)
        try:
            version = str(version_getter()) if callable(version_getter) else str(getattr(module, "__version__", ""))
            available = bool(available_getter()) if callable(available_getter) else True
        except Exception as exc:  # noqa: BLE001 —— describe 的契约是不抛错
            return FFIInfo(self.name, self.module, self.transport, available=False, error=str(exc))
        return FFIInfo(self.name, self.module, self.transport, version=version, available=available)

    def invoke(self, method: str, *args: object, **kwargs: object) -> object:
        return resolve(self._load(), method)(*args, **kwargs)

    def close(self) -> None:
        self._module = None


def resolve(module: ModuleType, method: str) -> Callable[..., object]:
    """按点分路径解析模块属性，并把失败统一转换为 :class:`FFIError`。

    子进程 worker 复用同一解析语义，保证两种传输对「方法不存在」的判定一致。
    """
    if not method:
        raise FFIError("方法名不能为空")
    target: object = module
    for part in method.split("."):
        if not part:
            raise FFIError(f"方法路径 {method!r} 含空段")
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise FFIError(f"模块 {module.__name__!r} 不存在属性 {method!r}") from exc
    if not callable(target):
        raise FFIError(f"属性 {method!r} 不可调用")
    return target
