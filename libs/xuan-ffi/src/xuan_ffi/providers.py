"""内置 FFI 名称与提供者工厂。"""

from __future__ import annotations

from .api import FFIAPI, IN_PROCESS, SUBPROCESS
from .inprocess import InProcessFFI
from .keys import ffi_key
from .proxy import SubprocessFFI

#: 工作区内随包发行的 FFI 后端：名称 → 模块。名称即能力键的 ``<name>`` 段。
BUILTIN_FFI: dict[str, str] = {
    "caffe": "caffe_ffi",
    "demo": "demo_ffi",
    "npu": "npu_ffi",
}


def create_ffi(name: str, *, module: str | None = None, mode: str = IN_PROCESS, **options: object) -> FFIAPI:
    """按名称与传输模式创建提供者。

    ``options`` 只透传给子进程模式（``python`` / ``extra_paths`` / ``env``）；
    进程内模式会忽略它，因此同一份调用方配置可在两种模式间直接切换。
    """
    ffi_key(name)  # 提前校验名称，避免非法键进入装配
    module_name = module or BUILTIN_FFI.get(name) or f"{name}_ffi"
    if mode == IN_PROCESS:
        return InProcessFFI(name, module_name)
    if mode == SUBPROCESS:
        return SubprocessFFI(name, module_name, **options)  # type: ignore[arg-type]
    raise ValueError(f"未知传输模式 {mode!r}，期望 {IN_PROCESS!r} 或 {SUBPROCESS!r}")
