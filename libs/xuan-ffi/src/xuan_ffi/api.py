"""FFI 能力契约：一条 ``invoke`` 面同时适配进程内与子进程后端。

传输模式的差异被限制在提供者内部：调用方只拿到 :class:`FFIAPI`，
因此「换后端」与「换传输」都不需要改调用方代码（P2-3 验收）。

``invoke`` 的参数与返回值须可 JSON 序列化——这是子进程模式能成立的前提；
对象图密集的原生 API（如 ``Blob``/``Net``）不在本契约的覆盖范围内，
仍走进程内直连。
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

IN_PROCESS = "in-process"
SUBPROCESS = "subprocess"


class FFIError(RuntimeError):
    """FFI 调用失败：模块不可导入、方法不存在、子进程退出或结果不可序列化。"""


class FFIUnavailableError(FFIError):
    """目标模块不可导入（原生扩展缺失、依赖未见等）。"""


@dataclass(frozen=True, slots=True)
class FFIInfo:
    """提供者的只读快照，用于诊断与 ``xs`` 输出。"""

    name: str
    module: str
    transport: str
    version: str = ""
    available: bool = False
    error: str = ""


@runtime_checkable
class FFIAPI(Protocol):
    """能力面：``xuan:ffi/<name>#api`` 的值必须满足本协议。"""

    name: str
    transport: str

    def describe(self) -> FFIInfo:
        """返回版本与可用性；不可用时以 ``available=False`` 表达而非抛错。"""
        ...

    def invoke(self, method: str, *args: object, **kwargs: object) -> object:
        """调用目标模块的方法（``method`` 为点分属性路径）。"""
        ...

    def close(self) -> None:
        """逻辑退役：释放引用/结束子进程；不承诺卸载已映射的原生代码。"""
        ...
