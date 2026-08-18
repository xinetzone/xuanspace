"""可逆效应（Disposable）抽象 —— Cordis 启发。"""

from collections.abc import Callable
from dataclasses import dataclass, field

# 逆函数类型：调用即撤销一次上下文变换
Disposable = Callable[[], None]


@dataclass(frozen=True)
class EffectMeta:
    """效应元信息，用于调试与可观测性。"""
    label: str
    children: list[EffectMeta] = field(default_factory=list)


class DisposableList:
    """逆序回收的效应列表。

    后注册的效应先回收（栈式语义），对应 Cordis 逆序累积规则：
    (f1,g1)∘(f2,g2) = (f1∘f2, g2∘g1)
    """

    def __init__(self) -> None:
        self._disposables: list[Disposable] = []

    def push(self, dispose: Disposable) -> Disposable:
        """注册一个逆函数，返回可撤销此注册的函数。"""
        self._disposables.append(dispose)
        removed = False

        def undo() -> None:
            nonlocal removed
            if not removed and dispose in self._disposables:
                self._disposables.remove(dispose)
                removed = True

        return undo

    def clear(self) -> list[Disposable]:
        """逆序返回所有 disposable（后注册先返回），清空列表。"""
        result = list(reversed(self._disposables))
        self._disposables.clear()
        return result

    def __len__(self) -> int:
        return len(self._disposables)

    def __bool__(self) -> bool:
        return bool(self._disposables)
