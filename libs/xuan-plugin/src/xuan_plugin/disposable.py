"""可逆效应（Disposable）抽象 —— Cordis 启发。"""

from collections.abc import Callable
from dataclasses import dataclass, field

# 逆函数类型：调用即撤销一次上下文变换
Disposable = Callable[[], None]


@dataclass(frozen=True, slots=True)
class EffectMeta:
    """效应元信息，用于调试与可观测性。"""

    label: str
    children: list[EffectMeta] = field(default_factory=list)


class DisposalError(Exception):
    """卸载期异常聚合。

    单个效应撤销失败不阻断后续撤销——先保证已登记效应全部被尝试回收，
    再把全部失败统一上报，避免「首个异常掩盖其余残留」。
    """

    def __init__(self, errors: list[tuple[str, Exception]]) -> None:
        self.errors = errors
        detail = "; ".join(f"{label}: {type(exc).__name__}: {exc}" for label, exc in errors)
        super().__init__(f"{len(errors)} 个效应撤销失败: {detail}")


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

    def dispose_all(self, report: Callable[[str, Exception], None] | None = None) -> list[tuple[str, Exception]]:
        """逆序执行全部 disposable，逐个捕获异常并返回失败清单。

        与 ``clear()`` 的差别：``clear()`` 只交出列表，逐项调用与异常处理
        由调用方负责；本方法自带异常隔离——任一撤销失败都不中断同批后续
        撤销，从而保证「效应数归零」优先于「首个异常立即抛出」。
        """
        errors: list[tuple[str, Exception]] = []
        for index, dispose in enumerate(self.clear()):
            try:
                dispose()
            except Exception as exc:  # noqa: BLE001 —— 必须兜住任意异常以继续回收
                label = getattr(dispose, "__name__", f"dispose#{index}")
                errors.append((label, exc))
                if report is not None:
                    report(label, exc)
        return errors

    def __len__(self) -> int:
        return len(self._disposables)

    def __bool__(self) -> bool:
        return bool(self._disposables)
