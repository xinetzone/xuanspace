"""按解序装配与事务回滚（P1-2）。

对应设计文档 §5.4 的第 6–9 步与 P2「可撤销装配」的 4–5 条：逐节点装配，
任一步失败即回滚到装配前状态，并以**静默断言**收尾——回滚后效应数必须为 0、
已装配的提供键必须全部从 store 清除，避免「热替换不留半成品」只停留在承诺。

装配期的撤销能力沿用内核既有语义：插件经 ``Context.effect``/``Context.provide``
登记的效应由 ``Context.dispose`` 统一回收，故失败回滚以整批 ``dispose``
实现——本模块创建并独占该上下文，不会波及调用方的既有状态。
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .context import Context
from .plugin import FiberState, Plugin
from .solver import ReconcileReport, Solution, reconcile


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """一次装配的结果快照。"""

    applied: tuple[str, ...]
    """按解序成功生效的节点（失败时停在首个失败节点之前）。"""

    rolled_back: tuple[str, ...]
    """被逆序回滚的节点；成功时为空。"""

    errors: tuple[tuple[str, Exception], ...]
    """装配期与回滚期的失败清单，标签形如 ``apply:<id>`` / ``root:<label>``。"""

    reconciled: ReconcileReport | None
    """成功路径上的声明/实际对账报告；回滚路径为 None。"""

    residual_effects: int
    residual_keys: tuple[str, ...]
    """回滚后残留的效应数与已注册键；成功路径为装配后的存量，仅供观测。"""

    silenced: bool
    """静默断言：失败路径要求零残留；成功路径要求所有节点均已 ACTIVE。"""

    @property
    def ok(self) -> bool:
        """无任何失败即视为装配成功。"""
        return not self.errors


def assemble(
    solution: Solution,
    plugins: Mapping[str, Plugin],
    *,
    configs: Mapping[str, dict] | None = None,
) -> AssemblyResult:
    """按 ``solution.order`` 装配 ``plugins``，失败时逆序回滚并做静默断言。

    ``plugins`` 以清单 id 为键；清单声明与实现缺一即视为调用方错误，直接抛
    ``KeyError``——这属于编程错误，不应与「插件 apply 失败」混为一类。
    """
    absent = [node for node in solution.order if node not in plugins]
    if absent:
        raise KeyError(f"缺少插件实现：{', '.join(absent)}")

    ctx = Context()
    applied: list[str] = []
    errors: list[tuple[str, Exception]] = []
    configs = configs or {}

    for node in solution.order:
        fiber = ctx.plugin(plugins[node], configs.get(node))
        if fiber.error is not None:
            errors.append((f"apply:{node}", fiber.error))
            break
        if fiber.state is not FiberState.ACTIVE:
            # 依赖键未真正注册时 Fiber 停留在 INACTIVE；若计为成功会得到
            # 「顺序正确但节点未生效」的假成功，故按失败处理并回滚。
            detail = f"节点未激活：state={fiber.state.name}"
            if fiber.reason:
                detail += f"，缺少注入 {', '.join(fiber.reason)}"
            errors.append((f"apply:{node}", RuntimeError(detail)))
            break
        applied.append(node)

    if errors:
        rolled_back = tuple(reversed(applied))
        ctx.dispose()
        # 卸载成功的 Fiber 已清空 error（P2-8），装配失败不会在处置阶段被
        # 二次上报；这里只补充真正的卸载失败，无需再按异常对象去重。
        errors.extend(ctx.get_disposal_errors())
        residual_effects = len(ctx.get_effects())
        residual_keys = ctx.registered_keys()
        if residual_effects or residual_keys:
            errors.append(
                (
                    "silence",
                    AssertionError(f"回滚后仍有残留：效应 {residual_effects} 个，键 {list(residual_keys)}"),
                )
            )
        return AssemblyResult(
            applied=tuple(applied),
            rolled_back=rolled_back,
            errors=tuple(errors),
            reconciled=None,
            residual_effects=residual_effects,
            residual_keys=residual_keys,
            silenced=not residual_effects and not residual_keys,
        )

    return AssemblyResult(
        applied=tuple(applied),
        rolled_back=(),
        errors=(),
        reconciled=reconcile(solution, ctx.registered_keys()),
        residual_effects=len(ctx.get_effects()),
        residual_keys=ctx.registered_keys(),
        silenced=True,
    )
