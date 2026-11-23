"""装配求解器（P1-2）：拓扑定序、隔离域、可解性诊断与逐字段对账。

对应设计文档 §5.4 的第 3–5、7 步：

* ``solve`` 把候选清单求解为**确定性的装配计划**——同一清单集合无论输入
  顺序如何都得到逐项相同的顺序、隔离域与诊断，是「同声明必得同装配」的
  可复现性基础（H1/H2 的只读前置）；
* 缺失**必需**依赖的节点按 §5.4 第 3 步标为 INACTIVE 而非中断装配，
  且失活沿依赖闭包传播（提供者失活 → 消费者同样不可得）；
* 跨 id 环与同键冲突属于无解，抛 :class:`SolveError` 并给出可复现明细；
* ``reconcile`` 对比「声明的 provide」与「实际注册的键」，缺/多分别列出。

本模块只做纯计算，不触碰 ``Context``——装配与回滚由 ``assembly`` 负责。
"""

import enum
import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .manifest import PluginManifest, find_conflicts


class SolveErrorKind(enum.Enum):
    """求解失败的类别。取值参与输出，改动即兼容性变更。"""

    CYCLE = "cycle"
    MISSING = "missing"
    CONFLICT = "conflict"


class SolveError(ValueError):
    """装配计划无解。

    ``details`` 为逐条可复现明细（如 ``xuan:a/b#c: p1, p2``），并按字典序
    排序后输出——相同输入必得逐字节相同的错误文本，便于快照比对。
    """

    def __init__(self, kind: SolveErrorKind, message: str, *, details: Iterable[str] = ()) -> None:
        self.kind = kind
        self.details = tuple(sorted(details))
        body = message if not self.details else f"{message}：{'；'.join(self.details)}"
        super().__init__(f"[{kind.value}] {body}")


@dataclass(frozen=True, slots=True)
class InactiveReason:
    """节点失活的原因：某些必需能力键在候选集合内不可得。"""

    plugin: str
    missing: tuple[str, ...]

    def describe(self) -> str:
        return f"{self.plugin} 缺少必需依赖 {', '.join(self.missing)}"


@dataclass(frozen=True, slots=True)
class Domain:
    """一个隔离域：共享 ``isolate`` 标签的连通簇（无标签者归默认域）。"""

    labels: tuple[str, ...]
    nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Solution:
    """确定性装配计划。"""

    order: tuple[str, ...]
    """装配顺序：提供者先于消费者；同层按 ``(-priority, id)`` 定序。"""

    domains: tuple[Domain, ...]
    """隔离域（按标签、再按节点定序）；默认域 ``labels == ()`` 排在最前。"""

    inactive: tuple[InactiveReason, ...]
    """按 §5.4 第 3 步标为 INACTIVE 的节点及其缺失键。"""

    provides: tuple[tuple[str, str], ...]
    """选定提供者：``(key, provider_id)``，仅含实际参与装配的键，按 key 排序。"""


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """声明视图与实际注册视图的逐字段对账结果。"""

    missing: tuple[tuple[str, str], ...]
    """声明了但未注册：``(provider_id, key)``，按 id、key 排序。"""

    unexpected: tuple[str, ...]
    """注册了但未声明：按 key 排序。"""

    matched: tuple[str, ...]
    """两侧一致：按 key 排序。"""

    @property
    def ok(self) -> bool:
        """两侧完全一致时为真。"""
        return not self.missing and not self.unexpected


def _provider_index(active: tuple[PluginManifest, ...]) -> dict[str, tuple[tuple[str, str, int], ...]]:
    """key → [(plugin_id, mode, priority)]，按 ``(-priority, id)`` 定序（ADR-05）。"""
    index: dict[str, list[tuple[str, str, int]]] = {}
    for manifest in active:
        for decl in manifest.provide:
            index.setdefault(decl.key, []).append((manifest.id, decl.mode, manifest.priority))
    for providers in index.values():
        providers.sort(key=lambda item: (-item[2], item[0]))
    return {key: tuple(providers) for key, providers in index.items()}


def _inactive_nodes(active: tuple[PluginManifest, ...], owner: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """计算 INACTIVE 节点：必需键无提供者，或提供者自身已失活（沿闭包传播）。

    自提供键（插件 inject 自己 provide 的键）按 §5.4 第 4 步的 peer 语义视为
    可满足，不构成失活，也不构成环。
    """
    inactive: dict[str, tuple[str, ...]] = {}
    changed = True
    while changed:
        changed = False
        for manifest in active:
            if manifest.id in inactive:
                continue
            missing = sorted(
                decl.key
                for decl in manifest.inject
                if not decl.optional
                and (decl.key not in owner or (owner[decl.key] != manifest.id and owner[decl.key] in inactive))
            )
            if missing:
                inactive[manifest.id] = tuple(missing)
                changed = True
    return inactive


def _find_cycle(remaining: tuple[str, ...], deps: Mapping[str, set[str]]) -> tuple[str, ...]:
    """在无法定序的残余节点中找出首个环，格式 ``a -> b -> a``（边＝依赖方向）。

    遍历起点与邻接均按字典序，保证同一输入必得同一环。
    """
    visited: set[str] = set()

    def walk(node: str, stack: list[str]) -> tuple[str, ...]:
        visited.add(node)
        stack.append(node)
        for nxt in sorted(deps[node]):
            if nxt in stack:
                return tuple(stack[stack.index(nxt) :] + [nxt])
            if nxt not in visited:
                found = walk(nxt, stack)
                if found:
                    return found
        stack.pop()
        return ()

    for node in remaining:
        if node not in visited:
            found = walk(node, [])
            if found:
                return found
    return ()


def _order(
    solvable: tuple[str, ...],
    deps: Mapping[str, set[str]],
    priorities: Mapping[str, int],
) -> tuple[str, ...]:
    """Kahn 拓扑排序；就绪集按 ``(-priority, id)`` 定序，环残留抛 ``SolveError``。"""
    remaining = {node: set(values) for node, values in deps.items()}
    dependents: dict[str, set[str]] = {node: set() for node in solvable}
    for node, providers in remaining.items():
        for provider in providers:
            dependents[provider].add(node)

    ready = [(-priorities[node], node) for node in solvable if not remaining[node]]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for consumer in sorted(dependents[node]):
            remaining[consumer].discard(node)
            if not remaining[consumer]:
                heapq.heappush(ready, (-priorities[consumer], consumer))

    if len(order) < len(solvable):
        unresolved = tuple(sorted(node for node in solvable if node not in set(order)))
        cycle = _find_cycle(unresolved, remaining)
        raise SolveError(
            SolveErrorKind.CYCLE,
            "跨插件循环依赖无法定序",
            details=[" -> ".join(cycle) if cycle else ", ".join(unresolved)],
        )
    return tuple(order)


def _domains(solvable: tuple[str, ...], manifests: Mapping[str, PluginManifest]) -> tuple[Domain, ...]:
    """隔离域：共享任一 ``isolate`` 标签的节点归入同一连通簇。

    未声明 ``isolate`` 的节点共享根上下文，合并为一个默认域（``labels == ()``）。
    """
    parent = {node: node for node in solvable}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # 固定以字典序较小者为根，结果与输入顺序无关
            parent[max(root_left, root_right)] = min(root_left, root_right)

    label_owner: dict[str, str] = {}
    unlabeled: list[str] = []
    for node in solvable:
        labels = manifests[node].isolate
        if not labels:
            unlabeled.append(node)
            continue
        for label in labels:
            if label in label_owner:
                union(node, label_owner[label])
            else:
                label_owner[label] = node

    groups: dict[str, list[str]] = {}
    for node in solvable:
        if manifests[node].isolate:
            groups.setdefault(find(node), []).append(node)

    domains = [
        Domain(
            labels=tuple(sorted({label for node in nodes for label in manifests[node].isolate})),
            nodes=tuple(sorted(nodes)),
        )
        for nodes in groups.values()
    ]
    if unlabeled:
        domains.append(Domain(labels=(), nodes=tuple(sorted(unlabeled))))
    domains.sort(key=lambda domain: (domain.labels, domain.nodes))
    return tuple(domains)


def solve(manifests: Iterable[PluginManifest], *, allow_inactive: bool = True) -> Solution:
    """把候选清单求解为确定性装配计划。

    ``allow_inactive=False`` 时，缺失必需依赖不再是 INACTIVE 而是
    ``SolveError(MISSING)``——供需要「全量可激活」的调用方（如发布前检查）使用。
    """
    active = tuple(sorted((m for m in manifests if not m.disabled), key=lambda m: m.id))
    index = _provider_index(active)

    conflicts = find_conflicts(index)
    if conflicts:
        raise SolveError(
            SolveErrorKind.CONFLICT,
            "能力键存在多个无法判定胜出的提供者",
            details=[f"{key}: {', '.join(ids)}" for key, ids in conflicts],
        )

    owner = {key: providers[0][0] for key, providers in index.items()}
    inactive = _inactive_nodes(active, owner)
    if inactive and not allow_inactive:
        raise SolveError(
            SolveErrorKind.MISSING,
            "必需依赖在候选集合内不可满足",
            details=[f"{plugin_id} 缺少 {', '.join(keys)}" for plugin_id, keys in sorted(inactive.items())],
        )

    solvable = tuple(manifest.id for manifest in active if manifest.id not in inactive)
    by_id = {manifest.id: manifest for manifest in active}
    priorities = {manifest.id: manifest.priority for manifest in active}

    deps: dict[str, set[str]] = {node: set() for node in solvable}
    for node in solvable:
        for decl in by_id[node].inject:
            provider = owner.get(decl.key)
            if provider is None or provider == node or provider not in deps:
                continue
            deps[node].add(provider)

    order = _order(solvable, deps, priorities)
    order_set = set(order)
    return Solution(
        order=order,
        domains=_domains(solvable, by_id),
        inactive=tuple(InactiveReason(plugin=node, missing=keys) for node, keys in sorted(inactive.items())),
        provides=tuple((key, provider) for key, provider in sorted(owner.items()) if provider in order_set),
    )


def reconcile(solution: Solution, registered: Iterable[str]) -> ReconcileReport:
    """把声明的 provide 与实际注册的键逐项对账（H4 的运行期形态）。"""
    expected = dict(solution.provides)
    actual = set(registered)
    missing = tuple(sorted((owner, key) for key, owner in expected.items() if key not in actual))
    unexpected = tuple(sorted(actual - set(expected)))
    matched = tuple(sorted(set(expected) & actual))
    return ReconcileReport(missing=missing, unexpected=unexpected, matched=matched)
