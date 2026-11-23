"""``xuan.plugins.lock``：插件契约的派生快照（设计 §5.6 / ADR-04，P3-1 落地）。

lock 是**产物**而非输入：只记录插件 id 与契约摘要，契约字段剔除 ``source``
（来源路径不参与比较，见 ``PluginManifest``），因此同一 commit 在任意机器上
重算结果逐字节一致。可解性、隔离性等语义判定仍由 ``manifest`` / ``solver``
承担；本模块只回答「锁视图能否由当前清单重算得到」（H4 视图一致性）。

规范形态::

    {"schema": 1, "plugins": [{"id": "a.b", "digest": "sha256:..."}]}

``plugins`` 按 ``id`` 排序；``build_lock`` 返回可比较视图，``render_lock``
返回落盘文本（2 空格缩进、UTF-8、行尾 LF），``lock_diff`` 给出逐 id 差异。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .manifest import PluginManifest

LOCK_SCHEMA = 1


def contract_payload(manifest: PluginManifest) -> dict[str, object]:
    """清单的契约字段视图（剔除 ``source`` 等诊断标记）。"""
    return {
        "id": manifest.id,
        "version": manifest.version,
        "provide": [[decl.key, decl.mode] for decl in manifest.provide],
        "inject": [[decl.key, decl.optional, decl.constraint] for decl in manifest.inject],
        "isolate": list(manifest.isolate),
        "reversible": manifest.reversible,
        "priority": manifest.priority,
        "disabled": manifest.disabled,
        "lifecycle": manifest.lifecycle,
    }


def plugin_digest(manifest: PluginManifest) -> str:
    """清单契约摘要：契约字段规范 JSON（键排序、紧凑分隔符）的 sha256。"""
    canonical = json.dumps(contract_payload(manifest), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_lock(manifests: Iterable[PluginManifest]) -> dict[str, object]:
    """由当前清单构造规范 lock 视图（按 id 排序）。"""
    return {
        "schema": LOCK_SCHEMA,
        "plugins": [
            {"id": manifest.id, "digest": plugin_digest(manifest)} for manifest in sorted(manifests, key=lambda m: m.id)
        ],
    }


def render_lock(manifests: Iterable[PluginManifest]) -> str:
    """渲染落盘文本：规范 lock 视图 + 行尾换行，保证逐字节可重现。"""
    return json.dumps(build_lock(manifests), ensure_ascii=False, indent=2) + "\n"


def lock_diff(expected: dict, actual: object) -> str:
    """逐 id 比对 lock 与重算视图，返回可读差异（供门禁失败信息直接展示）。"""
    if not isinstance(actual, dict):
        return f"期望 JSON 对象，实际为 {type(actual).__name__}"
    parts: list[str] = []
    if actual.get("schema") != expected["schema"]:
        parts.append(f"schema 期望 {expected['schema']}、实际 {actual.get('schema')!r}")
    actual_plugins = actual.get("plugins")
    if not isinstance(actual_plugins, list):
        parts.append("plugins 字段缺失或不是数组")
        return "；".join(parts)
    actual_digests: dict[str, object] = {}
    for item in actual_plugins:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            actual_digests[item["id"]] = item.get("digest")
    expected_digests = {item["id"]: item["digest"] for item in expected["plugins"]}
    for plugin_id in sorted(expected_digests):
        if plugin_id not in actual_digests:
            parts.append(f"缺少插件 {plugin_id}")
        elif actual_digests[plugin_id] != expected_digests[plugin_id]:
            parts.append(
                f"{plugin_id} digest 期望 {expected_digests[plugin_id][:19]}…、实际 {str(actual_digests[plugin_id])[:19]}…"
            )
    for plugin_id in sorted(set(actual_digests) - set(expected_digests)):
        parts.append(f"多出插件 {plugin_id}（当前声明中不存在）")
    if not parts:
        parts.append("plugins 顺序或字段不规范（期望按 id 排序、仅含 id 与 digest）")
    return "；".join(parts)
