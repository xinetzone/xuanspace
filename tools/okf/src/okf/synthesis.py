"""OKF 索引与日志合成器：生成 index.md（§8）与 log.md（§9）。

零第三方依赖，仅使用 Python 标准库。
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from .models import Concept

__all__ = [
    "generate_index",
    "parse_log",
    "generate_log",
    "write_index",
    "write_log",
]

# ---------------------------------------------------------------
# §8  index.md 生成
# ---------------------------------------------------------------


def generate_index(directory: Path, concepts: list[Concept]) -> str:
    """按 OKF v0.2 §8 结构生成 index.md 内容。

    规则：
    - 无 YAML frontmatter
    - 标题 ``# Directory Name``（目录名）
    - 按概念的 ``type`` 字段分组（小节标题 ``## TypeName``）
    - 每个概念条目：``- [title](relative_path.md) — description``
    - 无 description 时省略 ``— description`` 部分
    - 无 type 的概念归入 ``_`` 分组
    """
    directory_name = directory.name or str(directory)
    lines: list[str] = [f"# {directory_name}", ""]

    grouped: dict[str, list[Concept]] = defaultdict(list)
    for concept in concepts:
        key = concept.type if concept.type else "_"
        grouped[key].append(concept)

    for type_name in sorted(grouped.keys()):
        lines.append(f"## {type_name}")
        lines.append("")
        for concept in grouped[type_name]:
            entry = _format_concept_entry(concept, directory)
            lines.append(entry)
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def _format_concept_entry(concept: Concept, base_dir: Path) -> str:
    """格式化单个概念条目。"""
    rel_path = _relative_path(concept.path, base_dir)
    title = concept.title if concept.title else concept.path.stem
    if concept.description:
        return f"- [{title}]({rel_path}) — {concept.description}"
    return f"- [{title}]({rel_path})"


def _relative_path(target: Path, base: Path) -> str:
    """计算 target 相对于 base 的路径，使用正斜杠分隔符。"""
    try:
        rel = target.relative_to(base)
    except ValueError:
        rel = target
    return rel.as_posix()


# ---------------------------------------------------------------
# §9  log.md 解析
# ---------------------------------------------------------------

# 日期标题格式：## YYYY-MM-DD
_DATE_HEADING_RE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s*$")


def parse_log(filepath: Path) -> dict[date, list[str]]:
    """解析 log.md（§9），返回 ``{date: [entries]}`` 字典。

    日期标题采用 ISO 8601 ``YYYY-MM-DD`` 格式（``## 2024-01-15``）。
    日期标题下的条目收集为列表。粗体动词（``**Update**`` 等）原样保留。
    """
    if not filepath.exists():
        return {}

    text = filepath.read_text(encoding="utf-8")
    result: dict[date, list[str]] = {}
    current_date: date | None = None
    current_entries: list[str] = []

    for line in text.splitlines():
        m = _DATE_HEADING_RE.match(line.strip())
        if m:
            if current_date is not None:
                result[current_date] = current_entries
            current_date = date.fromisoformat(m.group(1))
            current_entries = []
        elif current_date is not None:
            stripped = line.strip()
            if stripped and stripped.startswith(("-", "*")):
                # 去除列表标记前缀（"- " 或 "* "）
                entry = _strip_list_marker(stripped)
                if entry:
                    current_entries.append(entry)

    if current_date is not None:
        result[current_date] = current_entries

    return result


def _strip_list_marker(line: str) -> str:
    """去除无序列表标记前缀（``- `` 或 ``* ``）。"""
    if line.startswith("- "):
        return line[2:].strip()
    if line.startswith("* "):
        return line[2:].strip()
    return line


# ---------------------------------------------------------------
# §9  log.md 生成
# ---------------------------------------------------------------


def generate_log(entries: dict[date, list[str]]) -> str:
    """生成 log.md（§9），日期倒序。

    格式：
    - 标题 ``# Change Log``
    - 按日期倒序排列（最新在前）
    - 每个日期组：``## YYYY-MM-DD``，条目为无序列表
    """
    lines: list[str] = ["# Change Log", ""]
    sorted_dates = sorted(entries.keys(), reverse=True)

    for d in sorted_dates:
        lines.append(f"## {d.isoformat()}")
        lines.append("")
        for entry in entries[d]:
            lines.append(f"- {entry}")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------
# 便捷写入
# ---------------------------------------------------------------


def write_index(directory: Path, concepts: list[Concept]) -> Path:
    """生成 index.md 并写入 directory 目录，返回文件路径。"""
    content = generate_index(directory, concepts)
    target = directory / "index.md"
    target.write_text(content, encoding="utf-8")
    return target


def write_log(filepath: Path, entries: dict[date, list[str]]) -> Path:
    """生成 log.md 并写入 filepath，返回文件路径。"""
    content = generate_log(entries)
    filepath.write_text(content, encoding="utf-8")
    return filepath
