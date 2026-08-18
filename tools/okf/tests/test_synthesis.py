"""OKF 索引与日志合成器（§8/§9）测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from okf.models import Concept
from okf.synthesis import (
    generate_index,
    generate_log,
    parse_log,
    write_index,
    write_log,
)


def _concept(path: Path, type_: str, title: str = "", description: str = "") -> Concept:
    return Concept(path=path, type=type_, title=title, description=description)


# ── generate_index ────────────────────────────────────────────────────────


def test_generate_index_groups_by_type(tmp_path):
    directory = tmp_path / "mybundle"
    directory.mkdir()
    concepts = [
        _concept(directory / "a.md", "Metric", "A", "desc A"),
        _concept(directory / "b.md", "Metric", "B"),
        _concept(directory / "c.md", "Table", "C", "desc C"),
        _concept(directory / "d.md", "", "D"),
    ]
    content = generate_index(directory, concepts)
    # 无 frontmatter
    assert not content.startswith("---")
    # 标题为目录名
    assert "# mybundle" in content
    # 按 type 分组
    assert "## Metric" in content
    assert "## Table" in content
    assert "## _" in content
    # 条目格式
    assert "- [A](a.md) — desc A" in content
    assert "- [B](b.md)" in content
    assert "- [C](c.md) — desc C" in content
    # 分组顺序（排序后 Metric 在 Table 前）
    assert content.index("## Metric") < content.index("## Table")


# ── parse_log ─────────────────────────────────────────────────────────────


def test_parse_log(tmp_path):
    log_path = tmp_path / "log.md"
    log_path.write_text(
        "# Change Log\n\n"
        "## 2024-02-01\n\n"
        "- **Update** Updated revenue\n"
        "- **Creation** Added customer table\n\n"
        "## 2024-01-15\n\n"
        "- **Creation** Initial bundle setup\n",
        encoding="utf-8",
    )
    result = parse_log(log_path)
    assert result[date(2024, 2, 1)] == [
        "**Update** Updated revenue",
        "**Creation** Added customer table",
    ]
    assert result[date(2024, 1, 15)] == ["**Creation** Initial bundle setup"]


def test_parse_log_missing_file(tmp_path):
    assert parse_log(tmp_path / "missing.md") == {}


# ── generate_log ──────────────────────────────────────────────────────────


def test_generate_log_descending():
    entries = {
        date(2024, 1, 15): ["**Creation** Initial bundle setup"],
        date(2024, 2, 1): ["**Update** Updated revenue"],
    }
    content = generate_log(entries)
    assert content.startswith("# Change Log\n")
    # 日期倒序：最新在前
    assert content.index("## 2024-02-01") < content.index("## 2024-01-15")
    assert "- **Update** Updated revenue" in content
    assert "- **Creation** Initial bundle setup" in content


# ── write_index / write_log ───────────────────────────────────────────────


def test_write_index(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    concepts = [_concept(directory / "x.md", "Metric", "X", "desc")]
    written = write_index(directory, concepts)
    assert written == directory / "index.md"
    assert written.exists()
    assert "- [X](x.md) — desc" in written.read_text(encoding="utf-8")


def test_write_log(tmp_path):
    target = tmp_path / "log.md"
    entries = {date(2024, 2, 1): ["**Update** Updated revenue"]}
    written = write_log(target, entries)
    assert written == target
    assert written.exists()
    assert "## 2024-02-01" in written.read_text(encoding="utf-8")
