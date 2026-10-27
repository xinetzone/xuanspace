"""Tests for xs meta 写入路径 — tomlkit 原地编辑、键名归一化与命令集成。"""

import tomllib
from datetime import datetime
from pathlib import Path

from xs.commands.meta_cmd import (
    _build_frontmatter,
    _get_toml_path,
    _read_toml_meta,
    _write_toml_meta,
    meta_sync,
    meta_validate,
)

# 覆盖注释、行内注释、空行与表顺序的既有文件
PREEXISTING = """# 顶部注释

title = "旧标题"  # 行内注释
category = "docs"

[meta]
owner = "team"
"""


def _load(toml_path: Path) -> dict:
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def _make_workspace(root: Path) -> Path:
    """构造最小工作区（pyproject.toml + apps/ + libs/），返回已解析的根路径。"""
    root = root.resolve()
    (root / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    (root / "apps").mkdir()
    (root / "libs").mkdir()
    return root


# --- _write_toml_meta 直接覆盖 ---


def test_write_creates_file_and_parent_dirs(tmp_path):
    """新建文件：自动创建父目录，并补默认 date。"""
    toml_path = tmp_path / ".meta" / "toml" / "doc" / "guide" / "index.toml"

    _write_toml_meta(toml_path, {"title": "指南", "category": "docs"})

    assert toml_path.is_file()
    data = _load(toml_path)
    assert data["title"] == "指南"
    assert data["category"] == "docs"
    assert data["date"] == datetime.now().strftime("%Y-%m-%d")


def test_write_preserves_comments_blank_lines_and_key_order(tmp_path):
    """原地编辑：未被触及的注释、空行、键序与子表原样保留。"""
    toml_path = tmp_path / "doc.toml"
    toml_path.write_text(PREEXISTING, encoding="utf-8")

    _write_toml_meta(toml_path, {"title": "新标题"})

    text = toml_path.read_text(encoding="utf-8")
    assert "# 顶部注释" in text
    assert "# 行内注释" in text
    assert "\n\n[meta]" in text
    assert text.index("title") < text.index("category") < text.index("[meta]")
    data = _load(toml_path)
    assert data["title"] == "新标题"
    assert data["meta"]["owner"] == "team"


def test_write_normalizes_hyphenated_keys_and_read_restores_them(tmp_path):
    """键名归一化：写入时 - 转 _，读回时 _ 还原为 -。"""
    toml_path = tmp_path / "doc.toml"

    _write_toml_meta(toml_path, {"x-toml-ref": ".meta/toml/doc.toml"})

    assert _load(toml_path)["x_toml_ref"] == ".meta/toml/doc.toml"
    assert _read_toml_meta(toml_path)["x-toml-ref"] == ".meta/toml/doc.toml"


def test_write_normalizes_tags_string_to_list(tmp_path):
    """tags 归一化：字符串形式转为列表，列表原样透传。"""
    toml_path = tmp_path / "doc.toml"

    _write_toml_meta(toml_path, {"tags": "[a, b, c]"})
    assert _load(toml_path)["tags"] == ["a", "b", "c"]

    _write_toml_meta(toml_path, {"tags": ["x", "y"]})
    assert _load(toml_path)["tags"] == ["x", "y"]


def test_write_keeps_existing_date(tmp_path):
    """已有 date 不被覆盖。"""
    toml_path = tmp_path / "doc.toml"
    toml_path.write_text('date = "2020-01-01"\n', encoding="utf-8")

    _write_toml_meta(toml_path, {"title": "t"})

    assert _load(toml_path)["date"] == "2020-01-01"


def test_write_falls_back_on_corrupt_file(tmp_path):
    """损坏文件：回退为新建文档，写入仍成功。"""
    toml_path = tmp_path / "doc.toml"
    toml_path.write_text("this is not = = toml", encoding="utf-8")

    _write_toml_meta(toml_path, {"title": "t"})

    data = _load(toml_path)
    assert data["title"] == "t"
    assert data["date"]


def test_write_is_idempotent(tmp_path):
    """幂等：重复写入同一批字段，产出文本不变。"""
    toml_path = tmp_path / "doc.toml"

    _write_toml_meta(toml_path, {"title": "t", "tags": "[a, b]"})
    first = toml_path.read_text(encoding="utf-8")

    _write_toml_meta(toml_path, {"title": "t", "tags": "[a, b]"})

    assert toml_path.read_text(encoding="utf-8") == first


# --- 路径映射与读取容错 ---


def test_get_toml_path_mirrors_doc_tree(tmp_path):
    """TOML 路径与 doc/ 目录结构镜像对应。"""
    doc = tmp_path / "doc" / "guide" / "index.md"

    assert _get_toml_path(tmp_path, doc) == tmp_path / ".meta" / "toml" / "doc" / "guide" / "index.toml"


def test_read_missing_or_corrupt_returns_empty(tmp_path):
    """读取缺失或损坏的 TOML 文件返回空字典，不抛异常。"""
    assert _read_toml_meta(tmp_path / "none.toml") == {}

    bad = tmp_path / "bad.toml"
    bad.write_text("this is not = = toml", encoding="utf-8")
    assert _read_toml_meta(bad) == {}


def test_build_frontmatter_renders_list_bool_and_quoted_values():
    """frontmatter 重建：列表、布尔与需转义值按固定键序输出。"""
    text = _build_frontmatter({"id": "a b", "source": ["x.md"], "version": True})

    assert text == '---\nid: "a b"\nsource: [x.md]\nversion: true\n---\n'
    assert _build_frontmatter({"id": "a-b"}) == '---\nid: "a-b"\n---\n'
    assert _build_frontmatter({}) == "---\n---\n"


# --- 命令集成（monkeypatch 工作区根）---


def test_meta_validate_fix_moves_forbidden_fields_to_toml(tmp_path, monkeypatch):
    """xs meta validate --fix：禁止字段迁入 TOML，正文写入 x-toml-ref。"""
    root = _make_workspace(tmp_path)
    doc = root / "doc" / "guide.md"
    doc.parent.mkdir()
    doc.write_text("---\nid: guide\ntitle: 指南\ntags: [a, b]\n---\n\n正文\n", encoding="utf-8")
    monkeypatch.setattr("xs.config.find_workspace_root", lambda *a, **k: root)

    meta_validate(doc, fix=True)

    text = doc.read_text(encoding="utf-8")
    frontmatter = text.split("---")[1]
    assert "title" not in frontmatter
    assert "tags" not in frontmatter
    assert "x-toml-ref: .meta/toml/doc/guide.toml" in frontmatter
    assert text.endswith("正文\n")

    data = _load(root / ".meta" / "toml" / "doc" / "guide.toml")
    assert data["title"] == "指南"
    assert data["tags"] == ["a", "b"]


def test_meta_sync_links_existing_toml(tmp_path, monkeypatch):
    """xs meta sync：已存在的 TOML 元数据被回填为 frontmatter 的 x-toml-ref。"""
    root = _make_workspace(tmp_path)
    doc = root / "doc" / "guide.md"
    doc.parent.mkdir()
    doc.write_text("---\nid: guide\n---\n\n正文\n", encoding="utf-8")

    _write_toml_meta(root / ".meta" / "toml" / "doc" / "guide.toml", {"title": "指南"})
    monkeypatch.setattr("xs.config.find_workspace_root", lambda *a, **k: root)

    meta_sync(doc)

    text = doc.read_text(encoding="utf-8")
    assert "x-toml-ref: .meta/toml/doc/guide.toml" in text
    assert text.endswith("正文\n")


def test_meta_validate_without_fix_does_not_write(tmp_path, monkeypatch):
    """xs meta validate（无 --fix）：只报告不落盘。

    必须显式传 fix=False：直接调用 typer 命令函数时，未传参的 Option 默认值是
    真值 OptionInfo 对象（仅 CLI 运行时才被解析为 False）。
    """
    root = _make_workspace(tmp_path)
    doc = root / "doc" / "guide.md"
    doc.parent.mkdir()
    original = "---\nid: guide\ntitle: 指南\n---\n\n正文\n"
    doc.write_text(original, encoding="utf-8")
    monkeypatch.setattr("xs.config.find_workspace_root", lambda *a, **k: root)

    meta_validate(doc, fix=False)

    assert doc.read_text(encoding="utf-8") == original
    assert not (root / ".meta").exists()


def test_meta_sync_moves_forbidden_fields_to_toml(tmp_path, monkeypatch):
    """xs meta sync：禁止字段迁入 TOML 并从 frontmatter 移除。"""
    root = _make_workspace(tmp_path)
    doc = root / "doc" / "guide.md"
    doc.parent.mkdir()
    doc.write_text("---\nid: guide\ntitle: 指南\n---\n\n正文\n", encoding="utf-8")
    monkeypatch.setattr("xs.config.find_workspace_root", lambda *a, **k: root)

    meta_sync(doc)

    text = doc.read_text(encoding="utf-8")
    assert "title" not in text.split("---")[1]
    assert "x-toml-ref: .meta/toml/doc/guide.toml" in text
    assert _load(root / ".meta" / "toml" / "doc" / "guide.toml")["title"] == "指南"
