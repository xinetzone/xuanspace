"""Tests for xs doctor Sphinx documentation directory detection."""

from pathlib import Path

from xs.commands.doctor_cmd import check_sphinx


def test_check_sphinx_detects_doc_directory(tmp_path: Path):
    # 仓库约定：Sphinx 源目录为 doc/（根目录、libs/mystx、libs/tvm-book 均如此）
    (tmp_path / "doc").mkdir()

    result = check_sphinx(tmp_path)

    assert result.status in ("ok", "warning")
    assert result.status != "skip"


def test_check_sphinx_falls_back_to_docs_directory(tmp_path: Path):
    # 兼容通用 docs/ 命名
    (tmp_path / "docs").mkdir()

    result = check_sphinx(tmp_path)

    assert result.status in ("ok", "warning")
    assert result.status != "skip"


def test_check_sphinx_skips_without_documentation_directory(tmp_path: Path):
    result = check_sphinx(tmp_path)

    assert result.status == "skip"
    assert result.required is False
