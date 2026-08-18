"""``okf.frontmatter`` 解析器（零第三方依赖）的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

import okf.frontmatter as frontmatter
from okf.frontmatter import FrontmatterError, parse_concept, parse_frontmatter


class TestParseFrontmatter:
    def test_separates_frontmatter_and_body(self) -> None:
        text = "---\ntype: Metric\n---\n# Body\n"
        data, body = parse_frontmatter(text)
        assert data == {"type": "Metric"}
        assert body == "# Body\n"

    def test_scalars(self) -> None:
        text = (
            "---\n"
            "title: Hello\n"
            "description: World\n"
            "count: 42\n"
            "active: true\n"
            "inactive: false\n"
            "---\n"
        )
        data, _ = parse_frontmatter(text)
        assert data["title"] == "Hello"
        assert data["description"] == "World"
        assert data["count"] == 42
        assert data["active"] is True
        assert data["inactive"] is False

    def test_flow_list(self) -> None:
        text = "---\ntags: [a, b]\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["tags"] == ["a", "b"]

    def test_flow_map(self) -> None:
        text = "---\ngenerated: {by: pipeline, at: 2024-01-15}\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["generated"] == {"by": "pipeline", "at": "2024-01-15"}

    def test_nested_block_map(self) -> None:
        text = (
            "---\n"
            "type: Metric\n"
            "generated:\n"
            "  by: pipeline\n"
            "  at: 2024-01-15\n"
            "---\n"
        )
        data, _ = parse_frontmatter(text)
        assert data["generated"]["by"] == "pipeline"
        assert data["generated"]["at"] == "2024-01-15"

    def test_no_frontmatter(self) -> None:
        text = "# Just a doc\n\nSome body.\n"
        data, body = parse_frontmatter(text)
        assert data == {}
        assert body == text

    def test_import_does_not_depend_on_pyyaml(self) -> None:
        source = Path(frontmatter.__file__).read_text(encoding="utf-8")
        assert "import yaml" not in source
        assert "from yaml" not in source


class TestParseConcept:
    @staticmethod
    def _write_file(tmp_path: Path, content: str) -> Path:
        path = tmp_path / "concept.md"
        path.write_text(content, encoding="utf-8")
        return path

    def test_builds_concept(self, tmp_path: Path) -> None:
        path = self._write_file(
            tmp_path, "---\ntype: Metric\ntitle: Revenue\n---\n# Body\n"
        )
        concept = parse_concept(path)
        assert concept.type == "Metric"
        assert concept.title == "Revenue"
        assert concept.path == path
        assert concept.body == "# Body\n"

    def test_missing_type_raises(self, tmp_path: Path) -> None:
        path = self._write_file(tmp_path, "---\ntitle: No type\n---\nbody\n")
        with pytest.raises(FrontmatterError):
            parse_concept(path)

    def test_empty_type_raises(self, tmp_path: Path) -> None:
        path = self._write_file(tmp_path, '---\ntype: ""\n---\nbody\n')
        with pytest.raises(FrontmatterError):
            parse_concept(path)

    def test_unknown_keys_preserved_in_extra(self, tmp_path: Path) -> None:
        path = self._write_file(
            tmp_path, "---\ntype: Metric\ncustom_field: hello\nanother: true\n---\nbody\n"
        )
        concept = parse_concept(path)
        assert concept.extra == {"custom_field": "hello", "another": True}
