"""``okf.frontmatter`` 补充测试：块级序列、嵌套、续行折叠与标量推断等边界分支。"""

from __future__ import annotations

from pathlib import Path

from okf.frontmatter import (
    _coerce_tags,
    _fold_lines,
    _parse_scalar,
    _split_top_level,
    _strip_comment,
    _unquote,
    _YamlParser,
    parse_concept,
    parse_frontmatter,
)

# ─── 标量推断 ──────────────────────────────────────────────────────────────


class TestParseScalar:
    def test_null_and_tilde(self) -> None:
        assert _parse_scalar("null") is None
        assert _parse_scalar("NULL") is None
        assert _parse_scalar("~") is None

    def test_empty_and_comment_only(self) -> None:
        assert _parse_scalar("") == ""
        assert _parse_scalar("   # comment") == ""

    def test_integers_signed(self) -> None:
        assert _parse_scalar("-42") == -42
        assert _parse_scalar("+7") == 7
        assert _parse_scalar("0") == 0

    def test_booleans(self) -> None:
        assert _parse_scalar("true") is True
        assert _parse_scalar("FALSE") is False

    def test_plain_string(self) -> None:
        assert _parse_scalar("hello world") == "hello world"


class TestUnquote:
    def test_double_quote_escapes(self) -> None:
        assert _unquote('"a\\"b"') == 'a"b'
        assert _unquote('"a\\nb"') == "a\nb"
        assert _unquote('"a\\\\b"') == "a\\b"

    def test_single_quote_doubling(self) -> None:
        assert _unquote("'a''b'") == "a'b"

    def test_plain_and_unterminated(self) -> None:
        assert _unquote("plain") == "plain"
        assert _unquote("'unterminated") == "'unterminated"


class TestStripComment:
    def test_inline_comment(self) -> None:
        assert _strip_comment("value # note") == "value"

    def test_quoted_hash_preserved(self) -> None:
        assert _strip_comment('"a # b"') == '"a # b"'

    def test_escaped_quote_inside_quote(self) -> None:
        assert _strip_comment(r'"a\" x"') == r'"a\" x"'


class TestSplitTopLevel:
    def test_nested_container_comma_ignored(self) -> None:
        assert _split_top_level("a, [b, c], d") == ["a", " [b, c]", " d"]

    def test_quoted_comma_ignored(self) -> None:
        assert _split_top_level('"a, b", c') == ['"a, b"', ' c']

    def test_nested_flow_map(self) -> None:
        assert _split_top_level("{k: v}, x") == ["{k: v}", " x"]


class TestFoldLines:
    def test_blank_line_becomes_newline(self) -> None:
        assert _fold_lines(["a", "", "b"]) == "a\nb"

    def test_join_with_spaces(self) -> None:
        assert _fold_lines(["a", "b"]) == "a b"


class TestCoerceTags:
    def test_comma_string(self) -> None:
        assert _coerce_tags("a, b, c") == ["a", "b", "c"]

    def test_list(self) -> None:
        assert _coerce_tags(["a", 2]) == ["a", "2"]

    def test_scalar_non_string(self) -> None:
        assert _coerce_tags(42) == ["42"]

    def test_none(self) -> None:
        assert _coerce_tags(None) == []


# ─── parse_frontmatter 边界 ───────────────────────────────────────────────


class TestParseFrontmatterEdge:
    def test_malformed_frontmatter_returns_original(self) -> None:
        text = "---\ntype: X\n"
        data, body = parse_frontmatter(text)
        assert data == {}
        assert body == text

    def test_block_sequence_inline_mapping(self) -> None:
        text = (
            "---\n"
            "type: Metric\n"
            "sources:\n"
            "  - resource: foo\n"
            "    author: Alice\n"
            "  - resource: bar\n"
            "---\n"
        )
        data, _ = parse_frontmatter(text)
        assert data["sources"] == [
            {"resource": "foo", "author": "Alice"},
            {"resource": "bar"},
        ]

    def test_block_sequence_scalar_items(self) -> None:
        text = "---\ntype: Metric\nitems:\n  - a\n  - b\n  - 3\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["items"] == ["a", "b", 3]

    def test_block_sequence_followed_by_sibling_key(self) -> None:
        text = "---\ntype: Metric\nitems:\n  - a\nafter: b\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["items"] == ["a"]
        assert data["after"] == "b"

    def test_block_sequence_flow_items(self) -> None:
        text = "---\ntype: Metric\nmatrix:\n  - [1, 2]\n  - {k: v}\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["matrix"] == [[1, 2], {"k": "v"}]

    def test_block_sequence_multiline_block(self) -> None:
        text = "---\ntype: Metric\nnote:\n  - >\n    line one\n    line two\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["note"] == ["line one line two"]

    def test_block_sequence_empty_item_with_mapping(self) -> None:
        text = "---\ntype: Metric\nitems:\n  -\n    a: b\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["items"] == [{"a": "b"}]

    def test_deep_nested_mapping(self) -> None:
        text = "---\ntype: Metric\na:\n  b:\n    c:\n      d: 1\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["a"]["b"]["c"]["d"] == 1

    def test_multiline_scalar_fold(self) -> None:
        text = "---\ntype: Metric\ndescription: First line\n  second line\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["description"] == "First line second line"

    def test_multiline_stops_at_next_key(self) -> None:
        text = "---\ntype: Metric\ndescription: base\nnext_key: value\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["description"] == "base"
        assert data["next_key"] == "value"

    def test_indented_multiline_value(self) -> None:
        text = "---\ntype: Metric\ndescription:\n  line one\n  line two\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["description"] == "line one line two"

    def test_indented_multiline_with_blank(self) -> None:
        text = "---\ntype: Metric\ndescription:\n  line one\n  \n  line two\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["description"] == "line one\nline two"

    def test_flow_map_quoted_key(self) -> None:
        text = '---\ntype: Metric\ngenerated: {"by": pipeline}\n---\n'
        data, _ = parse_frontmatter(text)
        assert data["generated"] == {"by": "pipeline"}

    def test_quoted_block_key(self) -> None:
        text = '---\n"quoted key": value\n---\n'
        data, _ = parse_frontmatter(text)
        assert data["quoted key"] == "value"

    def test_blank_and_comment_lines_skipped(self) -> None:
        text = "---\ntype: Metric\n# a comment\ntitle: X\n\nstatus: ok\n---\n"
        data, _ = parse_frontmatter(text)
        assert data == {"type": "Metric", "title": "X", "status": "ok"}

    def test_scalar_null_in_mapping(self) -> None:
        text = "---\ntype: Metric\na: null\nb: ~\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["a"] is None
        assert data["b"] is None


# ─── 解析器防御性分支 ──────────────────────────────────────────────────────


class TestYamlParserDefensive:
    def test_mapping_breaks_on_sequence(self) -> None:
        parser = _YamlParser("key: value\n- item\n")
        assert parser.parse() == {"key": "value"}

    def test_mapping_key_without_colon(self) -> None:
        parser = _YamlParser("justtext\n")
        assert parser.parse() == {}

    def test_unterminated_quoted_key(self) -> None:
        parser = _YamlParser('"unterminated\n')
        assert parser.parse() == {}


# ─── _split_top_level 反斜杠转义 ───────────────────────────────────────────


def test_split_top_level_backslash_escape_in_quote() -> None:
    # 引号内反斜杠跳过后继字符，逗号不视为分隔符
    assert _split_top_level(r'"a\,b"') == [r'"a\,b"']


# ─── _YamlParser 续行折叠 / 缩进值 / 多行字符串边界 ───────────────────────


class TestYamlParserFolding:
    def test_fold_continuation_breaks_on_list_item(self) -> None:
        parser = _YamlParser("key: value\n  - item\n")
        assert parser.parse() == {"key": "value"}

    def test_fold_continuation_breaks_on_colon(self) -> None:
        parser = _YamlParser("key: value\n  child: x\n")
        assert parser.parse() == {"key": "value"}

    def test_parse_flow_or_scalar_empty(self) -> None:
        assert _YamlParser("")._parse_flow_or_scalar("   ") == ""

    def test_parse_indented_value_eof(self) -> None:
        assert _YamlParser("k:\n").parse() == {"k": ""}

    def test_parse_indented_value_not_indented(self) -> None:
        assert _YamlParser("k:\nother: 1\n").parse() == {"k": "", "other": 1}

    def test_parse_multiline_string_stops_at_same_indent(self) -> None:
        assert _YamlParser("k:\n  a\nrest: x\n").parse() == {"k": "a", "rest": "x"}

    def test_parse_multiline_string_skips_comment(self) -> None:
        assert _YamlParser("k:\n  # comment\n  value\n").parse() == {"k": "value"}

    def test_parse_multiline_string_skips_mid_comment(self) -> None:
        # 注释行位于多行字符串中间，触发 _parse_multiline_string 内部的
        # stripped.startswith("#") 分支（skip 而不追加）
        assert _YamlParser("k:\n  first\n  # mid comment\n  second\n").parse() == {
            "k": "first second"
        }


# ─── _YamlParser 列表 / 内联映射 / 流程映射边界 ───────────────────────────


class TestYamlParserListMapping:
    def test_parse_list_breaks_on_non_item(self) -> None:
        assert _YamlParser("k:\n  - a\n  other\n").parse() == {"k": ["a"]}

    def test_continue_mapping_breaks_on_list_item(self) -> None:
        assert _YamlParser("sources:\n  - resource: foo\n    - bad\n").parse() == {
            "sources": [{"resource": "foo"}]
        }

    def test_continue_mapping_breaks_on_invalid_key(self) -> None:
        assert _YamlParser("sources:\n  - resource: foo\n    no_colon_here\n").parse() == {
            "sources": [{"resource": "foo"}]
        }

    def test_parse_flow_map_empty_pair(self) -> None:
        assert _YamlParser("")._parse_flow_map("{a: b, }") == {"a": "b"}


# ─── parse_concept 已知字段 ────────────────────────────────────────────────


class TestParseConceptKnownFields:
    def test_known_fields_stay_in_frontmatter_not_extra(self, tmp_path: Path) -> None:
        path = tmp_path / "concept.md"
        path.write_text(
            "---\n"
            "type: Metric\n"
            "executor:\n  resource: pipelinerun\n  receipt: [a, b]\n"
            "attester:\n  resource: verifier\n"
            "parameters:\n  - name: x\n    type: string\n"
            "sources:\n  - resource: src\n    author: Alice\n"
            "verified:\n  by: human\n  at: 2024-01-15\n"
            "---\nbody\n",
            encoding="utf-8",
        )
        concept = parse_concept(path)
        # 已知字段保留在 frontmatter、不进入 extra
        assert "executor" in concept.frontmatter
        assert "attester" in concept.frontmatter
        assert "parameters" in concept.frontmatter
        assert "sources" in concept.frontmatter
        assert "verified" in concept.frontmatter
        assert concept.extra == {}
        # verified 裸映射归一化为单元素列表
        assert concept.frontmatter["verified"] == [{"by": "human", "at": "2024-01-15"}]

    def test_tags_comma_string(self, tmp_path: Path) -> None:
        path = tmp_path / "concept.md"
        path.write_text(
            "---\ntype: Metric\ntags: alpha, beta, gamma\n---\nbody\n", encoding="utf-8"
        )
        concept = parse_concept(path)
        assert concept.tags == ["alpha", "beta", "gamma"]

    def test_tags_scalar_int(self, tmp_path: Path) -> None:
        path = tmp_path / "concept.md"
        path.write_text("---\ntype: Metric\ntags: 42\n---\nbody\n", encoding="utf-8")
        concept = parse_concept(path)
        assert concept.tags == ["42"]
