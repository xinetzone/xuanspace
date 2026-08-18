"""OKF 一致性校验（§11）测试。"""

from pathlib import Path

from okf.conformance import (
    check_bundle,
    format_report,
    is_conformant,
    validate_lenient,
    validate_strict,
)
from okf.loader import load_bundle
from okf.models import Bundle, Concept, ConformanceReport

SAMPLE_BUNDLE = Path(__file__).parent / "fixtures" / "sample_bundle"


# ── 严格项 ────────────────────────────────────────────────────────────────


def test_sample_bundle_has_no_errors():
    bundle = load_bundle(SAMPLE_BUNDLE)
    report = check_bundle(bundle)
    assert report.errors == []


def test_strict_flags_missing_frontmatter_and_type():
    concept = Concept(path=Path("x.md"), type="", frontmatter={}, body="body")
    bundle = Bundle(
        root=Path("."),
        concepts={"x": concept},
        indices=[Path("index.md")],
        logs=[Path("log.md")],
    )
    errors = validate_strict(bundle)
    assert any("no frontmatter" in e for e in errors)
    assert any("empty type" in e for e in errors)


def test_strict_flags_missing_index_and_log():
    bundle = Bundle(root=Path("."), concepts={}, indices=[], logs=[])
    errors = validate_strict(bundle)
    assert any("Missing index.md" in e for e in errors)
    assert any("Missing log.md" in e for e in errors)


# ── 宽松项 ────────────────────────────────────────────────────────────────


def test_lenient_missing_optional_fields_and_broken_links(tmp_path):
    note_path = tmp_path / "concepts" / "note.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "---\ntype: Note\n---\n\n# Note\n\nSee [missing](target.md).\n",
        encoding="utf-8",
    )
    concept = Concept(
        path=note_path,
        type="Note",
        title="",
        description="",
        frontmatter={"type": "Note", "custom": "value"},
        body="See [missing](target.md).",
        extra={"custom": "value"},
    )
    bundle = Bundle(root=tmp_path, concepts={"concepts/note": concept}, indices=[], logs=[])
    warnings = validate_lenient(bundle)
    assert any("missing title" in w for w in warnings)
    assert any("missing description" in w for w in warnings)
    assert any("unknown extension keys" in w for w in warnings)
    assert any("Broken link" in w for w in warnings)
    # 宽松项输出均为警告而非错误：不以 "ERROR:" 开头
    assert not any(w.startswith("ERROR:") for w in warnings)


def test_lenient_missing_index_and_log(tmp_path):
    bundle = Bundle(root=tmp_path, concepts={}, indices=[], logs=[])
    warnings = validate_lenient(bundle)
    assert any("Missing index.md" in w for w in warnings)
    assert any("Missing log.md" in w for w in warnings)
    assert all(w.startswith("WARNING:") for w in warnings)


def test_missing_optional_fields_are_not_strict_errors(tmp_path):
    note_path = tmp_path / "note.md"
    note_path.write_text(
        "---\ntype: Note\n---\n\n# Note\n\nBody.\n",
        encoding="utf-8",
    )
    concept = Concept(
        path=note_path,
        type="Note",
        frontmatter={"type": "Note"},
        body="Body.",
    )
    bundle = Bundle(
        root=tmp_path,
        concepts={"note": concept},
        indices=[tmp_path / "index.md"],
        logs=[tmp_path / "log.md"],
    )
    errors = validate_strict(bundle)
    # 缺失 title/description 不产生严格错误
    assert not any("title" in e.lower() for e in errors)
    assert not any("description" in e.lower() for e in errors)


# ── format_report / is_conformant ─────────────────────────────────────────


def test_format_report_pass():
    bundle = load_bundle(SAMPLE_BUNDLE)
    report = check_bundle(bundle)
    text = format_report(report)
    assert "Conformance Report" in text
    assert "Result: PASS" in text


def test_format_report_fail(tmp_path):
    bundle = Bundle(root=tmp_path, concepts={}, indices=[], logs=[])
    report = check_bundle(bundle)
    text = format_report(report)
    assert "Conformance Report" in text
    assert "Result: FAIL" in text


def test_is_conformant():
    bundle = load_bundle(SAMPLE_BUNDLE)
    report = check_bundle(bundle)
    assert is_conformant(report) is True


# ── 严格项：body 为空 ─────────────────────────────────────────────────────


def test_strict_flags_empty_body(tmp_path):
    concept = Concept(
        path=tmp_path / "x.md",
        type="Metric",
        frontmatter={"type": "Metric"},
        body="   \n  ",
    )
    bundle = Bundle(
        root=tmp_path,
        concepts={"x": concept},
        indices=[tmp_path / "index.md"],
        logs=[tmp_path / "log.md"],
    )
    errors = validate_strict(bundle)
    assert any("empty body" in e for e in errors)


# ── 宽松项：概念文件不存在（FileNotFoundError） ───────────────────────────


def test_lenient_concept_file_not_found(tmp_path):
    missing_path = tmp_path / "concepts" / "ghost.md"
    concept = Concept(
        path=missing_path,
        type="Metric",
        title="Ghost",
        description="No file",
        frontmatter={"type": "Metric"},
        body="body",
    )
    bundle = Bundle(
        root=tmp_path,
        concepts={"concepts/ghost": concept},
        indices=[tmp_path / "index.md"],
        logs=[tmp_path / "log.md"],
    )
    warnings = validate_lenient(bundle)
    assert any("file not found" in w for w in warnings)


# ── format_report：warnings 为空的 "(none)" 分支 ──────────────────────────


def test_format_report_empty_warnings():
    report = ConformanceReport(errors=[], warnings=[], bundle=None)
    text = format_report(report)
    assert "Warnings (0)" in text
    assert "  (none)" in text
    assert "Bundle: N/A" in text
