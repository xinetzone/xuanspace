"""OKF 交叉链接解析（§6）测试。"""

from __future__ import annotations

from pathlib import Path

from okf.links import (
    check_broken_links,
    parse_links,
    parse_links_with_context,
    resolve_link,
)
from okf.models import Bundle

# ── parse_links ───────────────────────────────────────────────────────────


def test_parse_links_classifies_targets(tmp_path):
    body = (
        "[root](/abs/path.md) "
        "[ext](https://example.com) "
        "[anch](#section) "
        "[rel](foo/bar.md)"
    )
    links = parse_links(body, tmp_path)
    assert [text for text, _ in links] == ["root", "ext", "anch", "rel"]
    assert links[0][1] == tmp_path / "abs" / "path.md"
    assert links[1][1] == "https://example.com"
    assert links[2][1] == "#section"
    assert links[3][1] == "foo/bar.md"


def test_parse_links_unparseable():
    body = "[empty]()"
    links = parse_links(body, Path("."))
    assert links == [("empty", None)]


# ── parse_links_with_context ──────────────────────────────────────────────


def test_parse_links_with_context_resolves_relative(tmp_path):
    current = tmp_path / "concepts" / "a.md"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("See [b](../b.md) and [c](/c.md).", encoding="utf-8")
    links = parse_links_with_context(current, tmp_path)
    assert links[0][1] == current.parent / "../b.md"
    assert links[1][1] == tmp_path / "c.md"


def test_parse_links_with_context_empty_and_external(tmp_path):
    current = tmp_path / "a.md"
    current.write_text("[empty]() [ext](https://example.com) [anch](#x)", encoding="utf-8")
    links = parse_links_with_context(current, tmp_path)
    assert links[0] == ("empty", None)
    assert links[1][1] == "https://example.com"
    assert links[2][1] == "#x"


# ── check_broken_links ────────────────────────────────────────────────────


def test_check_broken_links_not_found(tmp_path):
    bundle = Bundle(root=tmp_path, concepts={}, indices=[], logs=[])
    links = [("missing", tmp_path / "missing.md")]
    broken = check_broken_links(links, bundle)
    assert len(broken) == 1
    assert "not found" in broken[0]


def test_check_broken_links_concept_not_in_bundle(tmp_path):
    existing = tmp_path / "orphan.md"
    existing.write_text("body", encoding="utf-8")
    bundle = Bundle(root=tmp_path, concepts={}, indices=[], logs=[])
    links = [("orphan", existing)]
    broken = check_broken_links(links, bundle)
    assert len(broken) == 1
    assert "not in bundle" in broken[0]


def test_check_broken_links_ok(tmp_path):
    existing = tmp_path / "target.md"
    existing.write_text("body", encoding="utf-8")
    bundle = Bundle(root=tmp_path, concepts={"target": object()}, indices=[], logs=[])
    links = [("target", existing)]
    assert check_broken_links(links, bundle) == []


def test_check_broken_links_skips_non_path_targets():
    bundle = Bundle(root=Path("."), concepts={}, indices=[], logs=[])
    links = [("ext", "https://example.com"), ("anch", "#section"), ("none", None)]
    assert check_broken_links(links, bundle) == []


def test_check_broken_links_outside_bundle(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("body", encoding="utf-8")
    bundle = Bundle(root=bundle_root, concepts={}, indices=[], logs=[])
    links = [("outside", outside)]
    broken = check_broken_links(links, bundle)
    assert len(broken) == 1
    assert "outside bundle root" in broken[0]


# ── resolve_link ──────────────────────────────────────────────────────────


def test_resolve_link_variants(tmp_path):
    current = tmp_path / "dir" / "a.md"
    assert resolve_link("/abs/x.md", current, tmp_path) == tmp_path / "abs" / "x.md"
    assert resolve_link("https://example.com", current, tmp_path) == "https://example.com"
    assert resolve_link("#anchor", current, tmp_path) == "#anchor"
    assert resolve_link("rel.md", current, tmp_path) == current.parent / "rel.md"
    assert resolve_link("", current, tmp_path) is None
