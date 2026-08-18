"""OKF 信任与生命周期解析（§5）的测试。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from okf.models import Concept, TrustTier
from okf.trust import (
    _parse_date,
    _parse_datetime,
    derive_trust_tier,
    is_stale,
    parse_sources,
    parse_status,
    parse_verification,
)


def _concept(frontmatter: dict) -> Concept:
    return Concept(path=Path("concept.md"), type="Metric", frontmatter=frontmatter)


# ── parse_sources ─────────────────────────────────────────────────────────


def test_parse_sources_basic():
    fm = {
        "sources": [
            {
                "resource": "https://example.com/revenue",
                "author": "finance-team",
                "usage_count": 42,
                "last_modified": "2024-01-20",
            }
        ],
        "usage_window": {
            "from": "2023-01-01",
            "to": "2023-12-31",
        },
    }
    sources, usage_window = parse_sources(fm)
    assert len(sources) == 1
    source = sources[0]
    assert source.resource == "https://example.com/revenue"
    assert source.author == "finance-team"
    assert source.usage_count == 42
    assert source.last_modified == date(2024, 1, 20)
    assert usage_window is not None
    assert usage_window.from_date == date(2023, 1, 1)
    assert usage_window.to_date == date(2023, 12, 31)


def test_parse_sources_empty():
    sources, usage_window = parse_sources({})
    assert sources == []
    assert usage_window is None


def test_parse_sources_non_list_sources():
    sources, usage_window = parse_sources({"sources": "not-a-list"})
    assert sources == []
    assert usage_window is None


def test_parse_sources_incomplete_window():
    sources, usage_window = parse_sources({"usage_window": {"from": "2023-01-01"}})
    assert usage_window is None


# ── parse_verification ────────────────────────────────────────────────────


def test_parse_verification_generated():
    fm = {"generated": {"by": "pipeline:etl", "at": "2024-01-15T10:00:00"}}
    generated, events = parse_verification(fm)
    assert generated is not None
    assert generated.by == "pipeline:etl"
    assert generated.at == datetime(2024, 1, 15, 10, 0, 0)
    assert events == []


def test_parse_verification_bare_mapping_normalized():
    fm = {"verified": {"by": "human:alice", "at": "2024-01-18T09:30:00"}}
    generated, events = parse_verification(fm)
    assert generated is None
    assert len(events) == 1
    assert events[0].by == "human:alice"
    assert events[0].at == datetime(2024, 1, 18, 9, 30, 0)


def test_parse_verification_list():
    fm = {
        "verified": [
            {"by": "pipeline:etl", "at": "2024-01-01T00:00:00"},
            {"by": "human:bob", "at": "2024-01-02T00:00:00"},
        ]
    }
    generated, events = parse_verification(fm)
    assert generated is None
    assert [event.by for event in events] == ["pipeline:etl", "human:bob"]


def test_parse_verification_none():
    generated, events = parse_verification({})
    assert generated is None
    assert events == []


# ── derive_trust_tier ─────────────────────────────────────────────────────


def test_derive_trust_tier_unverified():
    assert derive_trust_tier(_concept({})) is TrustTier.UNVERIFIED


def test_derive_trust_tier_empty_verified_list():
    assert derive_trust_tier(_concept({"verified": []})) is TrustTier.UNVERIFIED


def test_derive_trust_tier_machine_confirmed():
    fm = {"verified": [{"by": "metric-collector", "at": "2024-01-01T00:00:00"}]}
    assert derive_trust_tier(_concept(fm)) is TrustTier.MACHINE_CONFIRMED


def test_derive_trust_tier_bare_mapping_machine():
    fm = {"verified": {"by": "pipeline:etl", "at": "2024-01-01T00:00:00"}}
    assert derive_trust_tier(_concept(fm)) is TrustTier.MACHINE_CONFIRMED


def test_derive_trust_tier_human_reviewed():
    fm = {
        "verified": [
            {"by": "pipeline:etl", "at": "2024-01-01T00:00:00"},
            {"by": "human:alice", "at": "2024-01-02T00:00:00"},
        ]
    }
    assert derive_trust_tier(_concept(fm)) is TrustTier.HUMAN_REVIEWED


# ── is_stale ──────────────────────────────────────────────────────────────


def test_is_stale_after():
    concept = _concept({"stale_after": "2024-01-01"})
    assert is_stale(concept, date(2024, 1, 1)) is True
    assert is_stale(concept, date(2024, 1, 2)) is True
    assert is_stale(concept, date(2023, 12, 31)) is False


def test_is_stale_no_field():
    assert is_stale(_concept({}), date(2100, 1, 1)) is False


# ── parse_status ──────────────────────────────────────────────────────────


def test_parse_status_known():
    assert parse_status({"status": "draft"}) == "draft"
    assert parse_status({"status": "stable"}) == "stable"
    assert parse_status({"status": "deprecated"}) == "deprecated"


def test_parse_status_default_draft():
    assert parse_status({}) == "draft"
    assert parse_status({"status": "unknown"}) == "draft"


# ── 非 dict 条目 / 边界分支 ────────────────────────────────────────────────


def test_parse_sources_skips_non_dict_items():
    sources, _ = parse_sources({"sources": ["not-a-dict", 123, {"resource": "ok"}]})
    assert len(sources) == 1
    assert sources[0].resource == "ok"


def test_parse_verification_skips_non_dict_items():
    generated, events = parse_verification(
        {"verified": [123, "abc", {"by": "human:x", "at": "2024-01-01T00:00:00"}]}
    )
    assert generated is None
    assert len(events) == 1
    assert events[0].by == "human:x"


def test_derive_trust_tier_non_list_non_dict():
    assert derive_trust_tier(_concept({"verified": "just-a-string"})) is TrustTier.UNVERIFIED


def test_derive_trust_tier_no_meaningful_by():
    # verified 列表有 dict 但 by 全为空 → has_any 为 False → UNVERIFIED
    assert derive_trust_tier(_concept({"verified": [{}]})) is TrustTier.UNVERIFIED
    assert derive_trust_tier(_concept({"verified": [{"by": ""}]})) is TrustTier.UNVERIFIED


def test_is_stale_invalid_date():
    assert is_stale(_concept({"stale_after": "not-a-date"}), date(2100, 1, 1)) is False
    assert is_stale(_concept({"stale_after": 123}), date(2100, 1, 1)) is False


# ── 私有日期解析辅助函数 ───────────────────────────────────────────────────


def test_parse_date_value_error():
    assert _parse_date("2024-13-45") is None
    assert _parse_date("garbage") is None


def test_parse_datetime_non_str_or_empty():
    assert _parse_datetime(None) is None
    assert _parse_datetime("") is None
    assert _parse_datetime(123) is None


def test_parse_datetime_value_error():
    assert _parse_datetime("not-a-datetime") is None
