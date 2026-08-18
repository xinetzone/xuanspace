"""OKF v0.2 信任与生命周期解析（§5）。

零第三方依赖，仅使用 Python 标准库 ``datetime``。
"""

from __future__ import annotations

from datetime import date, datetime

from .models import (
    Concept,
    GeneratedInfo,
    Source,
    TrustTier,
    UsageWindow,
    VerificationEvent,
)

__all__ = [
    "derive_trust_tier",
    "is_stale",
    "parse_sources",
    "parse_status",
    "parse_verification",
]


def parse_sources(frontmatter: dict) -> tuple[list[Source], UsageWindow | None]:
    """解析 ``sources`` 数组与 ``usage_window``。

    Returns:
        ``(sources, usage_window)`` 元组。
    """
    sources_raw = frontmatter.get("sources", [])
    if not isinstance(sources_raw, list):
        sources_raw = []

    sources: list[Source] = []
    for item in sources_raw:
        if not isinstance(item, dict):
            continue
        last_modified = _parse_date(item.get("last_modified"))
        sources.append(
            Source(
                resource=str(item.get("resource", "")),
                id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                author=str(item.get("author", "")),
                usage_count=int(item.get("usage_count", 0)),
                last_modified=last_modified,
            )
        )

    usage_window: UsageWindow | None = None
    uw_raw = frontmatter.get("usage_window")
    if isinstance(uw_raw, dict):
        from_date = _parse_date(uw_raw.get("from"))
        to_date = _parse_date(uw_raw.get("to"))
        if from_date is not None and to_date is not None:
            usage_window = UsageWindow(from_date=from_date, to_date=to_date)

    return sources, usage_window


def parse_verification(frontmatter: dict) -> tuple[GeneratedInfo | None, list[VerificationEvent]]:
    """解析 ``generated`` 与 ``verified``。

    ``verified`` 支持列表格式与裸映射格式；裸映射将被归一化为单元素列表（§5.2）。

    Returns:
        ``(generated, verification_events)`` 元组。
    """
    generated: GeneratedInfo | None = None
    gen_raw = frontmatter.get("generated")
    if isinstance(gen_raw, dict):
        by = str(gen_raw.get("by", ""))
        at = _parse_datetime(gen_raw.get("at"))
        if by and at is not None:
            generated = GeneratedInfo(by=by, at=at)

    verified_raw = frontmatter.get("verified")
    if not isinstance(verified_raw, list):
        verified_raw = [verified_raw] if isinstance(verified_raw, dict) else []

    verification_events: list[VerificationEvent] = []
    for item in verified_raw:
        if not isinstance(item, dict):
            continue
        by = str(item.get("by", ""))
        at = _parse_datetime(item.get("at"))
        if by and at is not None:
            verification_events.append(VerificationEvent(by=by, at=at))

    return generated, verification_events


def derive_trust_tier(concept: Concept) -> TrustTier:
    """信任等级推导（§5.3）。

    推导规则：
    - 无 ``verified`` 字段 → ``TrustTier.UNVERIFIED``
    - 有 ``verified`` 但所有 ``by`` 字段都不含 ``human:`` 前缀 → ``TrustTier.MACHINE_CONFIRMED``
    - 至少一个 ``by`` 字段含 ``human:<id>`` 前缀 → ``TrustTier.HUMAN_REVIEWED``
    """
    verified_raw = concept.frontmatter.get("verified")
    if verified_raw is None:
        return TrustTier.UNVERIFIED

    # 归一化：裸映射等价于单元素列表
    if not isinstance(verified_raw, list):
        if isinstance(verified_raw, dict):
            verified_raw = [verified_raw]
        else:
            return TrustTier.UNVERIFIED

    if len(verified_raw) == 0:
        return TrustTier.UNVERIFIED

    has_human = False
    has_any = False
    for item in verified_raw:
        if isinstance(item, dict):
            by_value = str(item.get("by", ""))
            if by_value:
                has_any = True
                if by_value.startswith("human:"):
                    has_human = True
                    break  # 只要有一个 human: 就足够判定

    if not has_any:
        return TrustTier.UNVERIFIED
    if has_human:
        return TrustTier.HUMAN_REVIEWED
    return TrustTier.MACHINE_CONFIRMED


def is_stale(concept: Concept, today: date | None = None) -> bool:
    """保鲜判定（§5.5）。

    Args:
        concept: 待判定的概念。
        today: 参考日期，默认为 ``date.today()``。

    Returns:
        ``today >= stale_after`` 时返回 ``True``；无 ``stale_after`` 字段时返回 ``False``（永不过期）。
    """
    if today is None:
        today = date.today()

    stale_after_raw = concept.frontmatter.get("stale_after")
    if stale_after_raw is None:
        return False

    stale_date = _parse_date(stale_after_raw)
    if stale_date is None:
        return False

    return today >= stale_date


def parse_status(frontmatter: dict) -> str:
    """解析 ``status`` 字段。

    Returns:
        ``"draft"`` / ``"stable"`` / ``"deprecated"``，默认返回 ``"draft"``。
    """
    raw = frontmatter.get("status")
    if isinstance(raw, str) and raw.strip() in ("draft", "stable", "deprecated"):
        return raw.strip()
    return "draft"


# ─── 内部辅助函数 ─────────────────────────────────────────────────────────


def _parse_date(value) -> date | None:
    """将 ISO 日期字符串（如 ``"2025-06-01"``）解析为 :class:`date`。"""
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_datetime(value) -> datetime | None:
    """将 ISO 日期时间字符串解析为 :class:`datetime`。"""
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None
