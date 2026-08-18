"""OKF cross-link parser: Markdown link extraction and resolution.

§6 Cross-linking and paths — extract, classify, and validate links
between concepts within a bundle.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Bundle

__all__ = [
    "parse_links",
    "parse_links_with_context",
    "check_broken_links",
    "resolve_link",
]

# Markdown link: [text](target) or [text](target "title")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")

# Strip optional title from target: "target" or "target \"title\""
_TARGET_RE = re.compile(r'^(\S+)(?:\s+"[^"]*")?$')


def _strip_target(raw: str) -> str:
    """Extract the bare target from a raw link target (strip optional title)."""
    m = _TARGET_RE.match(raw.strip())
    return m.group(1) if m else raw.strip()


def parse_links(body: str, bundle_root: Path) -> list[tuple[str, Path | str | None]]:
    """Extract Markdown links from *body* and classify each target.

    Classification rules (§6):

    - ``/`` prefix → bundle-relative absolute path, resolved to ``Path``
      by joining with *bundle_root*.
    - ``http://`` or ``https://`` → external URL, returned as ``str``.
    - ``#`` prefix → anchor, returned as ``str``.
    - Relative path → returned as ``str`` (no current-file context
      available; use :func:`parse_links_with_context` for resolution).
    - Unparseable → ``None``.

    Returns a list of ``(text, target)`` tuples.
    """
    result: list[tuple[str, Path | str | None]] = []
    for m in _LINK_RE.finditer(body):
        text = m.group(1)
        target_str = _strip_target(m.group(2))
        if not target_str:
            result.append((text, None))
            continue
        if target_str.startswith(("http://", "https://")) or target_str.startswith("#"):
            result.append((text, target_str))
        elif target_str.startswith("/"):
            result.append((text, bundle_root / target_str.lstrip("/")))
        else:
            result.append((text, target_str))  # relative, deferred
    return result


def parse_links_with_context(
    filepath: Path, bundle_root: Path
) -> list[tuple[str, Path | str | None]]:
    """Read *filepath* and extract links, resolving relative targets against
    the file's parent directory.

    Equivalent to :func:`parse_links` but with current-file context for
    relative-link resolution.
    """
    body = filepath.read_text(encoding="utf-8")
    result: list[tuple[str, Path | str | None]] = []
    for m in _LINK_RE.finditer(body):
        text = m.group(1)
        target_str = _strip_target(m.group(2))
        if not target_str:
            result.append((text, None))
            continue
        if target_str.startswith(("http://", "https://")) or target_str.startswith("#"):
            result.append((text, target_str))
        elif target_str.startswith("/"):
            result.append((text, bundle_root / target_str.lstrip("/")))
        else:
            result.append((text, filepath.parent / target_str))
    return result


def check_broken_links(
    links: list[tuple[str, Path | str | None]], bundle: Bundle
) -> list[str]:
    """Detect broken links in *links* against *bundle*.

    For each link whose *target* is a ``Path``:

    1. Checks whether the file exists on the filesystem.
    2. If the file exists, checks whether the corresponding concept ID
       (path relative to *bundle.root* with ``.md`` stripped) is present
       in ``bundle.concepts``.

    §6.1: Broken links are warnings, not errors.  Returns a list of
    human-readable warning strings.
    """
    broken: list[str] = []
    bundle_root = bundle.root.resolve()
    for text, target in links:
        if not isinstance(target, Path):
            continue
        resolved = target.resolve()
        if not resolved.exists():
            broken.append(f"Broken link: [{text}] -> {target} (not found)")
            continue
        try:
            rel = resolved.relative_to(bundle_root)
        except ValueError:
            broken.append(f"Broken link: [{text}] -> {target} (outside bundle root)")
            continue
        concept_id = rel.with_suffix("").as_posix()
        if concept_id not in bundle.concepts:
            broken.append(
                f"Broken link: [{text}] -> {target} "
                f"(concept ID '{concept_id}' not in bundle)"
            )
    return broken


def resolve_link(
    target: str, current_file: Path, bundle_root: Path
) -> Path | str | None:
    """Resolve a single link *target* string.

    - ``/`` prefix → ``bundle_root / target.lstrip('/')`` → ``Path``
    - ``http://`` or ``https://`` → ``str`` (unchanged)
    - ``#`` anchor → ``str`` (unchanged)
    - Relative → ``current_file.parent / target`` → ``Path``
    - Unparseable → ``None``
    """
    if not target:
        return None
    if target.startswith(("http://", "https://")):
        return target
    if target.startswith("#"):
        return target
    if target.startswith("/"):
        return bundle_root / target.lstrip("/")
    return current_file.parent / target
