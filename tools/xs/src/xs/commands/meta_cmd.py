"""
xs meta 命令模块
管理文档元数据（YAML frontmatter + TOML 二分法）
"""

from __future__ import annotations

import re
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

ALLOWED_YAML_FIELDS = {"id", "x-toml-ref", "source", "version"}
META_DIRNAME = ".meta"
TOML_SUBDIR = "toml"


def _get_meta_root(workspace_root: Path, doc_path: Path) -> Path:
    rel = doc_path.relative_to(workspace_root)
    return workspace_root / META_DIRNAME / TOML_SUBDIR / rel.parent / (doc_path.stem + ".toml")


def _has_frontmatter(content: str) -> bool:
    return content.startswith("---\n")


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    if not _has_frontmatter(content):
        return {}, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, content
    fm_text = content[4:end]
    body = content[end + 5:]
    meta: dict = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            meta[key] = val
    return meta, body


def _build_frontmatter(fields: dict) -> str:
    lines = ["---"]
    for key in ("id", "x-toml-ref", "source", "version"):
        if key in fields:
            lines.append(f"{key}: {fields[key]}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def meta_validate(
    path: Path = typer.Argument(..., help="要检查的文档文件或目录"),
    fix: bool = typer.Option(False, "--fix", help="自动修复不合规的 frontmatter"),
) -> None:
    """验证文档 frontmatter 是否符合二分法规范"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    files = _collect_doc_files(path)

    errors = 0
    fixed = 0

    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue

        fm, body = _parse_frontmatter(content)
        bad_fields = set(fm.keys()) - ALLOWED_YAML_FIELDS

        if bad_fields:
            errors += 1
            rel = f.relative_to(workspace_root)
            console.print(f"[red]✗ {rel}[/red]: 发现禁止字段: {', '.join(sorted(bad_fields))}")

            if fix:
                toml_path = _get_meta_root(workspace_root, f)
                toml_path.parent.mkdir(parents=True, exist_ok=True)
                _write_toml_meta(toml_path, fm, bad_fields)
                clean_fm = {k: v for k, v in fm.items() if k in ALLOWED_YAML_FIELDS}
                if "x-toml-ref" not in clean_fm and bad_fields:
                    rel_toml = toml_path.relative_to(workspace_root).as_posix()
                    clean_fm["x-toml-ref"] = rel_toml
                new_content = _build_frontmatter(clean_fm) + body
                f.write_text(new_content, encoding="utf-8")
                fixed += 1
                console.print(f"  [green]→ 已修复，元数据移至 {toml_path.relative_to(workspace_root)}[/green]")

    if errors == 0:
        console.print("[green]✓ 所有文档 frontmatter 合规[/green]")
    else:
        if fix:
            console.print(f"[yellow]修复了 {fixed}/{errors} 个文档[/yellow]")
        else:
            console.print(f"[yellow]发现 {errors} 个问题，使用 --fix 自动修复[/yellow]")


def _collect_doc_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for ext in ("*.md", "*.rst"):
        files.extend(path.rglob(ext))
    return sorted(files)


def _write_toml_meta(toml_path: Path, fm: dict, fields_to_move: set) -> None:
    existing: dict = {}
    if toml_path.exists():
        try:
            with open(toml_path, "rb") as f:
                existing = tomllib.load(f)
        except Exception:
            existing = {}

    title = existing.get("title", fields_to_move.pop("title", "") if "title" in fields_to_move else "")
    lines = []
    if title:
        lines.append(f'title = "{title}"')
    if "date" not in existing:
        lines.append(f'date = "{datetime.now().strftime("%Y-%m-%d")}"')

    tags = []
    for key in sorted(fields_to_move):
        val = fm.get(key, "")
        if key == "tags":
            tags = [t.strip() for t in str(val).strip("[]").split(",") if t.strip()]
            continue
        if isinstance(val, str):
            lines.append(f'{key.replace("-", "_")} = "{val}"')
        else:
            lines.append(f"{key.replace('-', '_')} = {val}")

    if tags:
        tag_lines = ", ".join(f'"{t}"' for t in tags)
        lines.append(f"tags = [{tag_lines}]")

    toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def meta_scan(
    directory: Optional[Path] = typer.Argument(None, help="要扫描的目录，默认为工作区根目录"),
) -> None:
    """扫描工作区文档元数据状态"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    scan_dir = directory or workspace_root

    md_files = _collect_doc_files(scan_dir)
    total = len(md_files)
    with_fm = 0
    bad_fm = 0
    has_toml = 0

    table = Table(title="文档元数据状态", show_header=True, header_style="bold magenta")
    table.add_column("文档", style="cyan")
    table.add_column("Frontmatter", width=12)
    table.add_column("禁止字段", width=15)
    table.add_column("TOML元数据", width=12)

    toml_root = workspace_root / META_DIRNAME / TOML_SUBDIR

    for f in md_files:
        content = f.read_text(encoding="utf-8")
        fm, _ = _parse_frontmatter(content)
        has_fm = bool(fm)
        bad = set(fm.keys()) - ALLOWED_YAML_FIELDS
        toml_path = _get_meta_root(workspace_root, f)
        toml_exists = toml_path.exists()

        if has_fm:
            with_fm += 1
        if bad:
            bad_fm += 1
        if toml_exists:
            has_toml += 1

        rel = f.relative_to(workspace_root)
        fm_status = "[green]✓[/green]" if has_fm else "[dim]-[/dim]"
        bad_status = f"[red]{', '.join(sorted(bad))}[/red]" if bad else "[dim]-[/dim]"
        toml_status = "[green]✓[/green]" if toml_exists else "[dim]-[/dim]"
        table.add_row(str(rel), fm_status, bad_status, toml_status)

    console.print(table)
    console.print()
    console.print(f"总计: {total} 个文档 | 有frontmatter: {with_fm} | 不合规: {bad_fm} | 有TOML元数据: {has_toml}")


app = typer.Typer(help="文档元数据管理命令", add_completion=False)
app.command("validate")(meta_validate)
app.command("scan")(meta_scan)
