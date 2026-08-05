"""
xs version 命令模块
版本管理：bump、changelog 生成
"""


import re
import subprocess
from datetime import date
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console

from ..config import find_workspace_root, load_pyproject
from ..discovery import discover_projects, find_project_by_name

console = Console()


class BumpPart(StrEnum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


def _parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return 0, 0, 0
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _bump_version(version: str, part: BumpPart) -> str:
    major, minor, patch = _parse_version(version)
    if part == BumpPart.MAJOR:
        return f"{major + 1}.0.0"
    elif part == BumpPart.MINOR:
        return f"{major}.{minor + 1}.0"
    else:
        return f"{major}.{minor}.{patch + 1}"


def _run_git_log(since_tag: str | None = None, cwd: Path | None = None) -> list[str]:
    args = ["log", "--oneline", "--no-merges"]
    if since_tag:
        args.append(f"{since_tag}..HEAD")
    else:
        args.append("-20")
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
    except FileNotFoundError:
        pass
    return []


def version_show(
    project_name: str | None = typer.Option(None, "--project", "-p", help="指定项目"),
) -> None:
    """显示当前版本"""
    workspace_root = find_workspace_root()
    projects = discover_projects(workspace_root)

    if project_name:
        proj = find_project_by_name(projects, project_name)
        if proj is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            raise typer.Exit(1)
        console.print(f"[bold cyan]{proj.name}[/bold cyan]: [green]{proj.version}[/green]")
        return

    try:
        root_data = load_pyproject(workspace_root / "pyproject.toml")
        root_ver = root_data.get("project", {}).get("version", "0.0.0")
        console.print(f"[bold]Xuanspace 工作区版本:[/bold] [green]{root_ver}[/green]")
    except Exception:
        pass
    console.print()
    for proj in sorted(projects, key=lambda p: p.name):
        console.print(
            f"  [cyan]{proj.name}[/cyan]: [green]{proj.version}[/green] [dim]({proj.project_type.value})[/dim]"
        )


def version_bump(
    part: BumpPart = typer.Argument(..., help="要升级的版本部分: major/minor/patch"),
    project_name: str | None = typer.Option(None, "--project", "-p", help="指定项目"),
    dry_run: bool = typer.Option(False, "--dry-run", help="仅显示将要做的更改，不实际修改"),
) -> None:
    """升级版本号"""
    workspace_root = find_workspace_root()
    projects = discover_projects(workspace_root)

    targets = projects
    if project_name:
        proj = find_project_by_name(projects, project_name)
        if proj is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            raise typer.Exit(1)
        targets = [proj]

    for proj in targets:
        if not proj.pyproject_path:
            continue
        old_ver = proj.version
        new_ver = _bump_version(old_ver, part)
        console.print(f"[cyan]{proj.name}[/cyan]: {old_ver} → [green]{new_ver}[/green]")

        if not dry_run:
            content = proj.pyproject_path.read_text(encoding="utf-8")
            new_content = re.sub(
                r'(version\s*=\s*)"' + re.escape(old_ver) + r'"',
                f'\\1"{new_ver}"',
                content,
                count=1,
            )
            proj.pyproject_path.write_text(new_content, encoding="utf-8")

            # 同步更新 __init__.py 中的 __version__（单源版本化）
            # 仅对 xs-cli 生效：pyproject 中 version 设为 dynamic，从 xs.__version__ 读取
            if proj.name == "xs-cli":
                init_path = proj.pyproject_path.parent / "src" / "xs" / "__init__.py"
                if init_path.exists():
                    init_content = init_path.read_text(encoding="utf-8")
                    new_init_content = re.sub(
                        r'(__version__\s*=\s*)"' + re.escape(old_ver) + r'"',
                        f'\\1"{new_ver}"',
                        init_content,
                        count=1,
                    )
                    init_path.write_text(new_init_content, encoding="utf-8")

            changelog = proj.path / "CHANGELOG.md"
            _append_changelog(changelog, new_ver, proj.path)

    if dry_run:
        console.print("[yellow](dry run 模式，未实际修改)[/yellow]")


def _append_changelog(changelog_path: Path, version: str, cwd: Path) -> None:
    """追加 CHANGELOG 条目"""
    today = date.today().isoformat()
    commits = _run_git_log(cwd=cwd)

    header = f"## [{version}] - {today}\n\n"
    if commits:
        entries = "\n".join(f"- {c.split(' ', 1)[1] if ' ' in c else c}" for c in commits if c)
        entry = header + entries + "\n\n"
    else:
        entry = header + "- 版本更新\n\n"

    if changelog_path.exists():
        content = changelog_path.read_text(encoding="utf-8")
        marker = "## ["
        idx = content.find(marker, 10)
        if idx > 0:
            content = content[:idx] + entry + content[idx:]
        else:
            content = content.rstrip() + "\n\n" + entry
        changelog_path.write_text(content, encoding="utf-8")
    else:
        changelog_path.write_text(
            f"# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n{entry}",
            encoding="utf-8",
        )


app = typer.Typer(help="版本管理命令", add_completion=False)
app.command("show")(version_show)
app.command("bump")(version_bump)
