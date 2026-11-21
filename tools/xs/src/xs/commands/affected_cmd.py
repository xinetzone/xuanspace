"""``xs affected``：基于 git diff 检测受影响的子项目。"""

import subprocess

from rich.console import Console

from ..config import find_workspace_root
from ..discovery import find_affected_projects

console = Console()


def affected() -> None:
    """基于 git diff 检测受影响的子项目"""
    root = find_workspace_root()

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=root,
        )
        changed_files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    except Exception:
        changed_files = []

    affected_projects = find_affected_projects(root, [root / f for f in changed_files])

    if not affected_projects:
        console.print("[green]没有发现受影响的子项目[/green]")
        return

    console.print("[bold]受影响的子项目:[/bold]")
    for proj in affected_projects:
        console.print(f"  [cyan]{proj.name}[/cyan] ([dim]{proj.path}[/dim])")
