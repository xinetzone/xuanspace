"""
xs lfs 命令模块
Git LFS 大文件检查与管理
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()

LFS_THRESHOLD_MB = 5  # 超过此大小的大文件建议使用 LFS

app = typer.Typer(help="Git LFS 大文件管理", add_completion=False)


def _parse_lfs_patterns(workspace_root: Path) -> list[str]:
    """从 .gitattributes 中解析 LFS 跟踪的文件模式"""
    ga_path = workspace_root / ".gitattributes"
    if not ga_path.exists():
        return []

    patterns: list[str] = []
    with open(ga_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "filter=lfs" in line:
                pattern = line.split()[0]
                patterns.append(pattern)
    return patterns


def _get_lfs_tracked_files(workspace_root: Path) -> set[str]:
    """获取已通过 LFS 跟踪的文件列表"""
    try:
        result = subprocess.run(
            ["git", "lfs", "ls-files", "--name-only"],
            capture_output=True,
            text=True,
            check=False,
            cwd=workspace_root,
        )
        return {f.strip() for f in result.stdout.split("\n") if f.strip()}
    except Exception:
        return set()


def _find_large_files(workspace_root: Path, threshold_mb: int = LFS_THRESHOLD_MB) -> list[tuple[Path, float]]:
    """查找工作区中超过阈值但未被 LFS 跟踪的大文件"""
    large_files: list[tuple[Path, float]] = []
    lfs_tracked = _get_lfs_tracked_files(workspace_root)
    lfs_patterns = _parse_lfs_patterns(workspace_root)

    # 排除的目录
    exclude_dirs = {".git", ".venv", "__pycache__", "node_modules", "build", "dist", "_build", ".pytest_cache", "attic"}

    for item in workspace_root.rglob("*"):
        if item.is_dir():
            continue
        # 跳过排除目录
        parts = item.relative_to(workspace_root).parts
        if any(p in exclude_dirs for p in parts):
            continue

        rel_path = item.relative_to(workspace_root).as_posix()

        # 检查是否匹配 LFS 模式但未通过 LFS 跟踪
        if lfs_patterns and not item.is_symlink():
            is_lfs_pattern = any(item.match(pattern) for pattern in lfs_patterns)
            if is_lfs_pattern and rel_path not in lfs_tracked:
                size_mb = item.stat().st_size / (1024 * 1024)
                large_files.append((item, round(size_mb, 2)))
                continue

        # 检查大文件（超过阈值且未被 LFS 跟踪）
        if item.is_file():
            size_bytes = item.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            if size_mb >= threshold_mb and rel_path not in lfs_tracked:
                large_files.append((item, round(size_mb, 2)))

    return large_files


@app.command("check")
def lfs_check(
    threshold: int = typer.Option(
        LFS_THRESHOLD_MB,
        "--threshold",
        "-t",
        help="大文件阈值 (MB)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="JSON 格式输出",
    ),
) -> None:
    """检查是否有应使用 LFS 但未跟踪的大文件"""
    from ..config import find_workspace_root

    root = find_workspace_root()
    lfs_patterns = _parse_lfs_patterns(root)
    large_files = _find_large_files(root, threshold)

    if json_output:
        import json

        result = {
            "lfs_patterns": lfs_patterns,
            "large_files": [{"path": str(item[0].relative_to(root)), "size_mb": item[1]} for item in large_files],
            "total": len(large_files),
        }
        console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 显示 LFS 跟踪模式
    console.print("[bold]Git LFS 跟踪模式:[/bold]")
    if lfs_patterns:
        for pattern in lfs_patterns:
            console.print(f"  [dim]{pattern}[/dim]")
    else:
        console.print("  [dim]未配置 LFS 跟踪模式[/dim]")

    console.print()

    # 显示检测结果
    if not large_files:
        console.print("[green]✓ 未发现应使用 LFS 的大文件[/green]")
        return

    console.print(f"[bold yellow]⚠ 发现 {len(large_files)} 个应使用 LFS 的文件:[/bold yellow]")
    console.print()

    table = Table(title="待 LFS 跟踪的文件")
    table.add_column("文件路径", style="cyan")
    table.add_column("大小 (MB)", style="yellow", justify="right")
    table.add_column("建议", style="dim")

    for file_path, size_mb in large_files:
        rel = file_path.relative_to(root).as_posix()
        suggestion = _get_suggestion(rel, lfs_patterns, size_mb)
        table.add_row(rel, f"{size_mb:.2f}", suggestion)

    console.print(table)
    console.print()
    console.print(
        "[dim]提示: 运行 [bold]git lfs track <pattern>[/bold] 添加跟踪规则，然后 [bold]git add .gitattributes[/bold] 提交[/dim]"
    )


def _get_suggestion(rel_path: str, lfs_patterns: list[str], size_mb: float) -> str:
    """根据文件路径生成跟踪建议"""
    suffix = Path(rel_path).suffix
    if suffix:
        pattern = f"*{suffix}"
        if pattern in lfs_patterns:
            if size_mb <= 0.01:
                return "文件很小，可能已是 LFS 指针"
            return "匹配现有规则，运行 git lfs migrate"
        return f'建议: git lfs track "{pattern}"'
    return '建议: git lfs track "完整路径"'


@app.command("patterns")
def lfs_patterns() -> None:
    """列出当前 .gitattributes 中的 LFS 跟踪模式"""
    from ..config import find_workspace_root

    root = find_workspace_root()
    patterns = _parse_lfs_patterns(root)

    if not patterns:
        console.print("[dim]未配置 LFS 跟踪模式[/dim]")
        return

    console.print("[bold]Git LFS 跟踪模式:[/bold]")
    for pattern in patterns:
        console.print(f"  [cyan]{pattern}[/cyan]")
