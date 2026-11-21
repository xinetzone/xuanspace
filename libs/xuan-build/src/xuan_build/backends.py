"""构建后端协议与内置实现。

后端只负责「如何构建」与「是否需要跳过」，不负责进度条、成功/失败汇总等
展示编排；展示通过 :class:`Reporter` 注入，便于测试替换为记录器。
"""

import subprocess
import sys
from pathlib import Path
from typing import Protocol


class Reporter(Protocol):
    """最小输出端口：Rich ``Console`` 结构上满足。"""

    def print(self, message: str) -> None: ...


class BuildBackend(Protocol):
    """项目类型对应的构建策略。"""

    kind: str

    def skip_message(self, project_name: str) -> str | None:
        """无需构建时返回提示文案；需要构建时返回 ``None``。"""
        ...

    def build(self, project_path: Path, reporter: Reporter) -> bool:
        """执行构建并返回是否成功。"""
        ...


class PythonBackend:
    """纯 Python 项目：``python -m build --wheel``。"""

    kind = "python"

    def skip_message(self, project_name: str) -> str | None:
        return None

    def build(self, project_path: Path, reporter: Reporter) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel"],
                cwd=project_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            reporter.print("[red]构建失败:[/red]")
            if exc.stdout:
                reporter.print(f"[red]{exc.stdout}[/red]")
            if exc.stderr:
                reporter.print(f"[red]{exc.stderr}[/red]")
            return False


class NativeBackend:
    """C/C++ 原生扩展：优先 ``pip install -e .``，失败回退 wheel 构建。"""

    kind = "native"

    def skip_message(self, project_name: str) -> str | None:
        return None

    def build(self, project_path: Path, reporter: Reporter) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=project_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as first_err:
            reporter.print("[yellow]pip install -e 失败，回退 python -m build --wheel[/yellow]")
            try:
                subprocess.run(
                    [sys.executable, "-m", "build", "--wheel"],
                    cwd=project_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return True
            except subprocess.CalledProcessError as exc:
                # ninja/CMake 的 FAILED: 详情写入 stdout，两路输出都必须保留
                reporter.print("[red]构建失败:[/red]")
                for err in (first_err, exc):
                    if err.stdout:
                        reporter.print(f"[red]{err.stdout}[/red]")
                    if err.stderr:
                        reporter.print(f"[red]{err.stderr}[/red]")
                return False


class StaticBackend:
    """静态站点：无需构建。"""

    kind = "static"

    def skip_message(self, project_name: str) -> str | None:
        return f"[yellow]跳过静态项目: {project_name} (无需构建)[/yellow]"

    def build(self, project_path: Path, reporter: Reporter) -> bool:
        return True


class OtherBackend:
    """未知类型：保持既有「跳过」语义。"""

    kind = "other"

    def skip_message(self, project_name: str) -> str | None:
        return f"[yellow]跳过未知类型项目: {project_name}[/yellow]"

    def build(self, project_path: Path, reporter: Reporter) -> bool:
        return True
