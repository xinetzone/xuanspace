"""工作区根目录发现与 ``pyproject.toml`` 解析。

本模块只做无状态纯函数：不读业务配置、不落盘，也不捕获解析异常。
"""

import sys
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ProjectType(StrEnum):
    PYTHON = "python"
    NATIVE = "native"
    STATIC = "static"
    OTHER = "other"


@dataclass
class ProjectInfo:
    """项目信息数据类"""

    name: str
    path: Path
    project_type: ProjectType
    version: str = "0.0.0"
    dependencies: list[str] = field(default_factory=list)
    pyproject_path: Path | None = None

    def __repr__(self) -> str:
        return f"ProjectInfo(name={self.name!r}, type={self.project_type!r}, version={self.version!r})"


def find_workspace_root(start_path: Path | None = None) -> Path:
    """向上查找工作区根目录。

    根目录标识：包含 ``pyproject.toml`` 且同时存在 ``apps/`` 和 ``libs/`` 目录。

    Args:
        start_path: 起始查找路径，默认为当前工作目录。

    Returns:
        工作区根目录路径。

    Raises:
        RuntimeError: 如果找不到工作区根目录。
    """
    if start_path is None:
        start_path = Path.cwd()

    current_path = start_path.resolve()

    while True:
        pyproject_path = current_path / "pyproject.toml"
        apps_dir = current_path / "apps"
        libs_dir = current_path / "libs"

        if pyproject_path.exists() and apps_dir.is_dir() and libs_dir.is_dir():
            return current_path

        parent_path = current_path.parent
        if parent_path == current_path:
            raise RuntimeError("未找到 Xuanspace 工作区根目录。请确保在包含 apps/ 和 libs/ 目录的工作区内运行此命令。")
        current_path = parent_path


def load_pyproject(path: Path) -> dict:
    """解析 ``pyproject.toml`` 文件。

    Args:
        path: pyproject.toml 文件路径。

    Returns:
        解析后的 TOML 数据字典。

    Raises:
        FileNotFoundError: 如果文件不存在。
        tomllib.TOMLDecodeError: 如果 TOML 格式错误。
    """
    if not path.exists():
        raise FileNotFoundError(f"pyproject.toml 文件不存在: {path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


def get_python_version_requirement(pyproject_data: dict) -> str | None:
    """从 ``pyproject.toml`` 数据中提取 Python 版本要求。"""
    project = pyproject_data.get("project", {})
    return project.get("requires-python")


def get_build_backend(pyproject_data: dict) -> str | None:
    """从 ``pyproject.toml`` 数据中提取构建后端。"""
    build_system = pyproject_data.get("build-system", {})
    return build_system.get("build-backend")


def get_project_dependencies(pyproject_data: dict) -> list[str]:
    """从 ``pyproject.toml`` 数据中提取项目依赖列表。"""
    project = pyproject_data.get("project", {})
    return project.get("dependencies", [])


def get_python_version() -> str:
    """获取当前 Python 版本字符串（如 ``"3.14.6"``）。"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_python_version(min_version: tuple[int, int, int] = (3, 14, 6)) -> bool:
    """检查当前 Python 版本是否满足最低要求。"""
    current = sys.version_info[:3]
    return current >= min_version
