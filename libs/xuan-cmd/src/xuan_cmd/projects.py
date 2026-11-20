"""按标记文件发现工程子目录：不维护第二份目录清单。"""

from os import PathLike
from pathlib import Path

from .errors import XuanCmdError

DEFAULT_MARKER = "CMakeLists.txt"
PathInput = str | PathLike[str]


def list_projects(root: PathInput, *, marker: str = DEFAULT_MARKER) -> list[Path]:
    """返回 ``root`` 下所有含 ``marker`` 文件的子目录，按目录名排序。

    返回路径沿用 ``root`` 的相对性（不做 ``resolve``），调用方可自行决定是否规范化。
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise XuanCmdError(f"根目录不存在：{root_path}")
    return sorted(
        (child for child in root_path.iterdir() if (child / marker).is_file()),
        key=lambda child: child.name,
    )


def locate_project(
    root: PathInput,
    name: str,
    *,
    marker: str = DEFAULT_MARKER,
    label: str = "项目",
) -> Path:
    """返回 ``root/name``；该目录缺少 ``marker`` 时抛 :class:`XuanCmdError`。"""
    root_path = Path(root)
    project = root_path / name
    if not (project / marker).is_file():
        raise XuanCmdError(f"未找到{label}：{name}（{root_path} 下没有含 {marker} 的同名目录）")
    return project
