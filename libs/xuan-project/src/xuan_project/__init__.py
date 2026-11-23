"""Xuan-Project：工作区项目发现与 ``pyproject.toml`` 解析工具库。"""

from .config import (
    ProjectInfo,
    ProjectType,
    check_python_version,
    find_workspace_root,
    get_build_backend,
    get_project_dependencies,
    get_python_version,
    get_python_version_requirement,
    load_pyproject,
)
from .discovery import (
    build_dependency_graph,
    detect_project_type,
    discover_projects,
    filter_projects,
    find_affected_projects,
    find_project_by_name,
    resolve_dynamic_version,
)

__version__ = "0.1.0"

__all__ = [
    "ProjectInfo",
    "ProjectType",
    "__version__",
    "build_dependency_graph",
    "check_python_version",
    "detect_project_type",
    "discover_projects",
    "filter_projects",
    "find_affected_projects",
    "find_project_by_name",
    "find_workspace_root",
    "get_build_backend",
    "get_project_dependencies",
    "get_python_version",
    "get_python_version_requirement",
    "load_pyproject",
    "resolve_dynamic_version",
]
