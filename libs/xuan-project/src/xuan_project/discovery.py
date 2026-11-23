"""工作区项目发现：类型检测、依赖图与受影响项目计算。

本模块只做无状态纯函数，目录扫描结果由调用方持有。
"""

import re
from pathlib import Path

from .config import (
    ProjectInfo,
    ProjectType,
    get_build_backend,
    get_project_dependencies,
    load_pyproject,
)


def detect_project_type(path: Path) -> ProjectType:
    """检测指定目录的项目类型。

    检测规则：

    - native：``build-backend`` 包含 ``scikit-build``；
    - python：有 ``pyproject.toml``，其余构建后端或解析失败；
    - static：有 ``index.html`` 但无 ``pyproject.toml``；
    - other：其他情况。
    """
    pyproject_path = path / "pyproject.toml"
    index_html_path = path / "index.html"

    has_pyproject = pyproject_path.exists()
    has_index_html = index_html_path.exists()

    if has_pyproject:
        try:
            data = load_pyproject(pyproject_path)
            build_backend = get_build_backend(data) or ""
            build_backend_lower = build_backend.lower()

            if "scikit-build" in build_backend_lower or "scikit_build" in build_backend_lower:
                return ProjectType.NATIVE
            if any(backend in build_backend_lower for backend in ["setuptools", "hatch", "flit", "pdm", "poetry"]):
                return ProjectType.PYTHON
        except Exception:
            pass
        return ProjectType.PYTHON

    if has_index_html:
        return ProjectType.STATIC

    return ProjectType.OTHER


def _extract_project_name_from_pyproject(path: Path, fallback: str) -> tuple[str, str, list[str]]:
    """从 ``pyproject.toml`` 提取项目名称、版本与依赖名。"""
    try:
        data = load_pyproject(path)
        project = data.get("project", {})
        name = project.get("name", fallback)
        version = project.get("version") or resolve_dynamic_version(path, data)
        dependencies = get_project_dependencies(data)
        dep_names = []
        for dep in dependencies:
            dep_name = re.split(r"[<>=!~;\[]", dep)[0].strip()
            dep_names.append(dep_name)
        return name, version, dep_names
    except Exception:
        return fallback, "0.0.0", []


def resolve_dynamic_version(pyproject_path: Path, data: dict) -> str:
    """解析 ``dynamic = ["version"]`` 的版本号。

    - scikit-build-core + setuptools-scm：读取 ``[tool.setuptools_scm]`` 的
      ``version_file``/``write_to`` 生成文件中的 ``__version__``；
    - setuptools：读取 ``[tool.setuptools.dynamic].version.attr``
      指向模块文件中的 ``__version__``。

    解析失败返回 ``"0.0.0"``。
    """
    try:
        tool_cfg = data.get("tool", {})
        scikit_build_cfg = tool_cfg.get("scikit-build", {})
        provider = scikit_build_cfg.get("metadata", {}).get("version", {}).get("provider", "")
        if provider == "scikit_build_core.metadata.setuptools_scm":
            scm_cfg = tool_cfg.get("setuptools_scm", {})
            version_file = scm_cfg.get("version_file") or scm_cfg.get("write_to")
            if version_file:
                version_path = pyproject_path.parent / version_file
                if version_path.exists():
                    text = version_path.read_text(encoding="utf-8")
                    match = re.search(
                        r'^__version__\s*=\s*(?:version\s*=\s*)?["\']([^"\']+)["\']',
                        text,
                        re.MULTILINE,
                    )
                    if match:
                        return match.group(1)
            return "0.0.0"

        setuptools_cfg = tool_cfg.get("setuptools", {})
        version_cfg = setuptools_cfg.get("dynamic", {}).get("version") or {}
        attr = version_cfg.get("attr", "")
        if not attr:
            return "0.0.0"
        parts = attr.split(".")
        if len(parts) < 2:
            return "0.0.0"
        module_parts, attr_name = parts[:-1], parts[-1]
        where = setuptools_cfg.get("packages", {}).get("find", {}).get("where", ["."])
        for base in where:
            init_path = pyproject_path.parent / base / Path(*module_parts) / "__init__.py"
            if not init_path.exists():
                continue
            text = init_path.read_text(encoding="utf-8")
            match = re.search(rf'{re.escape(attr_name)}\s*=\s*["\']([^"\']+)["\']', text)
            if match:
                return match.group(1)
        return "0.0.0"
    except Exception:
        return "0.0.0"


def _scan_directory(
    base_dir: Path,
    projects: list[ProjectInfo],
    max_depth: int = 2,
    current_depth: int = 0,
) -> None:
    """递归扫描目录中的项目。"""
    if not base_dir.is_dir() or current_depth > max_depth:
        return

    for item in base_dir.iterdir():
        if item.name.startswith(".") or item.name.startswith("_"):
            continue
        if not item.is_dir():
            continue

        if item.name in ("__pycache__", "node_modules", ".git", "build", "dist", "templates", "vendor"):
            continue

        if (item / ".git").exists():
            continue

        pyproject_path = item / "pyproject.toml"
        index_html_path = item / "index.html"

        if pyproject_path.exists() or index_html_path.exists():
            project_type = detect_project_type(item)
            name = item.name
            version = "0.0.0"
            dependencies: list[str] = []

            if pyproject_path.exists():
                name, version, dependencies = _extract_project_name_from_pyproject(pyproject_path, item.name)

            projects.append(
                ProjectInfo(
                    name=name,
                    path=item,
                    project_type=project_type,
                    version=version,
                    dependencies=dependencies,
                    pyproject_path=pyproject_path if pyproject_path.exists() else None,
                )
            )
        else:
            _scan_directory(item, projects, max_depth, current_depth + 1)


def discover_projects(workspace_root: Path) -> list[ProjectInfo]:
    """发现工作区中的所有项目。

    扫描 ``apps/*/``、``libs/*/`` 与 ``tools/*/``（支持一层子目录分组如
    ``apps/culture/``）。
    """
    projects: list[ProjectInfo] = []

    apps_dir = workspace_root / "apps"
    libs_dir = workspace_root / "libs"
    tools_dir = workspace_root / "tools"

    if apps_dir.is_dir():
        _scan_directory(apps_dir, projects)

    if libs_dir.is_dir():
        _scan_directory(libs_dir, projects)

    if tools_dir.is_dir():
        _scan_directory(tools_dir, projects)

    return projects


def build_dependency_graph(projects: list[ProjectInfo]) -> dict[str, list[str]]:
    """构建项目名称到其依赖项目名称列表的映射。"""
    project_names = {p.name for p in projects}
    graph: dict[str, list[str]] = {}

    for project in projects:
        deps = []
        for dep in project.dependencies:
            normalized_dep = dep.lower().replace("_", "-")
            for pname in project_names:
                if pname.lower().replace("_", "-") == normalized_dep:
                    deps.append(pname)
                    break
        graph[project.name] = deps

    return graph


def _get_project_by_path(path: Path, projects: list[ProjectInfo]) -> ProjectInfo | None:
    """根据路径查找所属项目。"""
    resolved_path = path.resolve()
    for project in projects:
        try:
            resolved_path.relative_to(project.path.resolve())
            return project
        except ValueError:
            continue
    return None


def find_affected_projects(
    workspace_root: Path,
    changed_files: list[Path],
) -> list[ProjectInfo]:
    """查找受变更文件影响的项目（含反向依赖传播）。"""
    projects = discover_projects(workspace_root)
    dep_graph = build_dependency_graph(projects)

    reverse_deps: dict[str, list[str]] = {p.name: [] for p in projects}
    for project_name, deps in dep_graph.items():
        for dep in deps:
            if dep in reverse_deps:
                reverse_deps[dep].append(project_name)

    directly_affected: set[str] = set()

    for changed_file in changed_files:
        resolved_changed = changed_file.resolve()
        project = _get_project_by_path(resolved_changed, projects)
        if project:
            directly_affected.add(project.name)

    all_affected: set[str] = set(directly_affected)
    queue = list(directly_affected)

    while queue:
        current = queue.pop(0)
        for dependent in reverse_deps.get(current, []):
            if dependent not in all_affected:
                all_affected.add(dependent)
                queue.append(dependent)

    affected_projects = [p for p in projects if p.name in all_affected]
    affected_projects.sort(key=lambda p: p.name)
    return affected_projects


def find_project_by_name(
    projects: list[ProjectInfo],
    name: str,
) -> ProjectInfo | None:
    """根据名称查找项目（大小写与下划线/连字符不敏感）。"""
    normalized = name.lower().replace("_", "-")
    for project in projects:
        if project.name.lower().replace("_", "-") == normalized:
            return project
    return None


def filter_projects(
    projects: list[ProjectInfo],
    project_type: ProjectType | None = None,
    directory: str | None = None,
) -> list[ProjectInfo]:
    """按类型和目录过滤项目。"""
    filtered = projects

    if project_type is not None:
        filtered = [p for p in filtered if p.project_type == project_type]

    if directory is not None:
        filtered = [p for p in filtered if directory.lower() in p.path.parts]

    return filtered
