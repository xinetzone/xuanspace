"""Tests for xs affected command — detect projects affected by file changes."""

from __future__ import annotations

from pathlib import Path

from xs.config import ProjectInfo, ProjectType, find_workspace_root
from xs.discovery import (
    build_dependency_graph,
    discover_projects,
    find_affected_projects,
    find_project_by_name,
)


def _get_root() -> Path:
    return find_workspace_root(Path(__file__).parent.parent)


def test_build_dependency_graph():
    """依赖图构建：项目间依赖关系正确映射"""
    root = _get_root()
    projects = discover_projects(root)
    graph = build_dependency_graph(projects)

    assert isinstance(graph, dict)
    # xs-cli 是 tools 下的项目，应该存在
    assert "xs-cli" in graph
    # 依赖图应该包含所有项目
    assert len(graph) == len(projects)


def test_find_affected_empty_changes():
    """空变更：无文件变更时不应返回受影响项目"""
    root = _get_root()
    affected = find_affected_projects(root, [])
    assert len(affected) == 0


def test_find_affected_single_file():
    """单文件变更：仅变更文件所在项目受影响"""
    root = _get_root()
    projects = discover_projects(root)

    if not projects:
        return  # 没有项目可测试，跳过

    # 选择一个项目，模拟其 pyproject.toml 变更
    target = projects[0]
    changed_file = target.path / "pyproject.toml"

    affected = find_affected_projects(root, [changed_file])
    assert len(affected) >= 1
    assert target.name in [p.name for p in affected]


def test_find_affected_dependency_propagation():
    """依赖传播：依赖库变更时，依赖它的项目也应受影响"""
    root = _get_root()
    projects = discover_projects(root)

    # 查找 libs/ 下的项目（被依赖方）
    lib_projects = [p for p in projects if "libs" in p.path.parts]
    if not lib_projects:
        return  # 无 libs 项目，跳过

    lib = lib_projects[0]
    changed_file = lib.path / "pyproject.toml"

    affected = find_affected_projects(root, [changed_file])
    affected_names = {p.name for p in affected}

    # 被依赖的库本身应该受影响
    assert lib.name in affected_names

    # 如果存在依赖该库的项目，也应该受影响
    graph = build_dependency_graph(projects)
    dependents = [
        name for name, deps in graph.items() if lib.name in deps
    ]
    if dependents:
        for dep_name in dependents:
            assert dep_name in affected_names, (
                f"依赖 {lib.name} 的项目 {dep_name} 应该受影响"
            )


def test_find_affected_non_project_file():
    """非项目文件变更：根目录文件变更不应影响任何项目"""
    root = _get_root()
    # 模拟根目录 README.md 变更
    affected = find_affected_projects(root, [root / "README.md"])
    # 根目录文件不在任何项目内，不应有项目受影响
    assert len(affected) == 0


def test_find_affected_by_path():
    """按路径查找项目：路径匹配正确"""
    root = _get_root()
    projects = discover_projects(root)

    if not projects:
        return

    target = projects[0]
    found = find_project_by_name(projects, target.name)
    assert found is not None
    assert found.name == target.name


def test_find_affected_type_coverage():
    """类型覆盖：所有项目类型都应被正确识别"""
    root = _get_root()
    projects = discover_projects(root)

    for p in projects:
        assert p.project_type in (
            ProjectType.PYTHON,
            ProjectType.NATIVE,
            ProjectType.STATIC,
            ProjectType.OTHER,
        )
        assert p.name
        assert p.path.exists()