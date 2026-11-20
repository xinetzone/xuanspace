"""工程子目录发现与定位。"""

import pytest

from xuan_cmd import DEFAULT_MARKER, XuanCmdError, list_projects, locate_project


def _make(root, name, marker=DEFAULT_MARKER):
    project = root / name
    project.mkdir()
    (project / marker).write_text("", encoding="utf-8")
    return project


def test_default_marker_is_cmakelists():
    assert DEFAULT_MARKER == "CMakeLists.txt"


def test_list_projects_is_sorted_and_filtered(tmp_path):
    _make(tmp_path, "b")
    _make(tmp_path, "a")
    _make(tmp_path, "c", marker="pyproject.toml")
    (tmp_path / "noise").mkdir()
    assert [p.name for p in list_projects(tmp_path)] == ["a", "b"]


def test_list_projects_custom_marker(tmp_path):
    _make(tmp_path, "a", marker="pyproject.toml")
    assert [p.name for p in list_projects(tmp_path, marker="pyproject.toml")] == ["a"]


def test_list_projects_empty(tmp_path):
    assert list_projects(tmp_path) == []


def test_list_projects_missing_root(tmp_path):
    with pytest.raises(XuanCmdError, match="根目录不存在"):
        list_projects(tmp_path / "absent")


def test_list_projects_returns_children_of_root(tmp_path):
    project = _make(tmp_path, "app")
    assert list_projects(tmp_path) == [project]


def test_locate_project_returns_path(tmp_path):
    project = _make(tmp_path, "app")
    assert locate_project(tmp_path, "app") == project


def test_locate_project_missing_marker(tmp_path):
    (tmp_path / "app").mkdir()
    with pytest.raises(XuanCmdError, match="未找到应用：app"):
        locate_project(tmp_path, "app", label="应用")


def test_locate_project_unknown_name(tmp_path):
    with pytest.raises(XuanCmdError, match="没有含 CMakeLists.txt 的同名目录"):
        locate_project(tmp_path, "ghost")
