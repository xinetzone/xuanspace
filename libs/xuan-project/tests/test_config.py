"""``xuan_project.config`` 的单元测试。"""

from pathlib import Path

import pytest

from xuan_project import (
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


def _make_workspace(root: Path) -> Path:
    (root / "apps").mkdir()
    (root / "libs").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    return root


def test_project_type_values() -> None:
    assert ProjectType.PYTHON.value == "python"
    assert ProjectType.NATIVE.value == "native"
    assert ProjectType.STATIC.value == "static"
    assert ProjectType.OTHER.value == "other"


def test_project_info_defaults_and_repr(tmp_path: Path) -> None:
    project = ProjectInfo(name="demo", path=tmp_path, project_type=ProjectType.PYTHON)

    assert project.version == "0.0.0"
    assert project.dependencies == []
    assert project.pyproject_path is None

    text = repr(project)
    assert "name='demo'" in text
    assert "ProjectType.PYTHON" in text


def test_find_workspace_root_from_nested_dir(tmp_path: Path) -> None:
    root = _make_workspace(tmp_path)
    nested = root / "libs" / "demo" / "src"
    nested.mkdir(parents=True)

    assert find_workspace_root(nested) == root.resolve()


def test_find_workspace_root_uses_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _make_workspace(tmp_path)
    monkeypatch.chdir(root)

    assert find_workspace_root() == root.resolve()


def test_find_workspace_root_missing_raises(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="未找到 Xuanspace 工作区根目录"):
        find_workspace_root(nested)


def test_find_workspace_root_requires_both_apps_and_libs(tmp_path: Path) -> None:
    (tmp_path / "apps").mkdir()
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    nested = tmp_path / "apps" / "demo"
    nested.mkdir()

    with pytest.raises(RuntimeError):
        find_workspace_root(nested)


def test_load_pyproject_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "demo"\n', encoding="utf-8")

    assert load_pyproject(path)["project"]["name"] == "demo"

    with pytest.raises(FileNotFoundError):
        load_pyproject(tmp_path / "missing.toml")


def test_load_pyproject_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text("not = [valid", encoding="utf-8")

    with pytest.raises(Exception):
        load_pyproject(path)


def test_pyproject_field_helpers() -> None:
    data = {
        "project": {
            "requires-python": ">=3.14.6",
            "dependencies": ["xuan-core>=0.1.0"],
        },
        "build-system": {"build-backend": "hatchling.build"},
    }

    assert get_python_version_requirement(data) == ">=3.14.6"
    assert get_build_backend(data) == "hatchling.build"
    assert get_project_dependencies(data) == ["xuan-core>=0.1.0"]


def test_pyproject_field_helpers_absent() -> None:
    data: dict = {}

    assert get_python_version_requirement(data) is None
    assert get_build_backend(data) is None
    assert get_project_dependencies(data) == []


def test_python_version_helpers() -> None:
    version = get_python_version()

    assert version.startswith("3.14")
    assert version.count(".") == 2
    assert check_python_version((3, 14, 6)) is True
    assert check_python_version((99, 0, 0)) is False
