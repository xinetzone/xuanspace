"""``xuan_project.discovery`` 的单元测试。"""

from pathlib import Path

from xuan_project import (
    ProjectInfo,
    ProjectType,
    build_dependency_graph,
    detect_project_type,
    discover_projects,
    filter_projects,
    find_affected_projects,
    find_project_by_name,
    resolve_dynamic_version,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _pyproject(path: Path, name: str, backend: str | None = None, version: str | None = None) -> Path:
    lines = ["[project]", f'name = "{name}"']
    if version is not None:
        lines.append(f'version = "{version}"')
    if backend is not None:
        lines.extend(["", "[build-system]", f'build-backend = "{backend}"'])
    return _write(path / "pyproject.toml", "\n".join(lines) + "\n")


# --- detect_project_type -------------------------------------------------


def test_detect_native_with_hyphen_and_underscore(tmp_path: Path) -> None:
    _pyproject(tmp_path / "a", "a", backend="scikit_build_core.build")
    _pyproject(tmp_path / "b", "b", backend="scikit-build-core.build")

    assert detect_project_type(tmp_path / "a") is ProjectType.NATIVE
    assert detect_project_type(tmp_path / "b") is ProjectType.NATIVE


def test_detect_python_backends(tmp_path: Path) -> None:
    for index, backend in enumerate(
        ["setuptools.build_meta", "hatchling.build", "flit_core.build", "pdm.backend", "poetry.core.masonry.api"]
    ):
        project = tmp_path / f"p{index}"
        _pyproject(project, f"p{index}", backend=backend)
        assert detect_project_type(project) is ProjectType.PYTHON


def test_detect_python_without_build_system(tmp_path: Path) -> None:
    project = tmp_path / "plain"
    _pyproject(project, "plain")

    assert detect_project_type(project) is ProjectType.PYTHON


def test_detect_python_falls_back_on_invalid_toml(tmp_path: Path) -> None:
    project = tmp_path / "broken"
    _write(project / "pyproject.toml", "not = [valid")

    assert detect_project_type(project) is ProjectType.PYTHON


def test_detect_static_and_other(tmp_path: Path) -> None:
    static = tmp_path / "site"
    _write(static / "index.html", "<html></html>")
    other = tmp_path / "empty"
    other.mkdir()

    assert detect_project_type(static) is ProjectType.STATIC
    assert detect_project_type(other) is ProjectType.OTHER


# --- resolve_dynamic_version --------------------------------------------


def test_resolve_version_from_setuptools_scm(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "demo" / "_version.py", '__version__ = version = "1.2.3"\n')
    data = {
        "tool": {
            "scikit-build": {"metadata": {"version": {"provider": "scikit_build_core.metadata.setuptools_scm"}}},
            "setuptools_scm": {"version_file": "src/demo/_version.py"},
        }
    }

    assert resolve_dynamic_version(tmp_path / "pyproject.toml", data) == "1.2.3"


def test_resolve_version_from_setuptools_scm_write_to(tmp_path: Path) -> None:
    _write(tmp_path / "_version.py", '__version__ = "2.0.0"\n')
    data = {
        "tool": {
            "scikit-build": {"metadata": {"version": {"provider": "scikit_build_core.metadata.setuptools_scm"}}},
            "setuptools_scm": {"write_to": "_version.py"},
        }
    }

    assert resolve_dynamic_version(tmp_path / "pyproject.toml", data) == "2.0.0"


def test_resolve_version_scm_missing_cases(tmp_path: Path) -> None:
    base = {
        "tool": {
            "scikit-build": {"metadata": {"version": {"provider": "scikit_build_core.metadata.setuptools_scm"}}},
        }
    }
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", base) == "0.0.0"

    missing_file = {
        "tool": {
            "scikit-build": {"metadata": {"version": {"provider": "scikit_build_core.metadata.setuptools_scm"}}},
            "setuptools_scm": {"version_file": "absent.py"},
        }
    }
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", missing_file) == "0.0.0"

    _write(tmp_path / "nomatch.py", "VERSION = '1.0.0'\n")
    no_match = {
        "tool": {
            "scikit-build": {"metadata": {"version": {"provider": "scikit_build_core.metadata.setuptools_scm"}}},
            "setuptools_scm": {"version_file": "nomatch.py"},
        }
    }
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", no_match) == "0.0.0"


def test_resolve_version_from_setuptools_attr(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "demo" / "__init__.py", '__version__ = "3.1.4"\n')
    data = {
        "tool": {
            "setuptools": {
                "dynamic": {"version": {"attr": "demo.__version__"}},
                "packages": {"find": {"where": ["src"]}},
            }
        }
    }

    assert resolve_dynamic_version(tmp_path / "pyproject.toml", data) == "3.1.4"


def test_resolve_version_setuptools_default_where(tmp_path: Path) -> None:
    _write(tmp_path / "demo" / "__init__.py", '__version__ = "4.0.0"\n')
    data = {"tool": {"setuptools": {"dynamic": {"version": {"attr": "demo.__version__"}}}}}

    assert resolve_dynamic_version(tmp_path / "pyproject.toml", data) == "4.0.0"


def test_resolve_version_setuptools_missing_cases(tmp_path: Path) -> None:
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", {"tool": {"setuptools": {}}}) == "0.0.0"

    shallow = {"tool": {"setuptools": {"dynamic": {"version": {"attr": "__version__"}}}}}
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", shallow) == "0.0.0"

    no_init = {"tool": {"setuptools": {"dynamic": {"version": {"attr": "demo.__version__"}}}}}
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", no_init) == "0.0.0"

    _write(tmp_path / "demo" / "__init__.py", "OTHER = '1.0.0'\n")
    no_match = {"tool": {"setuptools": {"dynamic": {"version": {"attr": "demo.__version__"}}}}}
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", no_match) == "0.0.0"


def test_resolve_version_returns_zero_on_malformed_data(tmp_path: Path) -> None:
    assert resolve_dynamic_version(tmp_path / "pyproject.toml", {"tool": "not-a-table"}) == "0.0.0"


# --- discover_projects ---------------------------------------------------


def test_discover_projects_scans_roots_and_skips_reserved(tmp_path: Path) -> None:
    root = tmp_path
    _pyproject(root / "apps" / "app-a", "app-a", version="1.0.0")
    _write(root / "apps" / "culture" / "app-b" / "index.html", "<html></html>")
    _pyproject(root / "libs" / "lib-a", "lib-a", backend="scikit_build_core.build")
    _pyproject(root / "tools" / "tool-a", "tool-a")

    # 以下目录应被跳过
    _pyproject(root / "libs" / ".hidden", "hidden")
    _pyproject(root / "libs" / "_private", "private")
    _pyproject(root / "libs" / "vendor", "vendor")
    _pyproject(root / "libs" / "node_modules", "node")
    _pyproject(root / "libs" / "build", "build")
    _pyproject(root / "libs" / "dist", "dist")
    _pyproject(root / "libs" / "templates", "templates")
    _pyproject(root / "libs" / "__pycache__", "cache")
    _pyproject(root / "libs" / "submodule", "submodule")
    (root / "libs" / "submodule" / ".git").mkdir()
    (root / "libs" / "plain-file.txt").write_text("x", encoding="utf-8")

    projects = discover_projects(root)
    names = {project.name for project in projects}

    assert names == {"app-a", "app-b", "lib-a", "tool-a"}

    by_name = {project.name: project for project in projects}
    assert by_name["app-a"].version == "1.0.0"
    assert by_name["app-a"].pyproject_path is not None
    assert by_name["app-b"].project_type is ProjectType.STATIC
    assert by_name["app-b"].pyproject_path is None
    assert by_name["lib-a"].project_type is ProjectType.NATIVE


def test_discover_projects_with_only_some_roots(tmp_path: Path) -> None:
    _pyproject(tmp_path / "apps" / "only-app", "only-app")

    projects = discover_projects(tmp_path)

    assert [project.name for project in projects] == ["only-app"]


def test_discover_projects_respects_max_depth(tmp_path: Path) -> None:
    _pyproject(tmp_path / "apps" / "a" / "b" / "c" / "deep", "deep")

    projects = discover_projects(tmp_path)

    assert projects == []


def test_discover_projects_extracts_dependency_names(tmp_path: Path) -> None:
    _write(
        tmp_path / "apps" / "consumer" / "pyproject.toml",
        "\n".join(
            [
                "[project]",
                'name = "consumer"',
                'version = "0.2.0"',
                'dependencies = ["xuan-core>=0.1", "demo[extra]>=2", "other; python_version > \\"3.8\\""]',
            ]
        )
        + "\n",
    )

    project = discover_projects(tmp_path)[0]

    assert project.dependencies == ["xuan-core", "demo", "other"]


def test_discover_projects_uses_dynamic_version(tmp_path: Path) -> None:
    _write(tmp_path / "libs" / "demo" / "src" / "demo" / "_version.py", '__version__ = "5.6.7"\n')
    _write(
        tmp_path / "libs" / "demo" / "pyproject.toml",
        "\n".join(
            [
                "[project]",
                'name = "demo"',
                'dynamic = ["version"]',
                "",
                "[tool.scikit-build.metadata.version]",
                'provider = "scikit_build_core.metadata.setuptools_scm"',
                "",
                "[tool.setuptools_scm]",
                'version_file = "src/demo/_version.py"',
            ]
        )
        + "\n",
    )

    project = discover_projects(tmp_path)[0]

    assert project.version == "5.6.7"


def test_discover_projects_invalid_pyproject_falls_back(tmp_path: Path) -> None:
    _write(tmp_path / "libs" / "broken" / "pyproject.toml", "not = [valid")

    project = discover_projects(tmp_path)[0]

    assert project.name == "broken"
    assert project.version == "0.0.0"
    assert project.dependencies == []
    assert project.project_type is ProjectType.PYTHON


# --- dependency graph / affected ----------------------------------------


def _project(name: str, path: Path, deps: list[str] | None = None) -> ProjectInfo:
    return ProjectInfo(
        name=name,
        path=path,
        project_type=ProjectType.PYTHON,
        dependencies=deps or [],
    )


def test_build_dependency_graph_normalizes_names(tmp_path: Path) -> None:
    lib = _project("xuan_core", tmp_path / "libs" / "xuan-core")
    consumer = _project("consumer", tmp_path / "apps" / "consumer", ["XUAN-CORE", "external"])
    projects = [lib, consumer]

    graph = build_dependency_graph(projects)

    assert graph == {"xuan_core": [], "consumer": ["xuan_core"]}


def test_find_affected_projects_empty_and_non_project(tmp_path: Path) -> None:
    _pyproject(tmp_path / "libs" / "demo", "demo", version="1.0.0")

    assert find_affected_projects(tmp_path, []) == []
    assert find_affected_projects(tmp_path, [tmp_path / "README.md"]) == []


def test_find_affected_projects_direct_and_propagation(tmp_path: Path) -> None:
    _write(
        tmp_path / "libs" / "base" / "pyproject.toml",
        '[project]\nname = "base"\nversion = "1.0.0"\n',
    )
    _write(
        tmp_path / "apps" / "consumer" / "pyproject.toml",
        '[project]\nname = "consumer"\nversion = "1.0.0"\ndependencies = ["base>=1.0"]\n',
    )
    changed = tmp_path / "libs" / "base" / "pyproject.toml"

    affected = find_affected_projects(tmp_path, [changed])

    assert [project.name for project in affected] == ["base", "consumer"]


def test_find_affected_projects_returns_sorted(tmp_path: Path) -> None:
    _pyproject(tmp_path / "libs" / "zeta", "zeta", version="1.0.0")
    _pyproject(tmp_path / "libs" / "alpha", "alpha", version="1.0.0")

    affected = find_affected_projects(
        tmp_path,
        [tmp_path / "libs" / "zeta" / "pyproject.toml", tmp_path / "libs" / "alpha" / "pyproject.toml"],
    )

    assert [project.name for project in affected] == ["alpha", "zeta"]


def test_find_project_by_name_normalizes(tmp_path: Path) -> None:
    projects = [_project("xuan-core", tmp_path / "libs" / "xuan-core")]

    assert find_project_by_name(projects, "XUAN_CORE") is projects[0]
    assert find_project_by_name(projects, "absent") is None


def test_filter_projects(tmp_path: Path) -> None:
    python_project = _project("py", tmp_path / "libs" / "py")
    native_project = ProjectInfo(
        name="native",
        path=tmp_path / "libs" / "native",
        project_type=ProjectType.NATIVE,
    )
    projects = [python_project, native_project]

    assert filter_projects(projects) == projects
    assert filter_projects(projects, project_type=ProjectType.NATIVE) == [native_project]
    assert filter_projects(projects, directory="libs") == projects
    assert filter_projects(projects, directory="apps") == []
