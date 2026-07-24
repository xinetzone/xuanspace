"""Tests for xs CLI discovery module."""

from __future__ import annotations

from pathlib import Path

from xs.config import ProjectType, find_workspace_root
from xs.discovery import discover_projects


def test_find_workspace_root():
    root = find_workspace_root(Path(__file__).parent.parent)
    assert (root / "pyproject.toml").exists()
    assert (root / "apps").is_dir() or (root / "libs").is_dir()


def test_discover_projects():
    root = find_workspace_root(Path(__file__).parent.parent)
    projects = discover_projects(root)
    names = {p.name for p in projects}
    assert "xs-cli" in names
    assert "xuan-core" in names


def test_project_type_enum():
    assert ProjectType.PYTHON.value == "python"
    assert ProjectType.NATIVE.value == "native"
    assert ProjectType.STATIC.value == "static"
