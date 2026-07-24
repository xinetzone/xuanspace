"""Tests for xs CLI config module."""

from __future__ import annotations

from xs.config import check_python_version, get_python_version


def test_python_version():
    version = get_python_version()
    assert version.startswith("3.13")


def test_check_python_version():
    assert check_python_version((3, 13)) is True
    assert check_python_version((3, 14)) is False
