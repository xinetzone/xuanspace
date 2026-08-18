"""Tests for xs CLI config module."""


from xs.config import check_python_version, get_python_version


def test_python_version():
    version = get_python_version()
    assert version.startswith("3.14")


def test_check_python_version():
    assert check_python_version((3, 14, 6)) is True
    assert check_python_version((3, 15)) is False
