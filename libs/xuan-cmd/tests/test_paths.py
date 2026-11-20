"""目录解析：优先级、相对路径锚点、存在性断言。"""

import os
from pathlib import Path

import pytest

from xuan_cmd import XuanCmdError, resolve_dir, resolve_work_dir


def test_explicit_absolute_is_returned(tmp_path):
    assert resolve_dir(str(tmp_path)) == tmp_path


def test_relative_value_resolves_against_base(tmp_path):
    (tmp_path / "sdk").mkdir()
    assert resolve_dir("sdk", base=tmp_path) == tmp_path / "sdk"


def test_relative_value_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sdk").mkdir()
    assert resolve_dir("sdk") == tmp_path / "sdk"


def test_envvar_is_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("XUAN_TEST_DIR", str(tmp_path))
    assert resolve_dir(None, envvar="XUAN_TEST_DIR") == tmp_path


def test_explicit_value_wins_over_envvar(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("XUAN_TEST_DIR", str(tmp_path))
    assert resolve_dir(str(other), envvar="XUAN_TEST_DIR") == other


def test_missing_value_reports_envvar(monkeypatch):
    monkeypatch.delenv("XUAN_TEST_DIR", raising=False)
    with pytest.raises(XuanCmdError, match="环境变量 XUAN_TEST_DIR"):
        resolve_dir(None, envvar="XUAN_TEST_DIR", label="SDK 根目录")


def test_missing_value_without_envvar():
    with pytest.raises(XuanCmdError, match="未提供目录"):
        resolve_dir(None)


def test_nonexistent_dir_raises(tmp_path):
    with pytest.raises(XuanCmdError, match="SDK 目录不存在"):
        resolve_dir(str(tmp_path / "absent"), label="SDK 目录")


def test_nonexistent_dir_allowed_when_not_required(tmp_path):
    target = tmp_path / "absent"
    assert resolve_dir(str(target), require_exists=False) == target


def test_file_is_not_a_dir(tmp_path):
    file_path = tmp_path / "afile"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(XuanCmdError, match="不存在"):
        resolve_dir(str(file_path))


@pytest.mark.skipif(os.name == "nt", reason="~ 展开依赖 POSIX HOME")
def test_tilde_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "sdk").mkdir()
    assert resolve_dir("~/sdk") == tmp_path / "sdk"


def test_work_dir_default_returned_as_is(tmp_path):
    assert resolve_work_dir(None, tmp_path / "build") == tmp_path / "build"


def test_work_dir_explicit_relative_uses_base(tmp_path):
    assert resolve_work_dir("out", tmp_path / "build", base=tmp_path) == tmp_path / "out"


def test_work_dir_explicit_absolute_kept(tmp_path):
    target = tmp_path / "out"
    assert resolve_work_dir(str(target), Path("build"), base=tmp_path / "other") == target
