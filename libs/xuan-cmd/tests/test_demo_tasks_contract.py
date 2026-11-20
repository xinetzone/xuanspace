"""与 xmapps 交付包 `apps/demo/tasks/common.py` 的语义对齐。

契约权威在交付包侧（xmapps `apps/.agents/rules/app.md` 第 11 条）。两侧不共享
运行时依赖，因此以本文件锁定调用形态：参数校验、环境变量回退、错误可读性、
默认构建目录。修改这些行为前先确认交付包侧规则。
"""

import os
import shlex
import subprocess

import pytest

from xuan_cmd import (
    XuanCmdError,
    format_command,
    list_projects,
    locate_project,
    quote_arg,
    resolve_dir,
    resolve_work_dir,
)


def _expected_quote(part):
    """common.quote 的等价实现，用于逐字符比对。"""
    return subprocess.list2cmdline([str(part)]) if os.name == "nt" else shlex.quote(str(part))


def test_quote_arg_matches_common_quote(tmp_path):
    parts = ["cmake", "-S", str(tmp_path / "a b"), "-DXM_SDK_ROOT=/opt/sdk"]
    assert [quote_arg(part) for part in parts] == [_expected_quote(part) for part in parts]


def test_format_command_matches_common_run(tmp_path):
    parts = ["cmake", "--build", str(tmp_path / "out dir")]
    assert format_command(parts) == " ".join(_expected_quote(part) for part in parts)


def test_format_command_windows_branch(monkeypatch):
    """CI 仅在 Linux 运行，按 ci.yml 约定用 monkeypatch 覆盖 Windows 分支。"""
    monkeypatch.setattr(os, "name", "nt")
    parts = ["cmake", "-S", "a b", "-DX=1"]
    assert format_command(parts) == " ".join(subprocess.list2cmdline([part]) for part in parts)


def test_sdk_root_envvar_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("XM_SDK_ROOT", str(tmp_path))
    assert resolve_dir(None, envvar="XM_SDK_ROOT", label="SDK 根目录") == tmp_path


def test_sdk_root_missing_mentions_envvar(monkeypatch):
    monkeypatch.delenv("XM_SDK_ROOT", raising=False)
    with pytest.raises(XuanCmdError, match="XM_SDK_ROOT"):
        resolve_dir(None, envvar="XM_SDK_ROOT", label="SDK 根目录")


def test_sdk_root_nonexistent_mentions_label(tmp_path):
    with pytest.raises(XuanCmdError, match="SDK 根目录不存在"):
        resolve_dir(str(tmp_path / "absent"), label="SDK 根目录")


def test_locate_app_by_cmakelists(tmp_path):
    app = tmp_path / "cpp_deploy"
    app.mkdir()
    (app / "CMakeLists.txt").write_text("", encoding="utf-8")
    assert locate_project(tmp_path, "cpp_deploy", label="应用") == app


def test_locate_app_missing_reports_name(tmp_path):
    with pytest.raises(XuanCmdError, match="未找到应用：ghost"):
        locate_project(tmp_path, "ghost", label="应用")


def test_list_apps_is_name_sorted(tmp_path):
    for name in ("b_app", "a_app"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "CMakeLists.txt").write_text("", encoding="utf-8")
    assert [path.name for path in list_projects(tmp_path)] == ["a_app", "b_app"]


def test_build_dir_default_is_app_build(tmp_path):
    app = tmp_path / "app"
    assert resolve_work_dir(None, app / "build") == app / "build"


def test_build_dir_explicit_relative_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_work_dir("out", tmp_path / "build") == tmp_path / "out"
