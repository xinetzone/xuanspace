# SPDX-License-Identifier: GPL-2.0-only
"""systemd 命令补测（T7 已覆盖 ls 与 create-unit 不可写打印；这里补
register 落盘与环境变量过滤、unregister 存在/不存在两分支、
create-unit 可写时真实落盘——全部 HOME 重定向到 tmp，不碰真实 ~）。"""

import asyncio
import io
import os
from contextlib import redirect_stdout
from unittest import mock
from unittest.mock import mock_open

import pytest

from xuan_compose.commands.systemd import compose_systemd


def _compose() -> mock.Mock:
    compose = mock.Mock()
    compose.project_name = "proj"
    compose.executable = "/opt/bin/podman-compose"
    compose.environ = {
        "COMPOSE_FILE": "compose.yaml",
        "PODMAN_FOO": "bar",
        "OTHER_VAR": "ignored",
    }
    return compose


def _env_path(tmp_path: str) -> str:
    return os.path.join(
        tmp_path, ".config", "containers", "compose", "projects", "proj.env"
    )


@pytest.fixture(autouse=False)
def _fake_home(tmp_path, monkeypatch):
    # 直接 patch expanduser：Windows 不读 HOME 环境变量（读 USERPROFILE），
    # patch 函数可保证两平台都落到 tmp_path
    def _expanduser(path: str) -> str:
        rel = path[2:] if path.startswith("~") else path
        return os.path.join(str(tmp_path), *rel.split("/"))

    monkeypatch.setattr(
        "xuan_compose.commands.systemd.os.path.expanduser", _expanduser
    )
    return tmp_path


def test_register_writes_filtered_env_file(_fake_home, tmp_path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(compose_systemd(_compose(), _ns("register")))
    fn = _env_path(str(tmp_path))
    assert os.path.exists(fn)
    content = open(fn, encoding="utf-8").read()
    assert "COMPOSE_FILE=compose.yaml\n" in content
    assert "PODMAN_FOO=bar\n" in content
    assert "OTHER_VAR" not in content
    assert "podman-compose@proj" in buf.getvalue()


def test_unregister_existing_file(_fake_home, tmp_path) -> None:
    # 先注册再注销
    asyncio.run(compose_systemd(_compose(), _ns("register")))
    fn = _env_path(str(tmp_path))
    assert os.path.exists(fn)
    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(compose_systemd(_compose(), _ns("unregister")))
    assert not os.path.exists(fn)
    assert "successfully unregistered" in buf.getvalue()


def test_unregister_missing_file_is_warning(_fake_home) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(compose_systemd(_compose(), _ns("unregister")))
    assert "is not registered" in buf.getvalue()


def test_create_unit_writes_to_etc_when_writable() -> None:
    # 可写分支（root 安装场景）：mock os.access 放行 + mock_open 捕获写入
    compose = _compose()
    writer = mock_open()
    buf = io.StringIO()
    with (
        mock.patch("xuan_compose.commands.systemd.os.access", return_value=True),
        mock.patch("xuan_compose.commands.systemd.open", writer),
        redirect_stdout(buf),
    ):
        asyncio.run(compose_systemd(compose, _ns("create-unit")))
    handle = writer()
    written = "".join(call.args[0] for call in handle.write.call_args_list)
    assert "/etc/systemd/user/podman-compose@.service" in written
    assert "ExecStart=/opt/bin/podman-compose wait" in written
    assert "EnvironmentFile=%h/.config/containers/compose/projects/%i.env" in written
    assert "while in your project" in buf.getvalue()


def _ns(action: str):
    from argparse import Namespace

    return Namespace(action=action)
