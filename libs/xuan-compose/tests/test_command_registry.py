# SPDX-License-Identifier: GPL-2.0-only
"""TR-7.1：命令层显式注册表与上游 ``@cmd_run`` 清单集合相等。

上游 podman_compose.py 第 3324-4902 行共 24 处 ``@cmd_run(...)`` 装饰器
（spec F-005 文字作"22 个"为计数笔误，以代码实证为准，T7 完成记录登记）。
"""

import inspect
from unittest import IsolatedAsyncioTestCase, mock

from xuan_compose.commands import COMMAND_HANDLERS, install_handlers

# 与上游 24 个 @cmd_run 注册名一一对应的硬快照（名字 → 上游函数名）。
UPSTREAM_CMD_RUN_NAMES = {
    "ls",
    "version",
    "wait",
    "systemd",
    "pull",
    "push",
    "build",
    "up",
    "down",
    "ps",
    "run",
    "cp",
    "exec",
    "start",
    "stop",
    "restart",
    "logs",
    "config",
    "port",
    "pause",
    "unpause",
    "kill",
    "stats",
    "images",
}


class TestCommandRegistry(IsolatedAsyncioTestCase):
    def test_registry_matches_upstream_cmd_run_set(self) -> None:
        assert set(COMMAND_HANDLERS.keys()) == UPSTREAM_CMD_RUN_NAMES
        assert len(COMMAND_HANDLERS) == 24

    def test_every_handler_is_async_callable(self) -> None:
        for name, handler in COMMAND_HANDLERS.items():
            assert callable(handler), name
            assert inspect.iscoroutinefunction(handler), name

    def test_every_handler_has_two_positional_params(self) -> None:
        for name, handler in COMMAND_HANDLERS.items():
            sig = inspect.signature(handler)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            assert len(params) == 2, (name, sig)

    async def test_install_handlers_populates_command_lookup(self) -> None:
        compose = mock.Mock()
        compose.commands = {}
        install_handlers(compose)
        assert compose.commands.keys() == COMMAND_HANDLERS.keys()
        # 查表语义与上游 compose.commands["up"] 一致
        assert compose.commands["up"] is COMMAND_HANDLERS["up"]

    async def test_install_is_idempotent_update(self) -> None:
        compose = mock.Mock()
        compose.commands = {}
        install_handlers(compose)
        sentinel = object()
        compose.commands["custom"] = sentinel  # type: ignore[assignment]
        install_handlers(compose)
        assert compose.commands["custom"] is sentinel
        assert len(compose.commands) == 25
