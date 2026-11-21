# SPDX-License-Identifier: GPL-2.0-only
"""TR-7.1：命令清单与上游 ``@cmd_run`` 清单集合相等（P2-2 登记化）。

上游 podman_compose.py 第 3324-4902 行共 24 处 ``@cmd_run(...)`` 装饰器
（spec F-005 文字作"22 个"为计数笔误，以代码实证为准，T7 完成记录登记）。

P2-2 起，handler/parser/help 三张静态表收敛为
``command_specs.COMMAND_SPECS`` 单一清单；本文件同时验证登记返回
``Disposable`` 的撤销语义（登记即副作用、撤销有证据）。
"""

import inspect
from unittest import IsolatedAsyncioTestCase, mock

from xuan_compose.cli.command_specs import COMMAND_HANDLERS, build_registry
from xuan_compose.commands import CommandSpec, install_handlers

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
            params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
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


class TestReversibleRegistration(IsolatedAsyncioTestCase):
    """P2-2：登记返回 ``Disposable``，撤销即回滚（不依赖进程重启）。"""

    def test_register_returns_undo_that_removes_only_that_entry(self) -> None:
        registry = build_registry()
        before = len(registry)
        undo = registry.register(CommandSpec("probe", COMMAND_HANDLERS["up"]))
        assert "probe" in registry
        assert len(registry) == before + 1
        undo()
        assert "probe" not in registry
        assert len(registry) == before

    def test_duplicate_registration_is_contract_conflict(self) -> None:
        registry = build_registry()
        with self.assertRaises(ValueError):
            registry.register(CommandSpec("ls", COMMAND_HANDLERS["ls"]))
        assert len(registry) == len(COMMAND_HANDLERS)

    def test_register_all_rolls_back_on_conflict(self) -> None:
        registry = build_registry()
        before = len(registry)
        with self.assertRaises(ValueError):
            registry.register_all(
                (
                    CommandSpec("probe", COMMAND_HANDLERS["up"]),
                    CommandSpec("ls", COMMAND_HANDLERS["ls"]),
                )
            )
        assert "probe" not in registry
        assert len(registry) == before

    async def test_bind_disposable_restores_previous_lookup(self) -> None:
        compose = mock.Mock()
        sentinel = object()
        compose.commands = {"up": sentinel}
        undo = install_handlers(compose)
        assert compose.commands["up"] is COMMAND_HANDLERS["up"]
        assert compose.commands["ls"] is COMMAND_HANDLERS["ls"]
        undo()
        assert compose.commands == {"up": sentinel}
