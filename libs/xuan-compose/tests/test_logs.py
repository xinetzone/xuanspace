# SPDX-License-Identifier: GPL-2.0-only
"""日志编排测试（T6 新增，覆盖 create_format_logs_task 三个前缀分支与
_task_cancelled 判定；上游无对应单测，本层按真实分支补齐）。"""

import argparse
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock

from xuan_compose.logs import _task_cancelled, create_format_logs_task


class _FakePodman:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Any], str, list[Any], Any]] = []

    async def run(self, podman_args: list[Any], cmd: str, args: list[Any], **kw: Any) -> int:
        self.calls.append((podman_args, cmd, args, kw.get("log_formatter")))
        return 0


def _make_engine() -> Any:
    engine = mock.Mock()
    engine.containers = [{"_service": "web", "log_prefix": "web_1"}]
    engine.container_names_by_service = {"web": ["web_1"]}
    engine.console_colors = ["\x1b[1;32m", "\x1b[1;33m"]
    engine.podman = _FakePodman()
    return engine


class TestCreateFormatLogsTask(IsolatedAsyncioTestCase):
    async def test_returns_none_for_unknown_service(self) -> None:
        engine = _make_engine()
        args = argparse.Namespace(no_log_prefix=False, no_color=False)
        assert create_format_logs_task(engine, args, "missing", ["-f"], 16) is None
        assert engine.podman.calls == []

    async def test_colored_prefix_uses_service_color(self) -> None:
        engine = _make_engine()
        args = argparse.Namespace(no_log_prefix=False, no_color=False)
        task = create_format_logs_task(engine, args, "web", ["-f"], max_service_length=5)
        assert task is not None
        await task

        space_suffix = " " * (5 - len("web_1") + 1)
        expected_formatter = f"\x1b[1;32m[web_1]{space_suffix}|\x1b[0m"
        assert engine.podman.calls == [
            ([], "logs", ["-f", "web_1"], expected_formatter)
        ]

    async def test_no_color_uses_monochrome_reset(self) -> None:
        engine = _make_engine()
        args = argparse.Namespace(no_log_prefix=False, no_color=True)
        task = create_format_logs_task(engine, args, "web", [], max_service_length=5)
        assert task is not None
        await task

        space_suffix = " " * (5 - len("web_1") + 1)
        expected_formatter = f"\x1b[0m[web_1]{space_suffix}|\x1b[0m"
        assert engine.podman.calls[0][3] == expected_formatter

    async def test_no_log_prefix_disables_formatter(self) -> None:
        engine = _make_engine()
        args = argparse.Namespace(no_log_prefix=True, no_color=False)
        task = create_format_logs_task(engine, args, "web", ["--tail=10"], max_service_length=5)
        assert task is not None
        await task

        assert engine.podman.calls == [
            ([], "logs", ["--tail=10", "web_1"], None)
        ]


class TestTaskCancelled(IsolatedAsyncioTestCase):
    async def test_running_task_not_cancelled(self) -> None:
        import asyncio

        async def noop() -> None:
            return None

        task = asyncio.create_task(noop())
        await asyncio.sleep(0)
        assert task.done()
        assert not task.cancelled()
        assert _task_cancelled(task) is False

    async def test_cancelled_task_detected(self) -> None:
        import asyncio

        async def hang() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(hang())
        task.cancel()
        assert _task_cancelled(task) is True
        try:
            await task
        except asyncio.CancelledError:
            pass
