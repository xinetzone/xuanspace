# SPDX-License-Identifier: GPL-2.0-only
"""runner 子进程边界补测（T4 登记其为 I/O 边界、上游无单测；本文件用
fake subprocess 全覆盖 output/run/exec/ls/existing_containers 的参数
拼装与错误分支——不启动任何真实 podman，仅验证翻译层到 argv 的契约）。"""

import asyncio
import io
import os
from contextlib import redirect_stdout
from unittest import mock

import pytest

from xuan_compose.runner import (
    CalledProcessError,
    ExistingContainer,
    Podman,
    wait_with_timeout,
)


class _FakeProcess:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"",
                 wait_side: object = None) -> None:
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._wait_side = wait_side
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.returncode = self._returncode
        return self._stdout, self._stderr

    async def wait(self) -> int:
        if self._wait_side is not None:
            side = self._wait_side
            self._wait_side = None
            if isinstance(side, BaseException):
                raise side
            return side  # type: ignore[return-value]
        self.returncode = self._returncode
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _FakeReader:
    """最小 StreamReader：按预置 chunk 序列吐数据，支持异常注入。"""

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks
        self._i = 0

    def at_eof(self) -> bool:
        return self._i >= len(self._chunks)

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        chunk = self._chunks[self._i]
        self._i += 1
        assert isinstance(chunk, bytes)
        return chunk

    async def read(self, n: int = -1) -> bytes:
        chunk = self._chunks[self._i]
        self._i += 1
        assert isinstance(chunk, bytes)
        return chunk


def _podman(monkeypatch, process_factory):
    compose = mock.Mock()
    compose.project_name = "proj"
    compose.get_podman_args.side_effect = lambda cmd: [f"--{cmd}-x"]
    factory = mock.AsyncMock(side_effect=process_factory)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    return Podman(compose), factory


class TestWaitWithTimeout:
    def test_success(self) -> None:
        assert asyncio.run(wait_with_timeout(_coro_value(7), 1)) == 7

    def test_timeout_reraised(self) -> None:
        with pytest.raises(TimeoutError):
            asyncio.run(wait_with_timeout(asyncio.sleep(10), 0.01))


async def _coro_value(value: object) -> object:
    return value


class TestOutput:
    def test_success_returns_stdout_and_builds_argv(self, monkeypatch) -> None:
        podman, factory = _podman(
            monkeypatch,
            lambda *a, **kw: _FakeProcess(0, stdout=b"line\n"),
        )
        out = asyncio.run(podman.output(["--version"], "ps", ["-q"]))
        assert out == b"line\n"
        argv = list(factory.await_args.args)
        assert argv[0] == "podman"
        assert argv[1] == "--version"
        assert argv[-2:] == ["--ps-x", "-q"]

    def test_empty_cmd_skips_compose_args(self, monkeypatch) -> None:
        podman, factory = _podman(monkeypatch, lambda *a, **kw: _FakeProcess(0))
        asyncio.run(podman.output(["--version"]))
        assert list(factory.await_args.args) == ["podman", "--version"]

    def test_nonzero_raises_with_stderr(self, monkeypatch) -> None:
        podman, _ = _podman(
            monkeypatch, lambda *a, **kw: _FakeProcess(2, stderr=b"boom")
        )
        with pytest.raises(CalledProcessError) as ctx:
            asyncio.run(podman.output([], "ps"))
        assert ctx.value.returncode == 2
        assert ctx.value.output == b"boom"


class TestRun:
    def test_dry_run_skips_subprocess(self, monkeypatch) -> None:
        podman, factory = _podman(monkeypatch, lambda *a, **kw: _FakeProcess(0))
        podman.dry_run = True
        assert asyncio.run(podman.run(["up"], "up")) is None
        factory.assert_not_awaited()

    def test_plain_run_returns_exit_code(self, monkeypatch) -> None:
        podman, factory = _podman(monkeypatch, lambda *a, **kw: _FakeProcess(0))
        code = asyncio.run(podman.run(["start", "svc"], "start", ["svc"]))
        assert code == 0
        assert factory.await_args.kwargs["close_fds"] is False

    def test_suppress_output_redirects_to_devnull(self, monkeypatch) -> None:
        podman, factory = _podman(monkeypatch, lambda *a, **kw: _FakeProcess(0))
        asyncio.run(podman.run(["x"], suppress_output=True))
        kwargs = factory.await_args.kwargs
        assert kwargs["stdout"] is asyncio.subprocess.DEVNULL
        assert kwargs["stderr"] is asyncio.subprocess.DEVNULL

    def test_log_formatter_pipes_through_tasks(self, monkeypatch) -> None:
        proc = _FakeProcess(0)
        proc.stdout = _FakeReader([b"hello\n"])
        proc.stderr = _FakeReader([b""])
        podman, _ = _podman(monkeypatch, lambda *a, **kw: proc)
        tasks: set[asyncio.Task] = set()
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = asyncio.run(
                podman.run(["logs"], "logs", log_formatter="web |", task_reference=tasks)
            )
            # 让两个 _format_stream task 跑完
            asyncio.run(_drain(tasks))
        assert code == 0
        assert "web | hello" in buf.getvalue()

    def test_cancelled_terminates_and_waits(self, monkeypatch) -> None:
        proc = _FakeProcess(wait_side=asyncio.CancelledError())
        proc._returncode = 137
        podman, _ = _podman(monkeypatch, lambda *a, **kw: proc)
        code = asyncio.run(podman.run(["up"], "up"))
        assert proc.terminated is True
        assert proc.killed is False
        assert code == 137

    def test_cancelled_then_timeout_kills(self, monkeypatch) -> None:
        # wait 调用链：① 取消 → terminate；② terminate 后等待仍超时（经
        # wait_with_timeout 透传 TimeoutError）→ kill；③ kill 后返回 9
        chain = iter([asyncio.CancelledError(), TimeoutError(), 9])

        async def wait_chain() -> int:
            nxt = next(chain)
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

        proc = _FakeProcess()
        proc.wait = wait_chain  # type: ignore[method-assign]
        podman, _ = _podman(monkeypatch, lambda *a, **kw: proc)
        code = asyncio.run(podman.run(["up"], "up"))
        assert proc.terminated is True
        assert proc.killed is True
        assert code == 9


async def _drain(tasks: set[asyncio.Task]) -> None:
    for _ in range(10):
        if not tasks:
            break
        await asyncio.gather(*list(tasks), return_exceptions=True)
        await asyncio.sleep(0)


class TestExec:
    def test_exec_replaces_process_with_full_argv(self, monkeypatch) -> None:
        podman, _ = _podman(monkeypatch, lambda *a, **kw: _FakeProcess(0))
        execlp = mock.Mock()
        monkeypatch.setattr(os, "execlp", execlp)
        podman.exec(["run"], "run", ["svc", 1])
        args = execlp.call_args.args
        assert args[0] == "podman"
        assert args[1] == "podman"
        assert list(args[2:]) == ["run", "--run-x", "svc", "1"]


class TestListingHelpers:
    def test_network_ls(self, monkeypatch) -> None:
        podman, _ = _podman(
            monkeypatch, lambda *a, **kw: _FakeProcess(0, stdout=b"a\nb\n")
        )
        assert asyncio.run(podman.network_ls()) == ["a", "b"]

    def test_volume_ls(self, monkeypatch) -> None:
        podman, _ = _podman(
            monkeypatch, lambda *a, **kw: _FakeProcess(0, stdout=b"v1\n")
        )
        assert asyncio.run(podman.volume_ls()) == ["v1"]

    def test_existing_containers_parses_json_rows(self, monkeypatch) -> None:
        payload = (
            b'[{"Names": ["proj_web_1"], "Id": "abc", "ImageID": "img1",'
            b' "Exited": true, "State": "exited", "Status": "Exited (0)",'
            b' "Labels": {"io.podman.compose.service": "web",'
            b' "io.podman.compose.config-hash": "h1"}}]'
        )
        podman, _ = _podman(monkeypatch, lambda *a, **kw: _FakeProcess(0, stdout=payload))
        result = asyncio.run(podman.existing_containers("proj"))
        cnt = result["proj_web_1"]
        assert isinstance(cnt, ExistingContainer)
        assert cnt.service_name == "web"
        assert cnt.config_hash == "h1"
        assert cnt.exited is True
        assert cnt.state == "exited"
