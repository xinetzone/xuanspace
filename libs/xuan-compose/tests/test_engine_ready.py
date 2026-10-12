# SPDX-License-Identifier: GPL-2.0-only
"""引擎就绪保障（podman machine 自愈）测试。

覆盖 xc 在命令分发前的服务端可达性保障契约：
- ``is_podman_connection_error``：exit 125 必须同时命中连接错误文案才判定
  （125 同时是容器运行时通用错误码，单靠退出码会误伤 ``podman run`` 失败）；
- ``Podman.ping``：``podman info`` 真实往返，区别于纯本地的 ``--version``；
- ``Podman.list_machines``：``machine list --format json`` 的解析与降级；
- ``Podman.ensure_ready``：已就绪零副作用 / 默认机停止时自动 start 并轮询 /
  无 machine、默认机已运行、start 失败、就绪超时等不可自愈分支抛出
  :class:`PodmanEngineNotReadyError`。

全部用 scripted fake subprocess，零真实 podman、零真实时钟等待。
"""

import asyncio
import contextlib
import io
import subprocess
from unittest import mock

import pytest

from xuan_compose.errors import PodmanEngineNotReadyError
from xuan_compose.runner import (
    Podman,
    is_podman_connection_error,
)

_CONN_REFUSED_STDERR = (
    b"Cannot connect to Podman. Please verify your connection to the Linux "
    b"system using `podman system connection list`\n"
    b"Error: unable to connect to Podman socket: failed to connect: "
    b"dial tcp 127.0.0.1:60231: connectex: No connection could be made "
    b"because the target machine actively refused it."
)
_MACHINE_JSON = (
    b'[{"Name": "podman-machine-default", "Default": true, "Running": false,'
    b' "Starting": false, "VMType": "wsl", "Port": 60231}]'
)


class _ScriptedProcess:
    """按预置 (returncode, stdout, stderr) 响应一次 create_subprocess_exec。"""

    def __init__(self, script: list[tuple[int, bytes, bytes]]) -> None:
        self._script = script
        self._i = 0
        self.calls: list[list[str]] = []
        self.returncode = 0

    async def __call__(self, *argv: str, **_kwargs: object) -> _ScriptedProcess:
        self.calls.append([str(a) for a in argv])
        return self

    async def communicate(self) -> tuple[bytes, bytes]:
        i = min(self._i, len(self._script) - 1)
        rc, out, err = self._script[i]
        self._i += 1
        self.returncode = rc
        return out, err


def _podman(script: list[tuple[int, bytes, bytes]], monkeypatch: pytest.MonkeyPatch):
    proc = _ScriptedProcess(script)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", proc)
    compose = mock.Mock()
    compose.get_podman_args.return_value = []
    return Podman(compose), proc


def _ok(stdout: bytes = b"amd64\n") -> tuple[int, bytes, bytes]:
    return (0, stdout, b"")


def _conn_refused() -> tuple[int, bytes, bytes]:
    return (125, b"", _CONN_REFUSED_STDERR)


class TestConnectionErrorClassification:
    def test_exit_125_with_cannot_connect_marker(self) -> None:
        assert is_podman_connection_error(125, _CONN_REFUSED_STDERR) is True

    def test_exit_125_with_socket_marker_only(self) -> None:
        assert is_podman_connection_error(125, b"unable to connect to Podman socket") is True

    def test_exit_125_with_unrelated_stderr_is_not_connection_error(self) -> None:
        # 125 同时是容器运行时通用错误码（如 run 失败），无连接文案不得误判
        assert is_podman_connection_error(125, b"Error: container exited with status 42") is False

    def test_zero_and_other_codes_never_match(self) -> None:
        assert is_podman_connection_error(0, _CONN_REFUSED_STDERR) is False
        assert is_podman_connection_error(126, _CONN_REFUSED_STDERR) is False
        assert is_podman_connection_error(1, _CONN_REFUSED_STDERR) is False


class TestPing:
    def test_info_success_means_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        podman, proc = _podman([_ok()], monkeypatch)
        assert asyncio.run(podman.ping()) is True
        assert proc.calls[0][:3] == ["podman", "info", "--format"]

    def test_connection_refused_means_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        podman, _ = _podman([_conn_refused()], monkeypatch)
        assert asyncio.run(podman.ping()) is False

    def test_missing_binary_means_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _raise(*_a: object, **_kw: object) -> object:
            raise FileNotFoundError("podman")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise)
        podman = Podman(mock.Mock())
        assert asyncio.run(podman.ping()) is False


class TestListMachines:
    def test_parses_machine_json_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        podman, _ = _podman([_ok(_MACHINE_JSON)], monkeypatch)
        machines = asyncio.run(podman.list_machines())
        assert len(machines) == 1
        assert machines[0]["Name"] == "podman-machine-default"
        assert machines[0]["Default"] is True

    def test_unparseable_json_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        podman, _ = _podman([_ok(b"not-json\n")], monkeypatch)
        assert asyncio.run(podman.list_machines()) == []

    def test_command_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        podman, _ = _podman([(125, b"", b"boom")], monkeypatch)
        assert asyncio.run(podman.list_machines()) == []


class TestEnsureReady:
    def test_already_ready_has_no_machine_side_effects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        podman, proc = _podman([_ok()], monkeypatch)
        asyncio.run(podman.ensure_ready(timeout=1.0))
        # 仅一次 info 探活，不触碰 machine 子命令
        assert len(proc.calls) == 1
        assert "machine" not in proc.calls[0]

    def test_stopped_default_machine_gets_started_and_waited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ① info 拒绝 → ② machine list 报默认机停止 → ③ start 成功
        # → ④ 轮询 info 成功
        podman, proc = _podman(
            [_conn_refused(), _ok(_MACHINE_JSON), _ok(b""), _ok()],
            monkeypatch,
        )
        sleeps: list[float] = []

        async def _instant_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            asyncio.run(podman.ensure_ready(timeout=30.0))

        flat = [" ".join(c) for c in proc.calls]
        assert any(c == "podman machine list --format json" for c in flat)
        assert any(c == "podman machine start podman-machine-default" for c in flat)
        assert "starting podman machine" in buf.getvalue()
        assert "ready" in buf.getvalue()

    def test_no_machine_configured_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        podman, _ = _podman([_conn_refused(), _ok(b"[]")], monkeypatch)
        with pytest.raises(PodmanEngineNotReadyError) as ctx:
            asyncio.run(podman.ensure_ready(timeout=1.0))
        assert "podman machine start" in str(ctx.value)

    def test_running_machine_still_unreachable_is_not_restarted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        running = _MACHINE_JSON.replace(b'"Running": false', b'"Running": true')
        podman, proc = _podman([_conn_refused(), _ok(running)], monkeypatch)
        with pytest.raises(PodmanEngineNotReadyError):
            asyncio.run(podman.ensure_ready(timeout=0.0))
        flat = [" ".join(c) for c in proc.calls]
        assert not any("machine start" in c for c in flat)

    def test_starting_machine_is_waited_without_second_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        starting = _MACHINE_JSON.replace(b'"Starting": false', b'"Starting": true')
        # ① 探活失败 → ② list 报 Starting → ③ 轮询 info 成功
        podman, proc = _podman([_conn_refused(), _ok(starting), _ok()], monkeypatch)
        monkeypatch.setattr(asyncio, "sleep", mock.AsyncMock(return_value=None))
        asyncio.run(podman.ensure_ready(timeout=30.0))
        flat = [" ".join(c) for c in proc.calls]
        assert not any("machine start" in c for c in flat)

    def test_non_default_machine_is_not_auto_started(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = _MACHINE_JSON.replace(b'"Default": true', b'"Default": false')
        podman, proc = _podman([_conn_refused(), _ok(other)], monkeypatch)
        with pytest.raises(PodmanEngineNotReadyError):
            asyncio.run(podman.ensure_ready(timeout=0.0))
        flat = [" ".join(c) for c in proc.calls]
        assert not any("machine start" in c for c in flat)

    def test_start_failure_is_wrapped_with_stderr_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        podman, _ = _podman(
            [
                _conn_refused(),
                _ok(_MACHINE_JSON),
                (125, b"", b'Error: failed to start command wsl: exec: "wsl": not found'),
            ],
            monkeypatch,
        )
        with pytest.raises(PodmanEngineNotReadyError) as ctx:
            asyncio.run(podman.ensure_ready(timeout=1.0))
        assert "podman-machine-default" in str(ctx.value)
        assert "wsl" in str(ctx.value)

    def test_ready_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # start 成功后 info 始终拒绝；timeout=0 使轮询体不进入，立即超时
        podman, proc = _podman(
            [_conn_refused(), _ok(_MACHINE_JSON), _ok(b""), _conn_refused()],
            monkeypatch,
        )
        monkeypatch.setattr(asyncio, "sleep", mock.AsyncMock(return_value=None))
        with pytest.raises(PodmanEngineNotReadyError, match="Timed out"):
            asyncio.run(podman.ensure_ready(timeout=0.0))
        flat = [" ".join(c) for c in proc.calls]
        assert any(c == "podman machine start podman-machine-default" for c in flat)

    def test_corrupt_machine_list_payload_is_treated_as_no_machine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        podman, _ = _podman([_conn_refused(), _ok(b"{broken")], monkeypatch)
        with pytest.raises(PodmanEngineNotReadyError):
            asyncio.run(podman.ensure_ready(timeout=0.0))

    def test_start_uses_same_podman_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        podman, proc = _podman(
            [_conn_refused(), _ok(_MACHINE_JSON), _ok(b""), _ok()],
            monkeypatch,
        )
        podman.podman_path = "/custom/bin/podman"
        asyncio.run(podman.ensure_ready(timeout=1.0))
        assert all(call[0] == "/custom/bin/podman" for call in proc.calls)


class TestOutputConnectionErrors:
    """业务调用出口（TOCTOU 与 --no-machine-start 路径）同样须干净收口。

    ``Podman.output`` 是包内唯一子进程出口（TR-4.2）：即使装配期
    ``ensure_ready`` 被跳过或探活后 VM 才停止，业务调用以 125 + 连接文案
    失败时也必须抛 :class:`PodmanEngineNotReadyError`（经 main 收口为
    单行 ``Error:`` + exit 1），而非裸 ``CalledProcessError`` traceback。
    """

    def test_connection_refused_wrapped_as_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        podman, _ = _podman([_conn_refused()], monkeypatch)
        with pytest.raises(PodmanEngineNotReadyError) as ctx:
            asyncio.run(podman.output(["ps", "--format", "json"]))
        message = str(ctx.value)
        assert "podman ps" in message
        assert "Cannot connect to Podman" in message
        assert "podman machine start" in message

    def test_unrelated_exit_125_keeps_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 容器运行时失败也以 125 退出，无连接文案不得误判为引擎未就绪
        podman, _ = _podman(
            [(125, b"", b"Error: container_linux.go: bad OCI spec")],
            monkeypatch,
        )
        with pytest.raises(subprocess.CalledProcessError):
            asyncio.run(podman.output(["run", "img"]))

    def test_other_failure_codes_keep_called_process_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        podman, _ = _podman([(1, b"", b"boom")], monkeypatch)
        with pytest.raises(subprocess.CalledProcessError):
            asyncio.run(podman.output(["inspect", "x"]))
