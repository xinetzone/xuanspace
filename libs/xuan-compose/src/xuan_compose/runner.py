# SPDX-License-Identifier: GPL-2.0-only
"""执行层——podman 子进程边界（翻译自 podman_compose.py 第 1809-2077 行）。

本模块是包内**唯一**导入 ``subprocess`` / ``asyncio.subprocess`` 的模块（TR-4.2）：
``Podman.output/run/exec`` 是全部子进程调用的唯一出口；
``ExistingContainer`` 为 ``podman ps --format json`` 的逐行翻译值对象。

与上游的唯一结构性差异：``compose`` 形参标注为 ``Any``——分层后引擎层（T6）
不允许被本层反向导入，调用方按鸭子类型提供 ``get_podman_args()`` /
``project_name``，与上游运行时契约完全一致。
"""

import asyncio
import codecs
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from .errors import PodmanEngineNotReadyError
from .logging_utils import log

__all__ = [
    "CONNECTION_REFUSED_EXIT_CODE",
    "DEFAULT_MACHINE_READY_TIMEOUT",
    "CalledProcessError",
    "ExistingContainer",
    "Podman",
    "is_podman_connection_error",
    "wait_with_timeout",
]

# 子进程异常类型从本边界模块统一再导出，避免其他模块直接 import subprocess（TR-4.2）
CalledProcessError = subprocess.CalledProcessError

#: Podman 服务端不可达时的退出码（Windows/macOS 下 ssh 拨号被拒实测为 125）。
#: 注意 125 同时是容器运行时通用错误码，故判定必须叠加 stderr 特征串。
CONNECTION_REFUSED_EXIT_CODE = 125

#: 自动启动 machine 后等待服务端就绪的默认总预算（秒），可由
#: XC_MACHINE_READY_TIMEOUT 覆盖。
DEFAULT_MACHINE_READY_TIMEOUT = 120.0

#: 就绪轮询间隔（秒）。
READY_POLL_INTERVAL = 1.0

_CONNECTION_ERROR_MARKERS = (
    b"Cannot connect to Podman",
    b"unable to connect to Podman socket",
    b"failed to connect",
)


def is_podman_connection_error(returncode: int, stderr: bytes) -> bool:
    """判定一次 podman 调用失败是否为"服务端不可达"（区别于普通命令错误）。

    双条件：退出码为 125 **且** stderr 命中连接错误特征文案。单看 125 会
    误伤 ``podman run`` 等场景的容器运行时失败（同样以 125 退出）。
    """
    if returncode == 0:
        return False
    return returncode == CONNECTION_REFUSED_EXIT_CODE and any(marker in stderr for marker in _CONNECTION_ERROR_MARKERS)


async def wait_with_timeout(coro: Any, timeout: int | float) -> Any:
    """
    Asynchronously waits for the given coroutine to complete with a timeout.

    Args:
        coro: The coroutine to wait for.
        timeout (int or float): The maximum number of seconds to wait for.

    Raises:
        TimeoutError: If the coroutine does not complete within the specified timeout.
    """
    try:
        return await asyncio.wait_for(coro, timeout)
    except TimeoutError:
        # Python 3.11+ 起 asyncio.TimeoutError 即内置 TimeoutError（ruff UP041）
        raise


@dataclass
class ExistingContainer:
    name: str
    id: str
    service_name: str
    config_hash: str
    image_id: str
    exited: bool
    state: str
    status: str


class Podman:
    def __init__(
        self,
        compose: Any,
        podman_path: str = "podman",
        dry_run: bool = False,
        semaphore: asyncio.Semaphore = asyncio.Semaphore(sys.maxsize),
    ) -> None:
        self.compose = compose
        self.podman_path = podman_path
        self.dry_run = dry_run
        self.semaphore = semaphore

    async def output(self, podman_args: list[str], cmd: str = "", cmd_args: list[str] | None = None) -> bytes:
        async with self.semaphore:
            cmd_args = cmd_args or []
            xargs = self.compose.get_podman_args(cmd) if cmd else []
            cmd_ls = [self.podman_path, *podman_args] + xargs + cmd_args
            log.info(str(cmd_ls))
            p = await asyncio.create_subprocess_exec(
                *cmd_ls,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_data, stderr_data = await p.communicate()
            assert p.returncode is not None
            if p.returncode == 0:
                return stdout_data

            # 服务端不可达（装配期未自愈、--no-machine-start 跳过自愈或
            # 探活后 VM 才停止的 TOCTOU 窗口）在此唯一出口统一收口为
            # PodmanEngineNotReadyError，由 main() 以单行 Error + exit 1
            # 呈现；其余失败保持上游逐行对等的 CalledProcessError。
            if is_podman_connection_error(p.returncode, stderr_data):
                detail = _first_stderr_line(stderr_data)
                suffix = f": {detail}" if detail else ""
                raise PodmanEngineNotReadyError(
                    f"'{' '.join(cmd_ls)}' could not reach the Podman service{suffix}. "
                    "Inspect machines with `podman machine list` and start the default "
                    "one with `podman machine start`, then rerun xc."
                )

            raise subprocess.CalledProcessError(p.returncode, " ".join(cmd_ls), stderr_data)

    async def _readchunk(self, reader: asyncio.StreamReader) -> bytes:
        try:
            return await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as e:
            return e.partial
        except asyncio.LimitOverrunError as e:
            return await reader.read(e.consumed)

    async def _format_stream(self, reader: asyncio.StreamReader, sink: Any, log_formatter: str) -> None:
        line_ongoing = False
        decoder = codecs.getincrementaldecoder("utf-8")()

        def _formatted_print_with_nl(s: str) -> None:
            if line_ongoing:
                print(s, file=sink, end="\n")
            else:
                print(log_formatter, s, file=sink, end="\n")

        def _formatted_print_without_nl(s: str) -> None:
            if line_ongoing:
                print(s, file=sink, end="")
            else:
                print(log_formatter, s, file=sink, end="")

        while not reader.at_eof():
            chunk = await self._readchunk(reader)
            parts = chunk.split(b"\n")

            for i, part in enumerate(parts):
                # Iff part is last and non-empty, we leave an ongoing line to be completed later
                if i < len(parts) - 1:
                    _formatted_print_with_nl(decoder.decode(part))
                    line_ongoing = False
                elif len(part) > 0:
                    _formatted_print_without_nl(decoder.decode(part))
                    line_ongoing = True
                else:
                    # When we have an empty string as the last part:
                    # Do nothing if it's the only part (=the chunk is empty).
                    # If it's 2nd or later part, an empty new line will be redundant.
                    pass

        buf, _ = decoder.getstate()
        if len(buf) > 0:
            log.error("Incomplete multibyte character ignored in log output: %s", buf)
        if line_ongoing:
            # Make sure the last line ends with EOL
            print(file=sink, end="\n")

    def exec(
        self,
        podman_args: list[str],
        cmd: str = "",
        cmd_args: list[str] | None = None,
    ) -> None:
        cmd_args = list(map(str, cmd_args or []))
        xargs = self.compose.get_podman_args(cmd) if cmd else []
        cmd_ls = [self.podman_path, *podman_args] + xargs + cmd_args
        log.info(" ".join([str(i) for i in cmd_ls]))
        os.execlp(self.podman_path, *cmd_ls)

    async def run(  # pylint: disable=dangerous-default-value
        self,
        podman_args: list[str],
        cmd: str = "",
        cmd_args: list[str] | None = None,
        log_formatter: str | None = None,
        *,
        suppress_output: bool = False,
        pass_stdin: bool = False,
        # Intentionally mutable default argument to hold references to tasks
        task_reference: set[asyncio.Task] = set(),
    ) -> int | None:
        async with self.semaphore:
            cmd_args = list(map(str, cmd_args or []))
            xargs = self.compose.get_podman_args(cmd) if cmd else []
            cmd_ls = [self.podman_path, *podman_args] + xargs + cmd_args
            log.info(" ".join([str(i) for i in cmd_ls]))
            if self.dry_run:
                return None

            # When pass_stdin is False, use PIPE so the subprocess does not
            # inherit podman-compose's stdin. Using DEVNULL would send EOF
            # to containers with stdin_open, causing them to exit
            # immediately instead of waiting for input as docker-compose does.
            stdin_arg = None if pass_stdin else asyncio.subprocess.PIPE

            if log_formatter is not None:
                p = await asyncio.create_subprocess_exec(
                    *cmd_ls,
                    stdin=stdin_arg,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    close_fds=False,
                )  # pylint: disable=consider-using-with

                assert p.stdout is not None
                assert p.stderr is not None

                # This is hacky to make the tasks not get garbage collected
                # https://github.com/python/cpython/issues/91887
                out_t = asyncio.create_task(self._format_stream(p.stdout, sys.stdout, log_formatter))
                task_reference.add(out_t)
                out_t.add_done_callback(task_reference.discard)

                err_t = asyncio.create_task(self._format_stream(p.stderr, sys.stdout, log_formatter))
                task_reference.add(err_t)
                err_t.add_done_callback(task_reference.discard)

            elif suppress_output:
                p = await asyncio.create_subprocess_exec(
                    *cmd_ls,
                    stdin=stdin_arg,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=False,
                )  # pylint: disable=consider-using-with

            else:
                p = await asyncio.create_subprocess_exec(*cmd_ls, stdin=stdin_arg, close_fds=False)  # pylint: disable=consider-using-with

            try:
                exit_code = await p.wait()
            except asyncio.CancelledError:
                log.info("Sending termination signal")
                p.terminate()
                try:
                    exit_code = await wait_with_timeout(p.wait(), 10)
                except TimeoutError:
                    log.warning("container did not shut down after 10 seconds, killing")
                    p.kill()
                    exit_code = await p.wait()

            log.info("exit code: %s", exit_code)
            return exit_code

    async def network_ls(self) -> list[str]:
        output = (
            await self.output(
                [],
                "network",
                [
                    "ls",
                    "--noheading",
                    "--filter",
                    f"label=io.podman.compose.project={self.compose.project_name}",
                    "--format",
                    "{{.Name}}",
                ],
            )
        ).decode()
        networks = output.splitlines()
        return networks

    async def volume_ls(self) -> list[str]:
        output = (
            await self.output(
                [],
                "volume",
                [
                    "ls",
                    "--noheading",
                    "--filter",
                    f"label=io.podman.compose.project={self.compose.project_name}",
                    "--format",
                    "{{.Name}}",
                ],
            )
        ).decode("utf-8")
        volumes = output.splitlines()
        return volumes

    async def existing_containers(self, project_name: str) -> dict[str, ExistingContainer]:
        output = await self.output(
            [],
            "ps",
            [
                "--filter",
                f"label=io.podman.compose.project={project_name}",
                "-a",
                "--format",
                "json",
            ],
        )

        containers = json.loads(output)
        return {
            c.get("Names")[0]: ExistingContainer(
                name=c.get("Names")[0],
                id=c.get("Id"),
                service_name=(
                    c.get("Labels", {}).get("io.podman.compose.service", "")
                    or c.get("Labels", {}).get("com.docker.compose.service", "")
                ),
                config_hash=c.get("Labels", {}).get("io.podman.compose.config-hash", ""),
                image_id=c.get("ImageID", ""),
                exited=c.get("Exited", False),
                state=c.get("State", ""),
                status=c.get("Status", ""),
            )
            for c in containers
        }

    async def ping(self) -> bool:
        """探测 Podman 服务端是否真正可达。

        ``podman info`` 需要一次到服务端的真实往返；而启动期探测所用的
        ``podman --version`` 是纯本地客户端调用，machine 停止时照样成功，
        不能作为服务端就绪判据。任何失败（连接拒绝、二进制缺失）一律按
        不可达处理，由调用方决定是否触发 machine 自愈。
        """
        try:
            await self.output(["info", "--format", "{{.Host.Arch}}"])
        except CalledProcessError, PodmanEngineNotReadyError, FileNotFoundError:
            return False
        return True

    async def list_machines(self) -> list[dict[str, Any]]:
        """枚举本机 podman machine（``machine list --format json``）。

        命令缺失/失败/输出不可解析时返回空列表——调用方据此判定当前后端
        不支持 machine 自愈（如 Linux 原生 socket），不得猜测性启动。
        """
        try:
            raw = await self.output(["machine", "list", "--format", "json"])
        except CalledProcessError, PodmanEngineNotReadyError, FileNotFoundError:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    async def start_machine(self, name: str) -> None:
        """启动指定 podman machine（幂等性由调用方先查 Running/Starting 保证）。"""
        await self.output(["machine", "start", name])

    async def ensure_ready(self, *, timeout: float = DEFAULT_MACHINE_READY_TIMEOUT) -> None:
        """保障 Podman 服务端就绪，必要时自动启动默认 machine。

        编排序列：
        1. ``podman info`` 探活，已就绪则零副作用返回；
        2. 不可达时枚举 machine，仅选择 ``Default=true`` 者；无默认机则
           抛 :class:`PodmanEngineNotReadyError`（Linux 原生/多机配置不盲动）；
        3. 默认机已 Running 仍不可达 → 不盲目重启，抛错指引人工排查；
           Starting 中 → 跳过 start 直接等待；Stopped → 执行 ``machine start``；
        4. 在 ``timeout`` 总预算内按 :data:`READY_POLL_INTERVAL` 轮询就绪。

        所有面向用户的进度信息写 stderr（默认日志级别为 WARN，log.info
        对用户不可见，而 machine 启动耗时可达数十秒，静默会像卡死）。
        """
        if await self.ping():
            return

        machines = await self.list_machines()
        default = next((m for m in machines if isinstance(m, dict) and m.get("Default") is True), None)
        if default is None:
            raise PodmanEngineNotReadyError(
                "Cannot connect to the Podman backend and no default podman machine "
                "was found. Start the backend manually (for example `podman machine start`), "
                "or verify `podman system connection list`, then rerun xc."
            )

        name = str(default.get("Name") or "")
        if default.get("Running") is True:
            raise PodmanEngineNotReadyError(
                f"Podman machine '{name}' is reported running but the service is still "
                f"unreachable. Inspect it with `podman machine logs {name}`, or restart "
                f"it with `podman machine stop {name}` followed by `podman machine start {name}`."
            )

        if default.get("Starting") is not True:
            print(
                f"Podman service is unreachable; starting podman machine '{name}' (this may take a while) ...",
                file=sys.stderr,
            )
            try:
                await self.start_machine(name)
            except FileNotFoundError as e:
                raise PodmanEngineNotReadyError(
                    f"Failed to start podman machine '{name}': the podman binary "
                    "disappeared between probes. Start it manually with "
                    f"`podman machine start {name}` and rerun xc."
                ) from e
            except CalledProcessError as e:
                detail = _first_stderr_line(e.output)
                raise PodmanEngineNotReadyError(
                    f"Failed to start podman machine '{name}'"
                    + (f": {detail}" if detail else "")
                    + f". Start it manually with `podman machine start {name}` and rerun xc."
                ) from e
        else:
            print(
                f"Podman machine '{name}' is already starting; waiting until it is ready ...",
                file=sys.stderr,
            )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.ping():
                print(f"Podman machine '{name}' is ready.", file=sys.stderr)
                return
            await asyncio.sleep(READY_POLL_INTERVAL)

        raise PodmanEngineNotReadyError(
            f"Timed out after {timeout:.0f}s waiting for podman machine '{name}' to become "
            f"ready. Check `podman machine logs {name}`, or start it manually with "
            f"`podman machine start {name}`."
        )


def _first_stderr_line(data: bytes | None, *, limit: int = 200) -> str:
    """从 CalledProcessError.output 提取首行非空 stderr 文本（失败诊断用）。"""
    if not data:
        return ""
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""
