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
from dataclasses import dataclass
from typing import Any

from .logging_utils import log

__all__ = [
    "CalledProcessError",
    "ExistingContainer",
    "Podman",
    "wait_with_timeout",
]

# 子进程异常类型从本边界模块统一再导出，避免其他模块直接 import subprocess（TR-4.2）
CalledProcessError = subprocess.CalledProcessError


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

    async def output(
        self, podman_args: list[str], cmd: str = "", cmd_args: list[str] | None = None
    ) -> bytes:
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

            raise subprocess.CalledProcessError(p.returncode, " ".join(cmd_ls), stderr_data)

    async def _readchunk(self, reader: asyncio.StreamReader) -> bytes:
        try:
            return await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as e:
            return e.partial
        except asyncio.LimitOverrunError as e:
            return await reader.read(e.consumed)

    async def _format_stream(
        self, reader: asyncio.StreamReader, sink: Any, log_formatter: str
    ) -> None:
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

            if log_formatter is not None:
                p = await asyncio.create_subprocess_exec(
                    *cmd_ls,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    close_fds=False,
                )  # pylint: disable=consider-using-with

                assert p.stdout is not None
                assert p.stderr is not None

                # This is hacky to make the tasks not get garbage collected
                # https://github.com/python/cpython/issues/91887
                out_t = asyncio.create_task(
                    self._format_stream(p.stdout, sys.stdout, log_formatter)
                )
                task_reference.add(out_t)
                out_t.add_done_callback(task_reference.discard)

                err_t = asyncio.create_task(
                    self._format_stream(p.stderr, sys.stdout, log_formatter)
                )
                task_reference.add(err_t)
                err_t.add_done_callback(task_reference.discard)

            elif suppress_output:
                p = await asyncio.create_subprocess_exec(
                    *cmd_ls,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=False,
                )  # pylint: disable=consider-using-with

            else:
                p = await asyncio.create_subprocess_exec(*cmd_ls, close_fds=False)  # pylint: disable=consider-using-with

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
