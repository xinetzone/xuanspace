"""子进程隔离代理：以 JSON Lines over stdio 调用目标模块。

隔离模式解决两类进程内模式解决不了的问题（ADR-03）：

1. **反复装卸 A/B**：``close()`` 结束子进程，``start()`` 重新拉起，原生静态
   状态随进程一起消失，因此可重复装载；
2. **崩溃可恢复**：原生段崩溃只影响 worker，调用方进程仍可继续。

边界：请求与响应须为 JSON 可序列化值；对象图密集的调用不进入本模式。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path

from .api import FFIAPI, SUBPROCESS, FFIError, FFIInfo

_DESCRIBE = "__describe__"


class SubprocessFFI(FFIAPI):
    """把目标模块放进独立解释器进程，经 JSON Lines 协议调用。"""

    transport = SUBPROCESS

    def __init__(
        self,
        name: str,
        module: str,
        *,
        python: str | None = None,
        extra_paths: Iterable[str | Path] = (),
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.module = module
        self._python = python or sys.executable
        self._extra_paths = tuple(str(path) for path in extra_paths)
        self._env = None if env is None else dict(env)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._next_id = 0

    @property
    def pid(self) -> int | None:
        """worker 进程号；未启动或已关闭时为 None。"""
        return None if self._process is None else self._process.pid

    def start(self) -> SubprocessFFI:
        """拉起 worker（幂等）；已存活则不重复创建。"""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self
            self._spawn()
            return self

    def _spawn(self) -> None:
        self._process = subprocess.Popen(
            [self._python, "-m", "xuan_ffi.worker", self.module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=self._build_env(),
        )

    def _build_env(self) -> dict[str, str]:
        environ = dict(os.environ if self._env is None else self._env)
        environ.setdefault("PYTHONIOENCODING", "utf-8")
        environ.setdefault("PYTHONUTF8", "1")
        if self._extra_paths:
            parts = list(self._extra_paths)
            current = environ.get("PYTHONPATH")
            if current:
                parts.append(current)
            environ["PYTHONPATH"] = os.pathsep.join(parts)
        return environ

    def _ensure(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            self._spawn()
        assert self._process is not None
        return self._process

    def _request(self, method: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        with self._lock:
            process = self._ensure()
            self._next_id += 1
            request_id = self._next_id
            payload = json.dumps(
                {"id": request_id, "method": method, "args": list(args), "kwargs": kwargs},
                ensure_ascii=False,
            )
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(payload + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise FFIError(f"与 FFI 子进程通信失败（{self.module}）：{exc}") from exc

        if not line:
            raise FFIError(f"FFI 子进程意外退出：{self.module}（returncode={process.poll()}）")
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise FFIError(f"FFI 子进程响应不合约：{line.strip()!r}")
        error = response.get("error")
        if error is not None:
            code = error.get("code", "Error") if isinstance(error, dict) else "Error"
            message = error.get("message", "") if isinstance(error, dict) else str(error)
            raise FFIError(f"{code}: {message}")
        return response.get("result")

    def describe(self) -> FFIInfo:
        try:
            data = self._request(_DESCRIBE, (), {})
        except (FFIError, ValueError) as exc:
            return FFIInfo(self.name, self.module, self.transport, available=False, error=str(exc))
        if not isinstance(data, dict):
            return FFIInfo(self.name, self.module, self.transport, available=False, error="describe 返回非对象")
        return FFIInfo(
            self.name,
            self.module,
            self.transport,
            version=str(data.get("version", "")),
            available=bool(data.get("available", False)),
            error=str(data.get("error", "")),
        )

    def invoke(self, method: str, *args: object, **kwargs: object) -> object:
        if not method:
            raise FFIError("方法名不能为空")
        if method.startswith("__"):
            raise FFIError(f"方法名 {method!r} 为保留名，不可经能力面调用")
        return self._request(method, args, dict(kwargs))

    def close(self) -> None:
        """结束 worker 并释放管道；幂等，可随后 ``start()`` 重新装载。"""
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()

    def __enter__(self) -> SubprocessFFI:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
