"""子进程 FFI worker：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应。

单独成模块以便 ``python -m xuan_ffi.worker <module>`` 直接运行；协议本身
不依赖调用方，任何语言都可实现同一线格式（P2-3 的最小可移植契约）。
"""

from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType

from .api import FFIError
from .inprocess import resolve

_DESCRIBE = "__describe__"


def _describe(module: ModuleType) -> dict[str, object]:
    version_getter = getattr(module, "version", None)
    version = str(version_getter()) if callable(version_getter) else str(getattr(module, "__version__", ""))
    available_getter = getattr(module, "is_available", None)
    available = bool(available_getter()) if callable(available_getter) else True
    return {"module": module.__name__, "version": version, "available": available}


def _dispatch(module: ModuleType, request: dict[str, object]) -> object:
    method = request.get("method")
    if not isinstance(method, str) or not method:
        raise FFIError("method 须为非空字符串")
    args = request.get("args", [])
    kwargs = request.get("kwargs", {})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise FFIError("args 须为数组、kwargs 须为对象")
    if method == _DESCRIBE:
        return _describe(module)
    target = resolve(module, method)
    return target(*args, **kwargs)


def _error(exc: BaseException) -> dict[str, str]:
    return {"code": type(exc).__name__, "message": str(exc)}


def main(argv: list[str] | None = None) -> int:
    """从 ``argv[0]`` 取模块名并进入请求循环；EOF 即退出。"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("用法：python -m xuan_ffi.worker <module>", file=sys.stderr)
        return 2
    module_name = arguments[0]

    module: ModuleType | None = None
    load_error: BaseException | None = None
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 —— 加载失败要回传给调用方而非直接崩溃
        load_error = exc

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise FFIError("请求须为 JSON 对象")
        except Exception as exc:  # noqa: BLE001
            _write({"id": None, "error": _error(exc)})
            continue

        request_id = request.get("id")
        try:
            if load_error is not None:
                raise FFIError(f"无法导入 FFI 模块 {module_name!r}：{load_error}")
            assert module is not None
            result = _dispatch(module, request)
            text = json.dumps({"id": request_id, "result": result}, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 —— 任何失败都转为可读响应
            text = json.dumps({"id": request_id, "error": _error(exc)}, ensure_ascii=False)
        _write_raw(text)
    return 0


def _write(payload: dict[str, object]) -> None:
    _write_raw(json.dumps(payload, ensure_ascii=False))


def _write_raw(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
