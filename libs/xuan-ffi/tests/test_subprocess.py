"""子进程隔离代理：JSON Lines 协议、错误传播与可重复装载。"""

import sys
from pathlib import Path

import pytest
from support import FakeFFI

from xuan_ffi import FFIError, SubprocessFFI

SRC = Path(__file__).resolve().parents[1] / "src"


def _api(fake_ffi: FakeFFI) -> SubprocessFFI:
    return SubprocessFFI(
        "fake",
        fake_ffi.module,
        python=sys.executable,
        extra_paths=[fake_ffi.directory, SRC],
    )


def test_invoke_roundtrip(fake_ffi: FakeFFI) -> None:
    api = _api(fake_ffi)
    try:
        info = api.describe()
        assert info.transport == "subprocess"
        assert info.version == "9.9.9"
        assert info.available is True
        assert api.invoke("add", 2, 3) == 5
        assert api.invoke("ops.mul", 3, 4) == 12
    finally:
        api.close()


def test_repeated_load_uses_fresh_processes(fake_ffi: FakeFFI) -> None:
    """子进程模式的核心收益：关掉即回收原生静态状态，可再装载。"""
    api = _api(fake_ffi)
    pids: list[int] = []
    for _ in range(3):
        api.start()
        assert api.pid is not None
        pids.append(api.pid)
        assert api.invoke("add", 1, 1) == 2
        api.close()
        assert api.pid is None
    assert len(set(pids)) == 3


def test_context_manager_closes(fake_ffi: FakeFFI) -> None:
    with _api(fake_ffi) as api:
        assert api.invoke("add", 4, 6) == 10
        pid = api.pid
    assert pid is not None
    assert api.pid is None


def test_missing_method_is_reported_as_error(fake_ffi: FakeFFI) -> None:
    api = _api(fake_ffi)
    try:
        with pytest.raises(FFIError, match="不存在属性"):
            api.invoke("nope")
    finally:
        api.close()


def test_non_serializable_result_is_rejected(fake_ffi: FakeFFI) -> None:
    """序列化边界必须显式报错，不能静默返回半个值。"""
    api = _api(fake_ffi)
    try:
        with pytest.raises(FFIError, match="serializable"):
            api.invoke("unsupported")
    finally:
        api.close()


def test_reserved_method_is_rejected_without_spawning(fake_ffi: FakeFFI) -> None:
    api = _api(fake_ffi)
    try:
        with pytest.raises(FFIError, match="保留名"):
            api.invoke("__describe__")
    finally:
        api.close()


def test_unavailable_module_describes_error() -> None:
    api = SubprocessFFI("ghost", "ghost_ffi_module_absent", python=sys.executable, extra_paths=[SRC])
    try:
        info = api.describe()
        assert info.available is False
        assert "无法导入" in info.error
    finally:
        api.close()
