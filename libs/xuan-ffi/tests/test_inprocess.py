"""进程内提供者：直接调用目标模块。"""

import uuid
from pathlib import Path

import pytest
from support import FakeFFI

from xuan_ffi import FFIError, FFIUnavailableError, InProcessFFI


def test_describe_reads_version_and_availability(fake_ffi: FakeFFI) -> None:
    info = InProcessFFI("fake", fake_ffi.module).describe()
    assert info.name == "fake"
    assert info.module == fake_ffi.module
    assert info.transport == "in-process"
    assert info.version == "9.9.9"
    assert info.available is True
    assert info.error == ""


def test_invoke_supports_positional_keyword_and_nested_paths(fake_ffi: FakeFFI) -> None:
    api = InProcessFFI("fake", fake_ffi.module)
    assert api.invoke("add", 1, 2) == 3
    assert api.invoke("add", left=2, right=5) == 7
    assert api.invoke("ops.mul", 3, 4) == 12


def test_missing_method_raises(fake_ffi: FakeFFI) -> None:
    with pytest.raises(FFIError, match="不存在属性"):
        InProcessFFI("fake", fake_ffi.module).invoke("nope")


def test_unavailable_module_is_reported_without_raising() -> None:
    api = InProcessFFI("ghost", "ghost_ffi_module_absent")
    info = api.describe()
    assert info.available is False
    assert "无法导入" in info.error
    with pytest.raises(FFIUnavailableError, match="无法导入"):
        api.invoke("version")


def test_close_is_idempotent_and_allows_reload(fake_ffi: FakeFFI) -> None:
    api = InProcessFFI("fake", fake_ffi.module)
    assert api.invoke("add", 1, 1) == 2
    api.close()
    api.close()
    # close 只丢弃引用；再次调用会重新 import（原生 .so 不卸载，见 ADR-03）
    assert api.invoke("add", 2, 2) == 4


def test_describe_does_not_raise_when_probe_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """describe 的契约是「不可用即 available=False」，探针自身抛错也须被吸收。"""
    module = f"broken_ffi_{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{module}.py").write_text(
        "def version():\n    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    info = InProcessFFI("broken", module).describe()
    assert info.available is False
    assert "boom" in info.error
