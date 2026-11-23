"""测试夹具：在临时目录伪造一个纯 Python FFI 模块。

用假模块而不是真实 ``caffe_ffi``，是为了让「传输切换」与「可重复装载」两条
验收不依赖原生扩展是否在本机可用；真实内置名称只在清单层校验。
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from support import SOURCE, FakeFFI


@pytest.fixture
def fake_ffi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeFFI]:
    module = f"fake_ffi_{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{module}.py").write_text(SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield FakeFFI(tmp_path, module)
    sys.modules.pop(module, None)
