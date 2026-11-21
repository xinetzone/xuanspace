"""P2-3 验收：切换后端/传输不改调用方代码。"""

import sys
from pathlib import Path

from support import FakeFFI

from xuan_ffi import FFIAPI, InProcessFFI, SubprocessFFI

SRC = Path(__file__).resolve().parents[1] / "src"


def consume(api: FFIAPI) -> dict[str, object]:
    """唯一的调用方代码：只依赖能力面，不 import 具体 FFI 发行包。"""
    info = api.describe()
    return {
        "version": info.version,
        "sum": api.invoke("add", 4, 6),
        "product": api.invoke("ops.mul", 3, 5),
    }


def test_same_consumer_works_across_transports(fake_ffi: FakeFFI) -> None:
    in_process = InProcessFFI("fake", fake_ffi.module)
    subprocess_api = SubprocessFFI(
        "fake",
        fake_ffi.module,
        python=sys.executable,
        extra_paths=[fake_ffi.directory, SRC],
    )
    try:
        expected = {"version": "9.9.9", "sum": 10, "product": 15}
        assert consume(in_process) == expected
        assert consume(subprocess_api) == expected
    finally:
        in_process.close()
        subprocess_api.close()
