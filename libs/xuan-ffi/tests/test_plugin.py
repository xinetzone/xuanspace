"""插件清单与装配：三条能力键可登记、可撤销，子进程模式可重复装载。"""

import sys
import tomllib
from pathlib import Path

from support import FakeFFI
from xuan_plugin.context import Context
from xuan_plugin.manifest import manifest_from_pyproject
from xuan_plugin.plugin import FiberState

from xuan_ffi import MANIFEST, PLUGIN, InProcessFFI, SubprocessFFI, ffi_key

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"


def test_manifest_declares_builtin_keys() -> None:
    assert MANIFEST.id == "xuan.ffi"
    assert MANIFEST.lifecycle == "static"
    assert MANIFEST.provide_keys() == (
        "xuan:ffi/caffe#api",
        "xuan:ffi/demo#api",
        "xuan:ffi/npu#api",
    )


def test_manifest_matches_pyproject_declaration() -> None:
    pyproject = REPO_ROOT / "libs" / "xuan-ffi" / "pyproject.toml"
    with open(pyproject, "rb") as handle:
        declared = manifest_from_pyproject(tomllib.load(handle), source="libs/xuan-ffi/pyproject.toml")
    assert declared is not None
    assert MANIFEST.contract() == declared.contract()


def test_apply_registers_and_disposes() -> None:
    ctx = Context()
    fiber = ctx.plugin(PLUGIN, {"mode": "in-process"})

    assert fiber.state is FiberState.ACTIVE
    assert ctx.registered_keys() == (ffi_key("caffe"), ffi_key("demo"), ffi_key("npu"))
    assert isinstance(ctx.get(ffi_key("demo")), InProcessFFI)

    assert ctx.dispose() == []
    assert ctx.registered_keys() == ()
    assert ctx.get_effects() == []


def test_subprocess_mode_is_repeatable(fake_ffi: FakeFFI) -> None:
    """同一进程内连续两轮装配—调用—撤销，每轮都拿到新的 worker 进程。"""
    pids: list[int | None] = []
    for _ in range(2):
        ctx = Context()
        fiber = ctx.plugin(
            PLUGIN,
            {
                "mode": "subprocess",
                "modules": {"demo": fake_ffi.module},
                "options": {
                    "python": sys.executable,
                    "extra_paths": [str(fake_ffi.directory), str(SRC)],
                },
            },
        )
        assert fiber.state is FiberState.ACTIVE

        api = ctx.get(ffi_key("demo"))
        assert isinstance(api, SubprocessFFI)
        assert api.invoke("add", 1, 2) == 3
        pids.append(api.pid)

        assert ctx.dispose() == []
        assert api.pid is None

    assert all(pid is not None for pid in pids)
    assert len(set(pids)) == 2


def test_unknown_mode_fails_activation() -> None:
    ctx = Context()
    fiber = ctx.plugin(PLUGIN, {"mode": "carrier-pigeon"})
    try:
        assert fiber.state is FiberState.FAILED
        assert fiber.error is not None
        assert "未知传输模式" in str(fiber.error)
    finally:
        ctx.dispose()
