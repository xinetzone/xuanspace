"""``xuan-build`` 的回归测试。"""

import subprocess
import tomllib
from pathlib import Path

import pytest
from xuan_plugin.context import Context
from xuan_plugin.plugin import FiberState

from xuan_build import DISPATCH_KEY, MANIFEST, PLUGIN, dispatch
from xuan_build.backends import NativeBackend, OtherBackend, PythonBackend, StaticBackend
from xuan_build.dispatch import BuildDispatcher

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB_PYPROJECT = REPO_ROOT / "libs" / "xuan-build" / "pyproject.toml"


class Recorder:
    """记录输出，替代 Rich Console。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


def test_dispatch_covers_all_project_types() -> None:
    dispatcher = BuildDispatcher()
    assert dispatcher.registered_kinds() == ("python", "native", "static", "other")
    assert isinstance(dispatch("python"), PythonBackend)
    assert isinstance(dispatch("native"), NativeBackend)
    assert isinstance(dispatch("static"), StaticBackend)
    assert isinstance(dispatch("other"), OtherBackend)
    assert isinstance(dispatch("unexpected"), OtherBackend)


def test_skip_messages_are_stable() -> None:
    assert dispatch("static").skip_message("site") == "[yellow]跳过静态项目: site (无需构建)[/yellow]"
    assert dispatch("other").skip_message("misc") == "[yellow]跳过未知类型项目: misc[/yellow]"
    assert dispatch("python").skip_message("demo") is None
    assert dispatch("native").skip_message("demo") is None


def test_python_backend_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert PythonBackend().build(tmp_path, Recorder()) is True
    assert calls and calls[0][-3:] == ["-m", "build", "--wheel"]


def test_python_backend_failure_reports_both_streams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output="OUT", stderr="ERR")

    monkeypatch.setattr(subprocess, "run", fake_run)
    recorder = Recorder()
    assert PythonBackend().build(tmp_path, recorder) is False
    assert recorder.messages == ["[red]构建失败:[/red]", "[red]OUT[/red]", "[red]ERR[/red]"]


def test_native_backend_falls_back_to_wheel_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[2:4] == ["pip", "install"]:
            raise subprocess.CalledProcessError(1, command, output="EDITED", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    recorder = Recorder()
    assert NativeBackend().build(tmp_path, recorder) is True
    assert commands[0][2:4] == ["pip", "install"]
    assert commands[1][-3:] == ["-m", "build", "--wheel"]
    assert recorder.messages == ["[yellow]pip install -e 失败，回退 python -m build --wheel[/yellow]"]


def test_manifest_matches_pyproject() -> None:
    declared = MANIFEST
    raw = tomllib.loads(LIB_PYPROJECT.read_text(encoding="utf-8"))
    plugin = raw["tool"]["xuan"]["plugin"]
    assert plugin["id"] == declared.id
    assert plugin["version"] == declared.version
    assert [item["key"] for item in plugin["provide"]] == [decl.key for decl in declared.provide]
    assert plugin["priority"] == declared.priority
    assert plugin["lifecycle"] == declared.lifecycle
    assert plugin["reversible"] is declared.reversible


def test_plugin_registration_is_reversible() -> None:
    ctx = Context()
    fiber = ctx.plugin(PLUGIN, {})
    assert fiber.state is FiberState.ACTIVE
    assert ctx.get(DISPATCH_KEY) is not None
    assert ctx.dispose() == []
    assert ctx.registered_keys() == ()
    assert ctx.get_effects() == []
