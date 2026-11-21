"""P2-1：``xs build`` 经能力键分派后端的行为约束。"""

from pathlib import Path

import pytest
from xs.commands import build_cmd
from xs.config import ProjectType
from xuan_build.backends import BuildBackend, Reporter


def _capture_build(project_path: Path, project_type: ProjectType) -> str:
    with build_cmd.console.capture() as capture:
        assert build_cmd._build_single_project(project_path, project_type) is True
    return capture.get()


def test_python_build_is_delegated_to_dispatched_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    class FakeBackend:
        kind = "python"

        def skip_message(self, project_name: str) -> str | None:
            return None

        def build(self, project_path: Path, reporter: Reporter) -> bool:
            calls.append((self.kind, project_path))
            return True

    backend: BuildBackend = FakeBackend()
    seen: list[str] = []
    monkeypatch.setattr(build_cmd, "dispatch", lambda project_type: seen.append(project_type) or backend)

    output = _capture_build(tmp_path, ProjectType.PYTHON)

    assert seen == [ProjectType.PYTHON]
    assert calls == [("python", tmp_path)]
    assert "构建成功" in output


def test_static_build_skips_without_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    class FakeBackend:
        kind = "static"

        def skip_message(self, project_name: str) -> str | None:
            return f"跳过静态项目: {project_name}"

        def build(self, project_path: Path, reporter: Reporter) -> bool:
            calls.append(project_path)
            return True

    backend: BuildBackend = FakeBackend()
    monkeypatch.setattr(build_cmd, "dispatch", lambda project_type: backend)

    output = _capture_build(tmp_path, ProjectType.STATIC)

    assert calls == []
    assert output.strip() == f"跳过静态项目: {tmp_path.name}"


def test_other_build_keeps_legacy_skip_message(tmp_path: Path) -> None:
    output = _capture_build(tmp_path, ProjectType.OTHER)
    assert output.strip() == f"跳过未知类型项目: {tmp_path.name}"
