"""Tests for xs doc command argument resolution."""

from pathlib import Path

from xs.commands import doc_cmd


class _Completed:
    returncode = 0
    stderr = ""


def test_doc_linkcheck_builds_str_argv(monkeypatch, tmp_path: Path):
    # 直接以 Python 调用 doc_linkcheck 时，参数默认值不经过 Typer 解析，
    # 必须已是普通值而非 typer.Option 对象，否则会拼进 sphinx 命令行。
    (tmp_path / "doc").mkdir()
    monkeypatch.setattr(doc_cmd, "find_workspace_root", lambda: tmp_path)

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return _Completed()

    monkeypatch.setattr(doc_cmd.subprocess, "run", fake_run)

    doc_cmd.doc_linkcheck()

    assert len(captured) == 1
    argv = captured[0]
    assert all(isinstance(arg, str) for arg in argv)
    assert argv[argv.index("-b") + 1] == "linkcheck"
    assert argv[-1] == str(tmp_path / "doc" / "_build" / "linkcheck")
