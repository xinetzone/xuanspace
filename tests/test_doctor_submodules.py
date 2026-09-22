"""Tests for xs doctor submodule consistency checks."""

import subprocess
from pathlib import Path

from xs.commands.doctor_cmd import check_submodules, parse_submodule_paths

SUB_PATH = "libs/demo"


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed in {cwd}:\n{result.stderr}")
    return result


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-b", "main", "-q")
    _git(path, "config", "user.name", "test")
    _git(path, "config", "user.email", "test@example.com")


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message, "-q")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _make_super_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin"
    super_repo = tmp_path / "super"
    _init_repo(origin)
    (origin / "README.md").write_text("v1", encoding="utf-8")
    _commit_all(origin, "initial")
    _init_repo(super_repo)
    _git(super_repo, "submodule", "add", "../origin", SUB_PATH)
    _commit_all(super_repo, "add submodule")
    return super_repo, origin


def _result_for(results, path: str):
    return next(result for result in results if result.name == path)


def test_parse_submodule_paths():
    text = (
        '[submodule "mystx"]\n'
        "\tpath = libs/mystx\n"
        "\turl = ../../mystx.git\n"
        '[submodule "tvm-ffi"]\n'
        "path = vendor/tvm-ffi\n"
        " url = https://github.com/apache/tvm-ffi.git\n"
    )
    assert parse_submodule_paths(text) == ["libs/mystx", "vendor/tvm-ffi"]


def test_check_submodules_consistent(tmp_path: Path):
    super_repo, _ = _make_super_with_submodule(tmp_path)

    result = _result_for(check_submodules(super_repo), SUB_PATH)

    assert result.status == "ok"
    assert result.required is True


def test_check_submodules_drift_reports_expected_and_actual(tmp_path: Path):
    super_repo, origin = _make_super_with_submodule(tmp_path)
    pinned = _git(origin, "rev-parse", "HEAD").stdout.strip()
    (origin / "README.md").write_text("v2", encoding="utf-8")
    advanced = _commit_all(origin, "second")
    sub = super_repo / SUB_PATH
    _git(sub, "fetch", "origin", "-q")
    _git(sub, "checkout", advanced, "-q")

    result = _result_for(check_submodules(super_repo), SUB_PATH)

    assert result.status == "error"
    assert pinned[:7] in (result.message or "")
    assert advanced[:7] in (result.message or "")
    assert "submodule update" in (result.install_hint or "")


def test_check_submodules_uninitialized_is_warning(tmp_path: Path):
    super_repo, _ = _make_super_with_submodule(tmp_path)
    _git(super_repo, "submodule", "deinit", "-f", SUB_PATH)

    result = _result_for(check_submodules(super_repo), SUB_PATH)

    assert result.status == "warning"
    assert result.required is False
    assert "--init" in (result.install_hint or "")


def test_check_submodules_dirty_is_warning(tmp_path: Path):
    super_repo, _ = _make_super_with_submodule(tmp_path)
    (super_repo / SUB_PATH / "README.md").write_text("locally modified", encoding="utf-8")

    result = _result_for(check_submodules(super_repo), SUB_PATH)

    assert result.status == "warning"
    assert result.required is False


def test_check_submodules_without_gitmodules_skips(tmp_path: Path):
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()

    results = check_submodules(plain_dir)

    assert len(results) == 1
    assert results[0].status == "skip"
