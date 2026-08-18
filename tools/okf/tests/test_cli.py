"""CLI 子命令的端到端集成测试。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pytest

import okf.cli as cli_module
from okf.cli import (
    _cmd_index,
    _cmd_init,
    _cmd_inspect,
    _cmd_list,
    _cmd_trust,
    _cmd_validate,
    _get_service,
)
from okf.harness import Harness


def _copy_bundle(tmp_path, sample_bundle_path) -> Path:
    """将示例 Bundle 复制到临时目录，避免索引命令覆盖夹具。"""
    dest = tmp_path / "bundle"
    shutil.copytree(sample_bundle_path, dest)
    return dest


def test_cmd_init_creates_skeleton(tmp_path):
    root = tmp_path / "my-bundle"

    code = _cmd_init(argparse.Namespace(path=str(root)))

    assert code == 0
    for name in ["concepts", "playbooks", "references"]:
        assert (root / name).is_dir()
    assert (root / "index.md").is_file()
    assert (root / "log.md").is_file()


def test_cmd_validate_sample_bundle_passes(sample_bundle_path, capsys):
    code = _cmd_validate(argparse.Namespace(path=str(sample_bundle_path), strict=False))

    assert code == 0
    out = capsys.readouterr().out
    assert "Result: PASS" in out
    assert "Errors (0)" in out


def test_cmd_list_all_concepts(sample_bundle_path, capsys):
    code = _cmd_list(
        argparse.Namespace(path=str(sample_bundle_path), type_filter=None, tag_filter=None)
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "concepts/metrics/active_users" in out
    assert "concepts/metrics/revenue" in out
    assert "concepts/tables/customers" in out


def test_cmd_list_filter_by_type(sample_bundle_path, capsys):
    code = _cmd_list(
        argparse.Namespace(path=str(sample_bundle_path), type_filter="Metric", tag_filter=None)
    )

    assert code == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 2
    assert all("Metric" in line for line in lines)
    assert "customers" not in out


def test_cmd_list_filter_by_tag(sample_bundle_path, capsys):
    code = _cmd_list(
        argparse.Namespace(path=str(sample_bundle_path), type_filter=None, tag_filter="finance")
    )

    assert code == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 1
    assert "revenue" in out


def test_cmd_inspect_overview(sample_bundle_path, capsys):
    code = _cmd_inspect(argparse.Namespace(path=str(sample_bundle_path), concept_id=None))

    assert code == 0
    out = capsys.readouterr().out
    assert "概念数量: 3" in out


def test_cmd_inspect_detail(sample_bundle_path, capsys):
    code = _cmd_inspect(
        argparse.Namespace(path=str(sample_bundle_path), concept_id="concepts/metrics/revenue")
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "ID: concepts/metrics/revenue" in out
    assert "Type: Metric" in out
    assert "Title: Revenue" in out


def test_cmd_trust_single_concept(sample_bundle_path, capsys):
    code = _cmd_trust(
        argparse.Namespace(path=str(sample_bundle_path), concept_id="concepts/metrics/revenue")
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "human_reviewed" in out


def test_cmd_trust_all_concepts(sample_bundle_path, capsys):
    code = _cmd_trust(argparse.Namespace(path=str(sample_bundle_path), concept_id=None))

    assert code == 0
    out = capsys.readouterr().out
    assert "concepts/metrics/revenue" in out
    assert "concepts/metrics/active_users" in out
    assert "concepts/tables/customers" in out


def test_cmd_index_generates(tmp_path, sample_bundle_path, capsys):
    dest = _copy_bundle(tmp_path, sample_bundle_path)

    code = _cmd_index(argparse.Namespace(path=str(dest)))

    assert code == 0
    content = (dest / "index.md").read_text(encoding="utf-8")
    assert content.startswith("# bundle")
    assert "## Metric" in content
    assert "## Table" in content


# ── _cmd_init 跳过已存在目录/文件 ─────────────────────────────────────────


def test_cmd_init_skips_existing(tmp_path, capsys):
    root = tmp_path / "my-bundle"
    for name in ["concepts", "playbooks", "references"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "index.md").write_text("# existing", encoding="utf-8")
    (root / "log.md").write_text("# existing", encoding="utf-8")

    code = _cmd_init(argparse.Namespace(path=str(root)))

    assert code == 0
    out = capsys.readouterr().out
    assert out.count("跳过（已存在）") == 5


# ── _get_service KeyError 分支 ────────────────────────────────────────────


def test_get_service_missing(capsys):
    harness = Harness()
    result = _get_service(harness, "nonexistent_service")
    assert result is None
    err = capsys.readouterr().err
    assert "未注册" in err
    assert "nonexistent_service" in err


# ── 服务缺失时的 return 1 分支 ─────────────────────────────────────────────


def test_cmd_validate_service_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "_get_service", lambda h, n: None)
    assert _cmd_validate(argparse.Namespace(path=str(tmp_path))) == 1


def test_cmd_index_service_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "_get_service", lambda h, n: None)
    assert _cmd_index(argparse.Namespace(path=str(tmp_path))) == 1


def test_cmd_inspect_service_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "_get_service", lambda h, n: None)
    assert _cmd_inspect(
        argparse.Namespace(path=str(tmp_path), concept_id=None)
    ) == 1


def test_cmd_trust_service_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "_get_service", lambda h, n: None)
    assert _cmd_trust(
        argparse.Namespace(path=str(tmp_path), concept_id=None)
    ) == 1


def test_cmd_list_service_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "_get_service", lambda h, n: None)
    assert _cmd_list(
        argparse.Namespace(path=str(tmp_path), type_filter=None, tag_filter=None)
    ) == 1


# ── inspect / trust 概念不存在分支 ─────────────────────────────────────────


def test_cmd_inspect_concept_not_found(sample_bundle_path, capsys):
    code = _cmd_inspect(
        argparse.Namespace(path=str(sample_bundle_path), concept_id="nonexistent")
    )
    assert code == 1
    assert "概念不存在" in capsys.readouterr().err


def test_cmd_trust_concept_not_found(sample_bundle_path, capsys):
    code = _cmd_trust(
        argparse.Namespace(path=str(sample_bundle_path), concept_id="nonexistent")
    )
    assert code == 1
    assert "概念不存在" in capsys.readouterr().err


# ── main() 全路径 ─────────────────────────────────────────────────────────


def test_main_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["okf", "--version"])
    with pytest.raises(SystemExit) as e:
        cli_module.main()
    assert e.value.code == 0
    assert "okf" in capsys.readouterr().out


def test_main_no_command_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["okf"])
    with pytest.raises(SystemExit) as e:
        cli_module.main()
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_main_validate(monkeypatch, sample_bundle_path, capsys):
    monkeypatch.setattr(sys, "argv", ["okf", "validate", str(sample_bundle_path)])
    with pytest.raises(SystemExit) as e:
        cli_module.main()
    assert e.value.code == 0
    assert "Result: PASS" in capsys.readouterr().out


def test_main_exception_prints_traceback(monkeypatch, capsys):
    def boom(args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "_cmd_validate", boom)
    monkeypatch.setattr(sys, "argv", ["okf", "validate", "somepath"])
    with pytest.raises(SystemExit) as e:
        cli_module.main()
    assert e.value.code == 1
    assert "boom" in capsys.readouterr().err


def test_import_main_module():
    """导入 okf.__main__ 模块不产生副作用（覆盖入口 import 行）。"""
    import okf.__main__  # noqa: F401

    assert okf.__main__.__name__ == "okf.__main__"
