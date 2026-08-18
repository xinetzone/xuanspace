"""CLI 子命令的端到端集成测试。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from okf.cli import (
    _cmd_index,
    _cmd_init,
    _cmd_inspect,
    _cmd_list,
    _cmd_trust,
    _cmd_validate,
)


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
