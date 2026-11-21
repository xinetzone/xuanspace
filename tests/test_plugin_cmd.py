"""``xs plugin`` 命令与内置命令清单的一致性测试（P1-1 验收）。"""

import json
import tomllib
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner
from xs.cli import app as cli_app
from xs.commands import plugin_cmd
from xuan_plugin.manifest import InjectDecl, PluginManifest, ProvideDecl, manifest_from_pyproject

REPO_ROOT = Path(__file__).resolve().parents[1]
XS_PYPROJECT = REPO_ROOT / "tools" / "xs" / "pyproject.toml"
runner = CliRunner()


def _declared_manifest():
    with open(XS_PYPROJECT, "rb") as handle:
        data = tomllib.load(handle)
    manifest = manifest_from_pyproject(data, source="tools/xs/pyproject.toml")
    assert manifest is not None, "tools/xs/pyproject.toml 缺少 [tool.xuan.plugin]"
    return manifest


def _registered_command_names() -> set[str]:
    names = {command.name for command in cli_app.registered_commands}
    names |= {group.name for group in cli_app.registered_groups}
    return names


def test_declared_commands_match_cli_registry() -> None:
    """清单声明的能力键必须与实际注册的命令逐一对应（防双真相漂移）。"""
    manifest = _declared_manifest()
    declared = {key.split("#", 1)[1] for key in manifest.provide_keys()}
    registered = _registered_command_names()
    assert declared == registered, (
        f"仅声明未注册：{sorted(declared - registered)}；仅注册未声明：{sorted(registered - declared)}"
    )
    assert len(declared) == len(registered)
    # 15 个既有子命令 + 本阶段新增的 plugin 组
    assert len(declared) == 16


def test_plugin_list_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    result = runner.invoke(cli_app, ["plugin", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    entry = next(item for item in payload if item["id"] == "xs.commands")
    assert entry["version"] == "0.1.0"
    assert entry["priority"] == 100
    assert len(entry["provide"]) == 16
    assert entry["source"] == "tools/xs/pyproject.toml"


def test_plugin_graph_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    result = runner.invoke(cli_app, ["plugin", "graph", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "xs.commands" in payload["nodes"]
    assert payload["conflicts"] == []


def test_plugin_graph_dot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    result = runner.invoke(cli_app, ["plugin", "graph", "--dot"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("digraph xuan {")
    assert '"xs.commands";' in result.output


def test_plugin_list_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    # 固定 80 列以复现窄终端：ID 与来源必须以换行而非省略号呈现
    monkeypatch.setattr(plugin_cmd, "console", Console(width=80))
    result = runner.invoke(cli_app, ["plugin", "list"])
    assert result.exit_code == 0, result.output
    assert "xs.commands" in result.output
    assert "reloadable" in result.output
    assert "tools/xs" in result.output
    assert "…" not in result.output
    assert "共" in result.output


def test_entry_points_are_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    """entry points 提供的清单与静态声明一并进入候选集。"""

    class _FakeEntryPoint:
        name = "demo.plugin"
        value = "demo:Plugin"

        def load(self):
            from xuan_plugin.manifest import PluginManifest

            return type("Plugin", (), {"MANIFEST": PluginManifest(id="demo.plugin", version="2.0.0")})

    monkeypatch.setattr(plugin_cmd, "entry_points", lambda group: [_FakeEntryPoint()])
    manifests = plugin_cmd.collect_manifests(REPO_ROOT)
    ids = {manifest.id for manifest in manifests}
    assert {"xs.commands", "demo.plugin"} <= ids


def test_broken_entry_point_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenEntryPoint:
        name = "broken.plugin"
        value = "broken:Plugin"

        def load(self):
            raise ImportError("missing dependency")

    monkeypatch.setattr(plugin_cmd, "entry_points", lambda group: [_BrokenEntryPoint()])
    manifests = plugin_cmd.collect_manifests(REPO_ROOT)
    assert {manifest.id for manifest in manifests} == {"xs.commands"}


# ─── P1-2：apply --dry-run 的求解面 ─────────────────────────────────────


def _fake_manifest(plugin_id: str, *, provide: tuple[str, ...] = (), inject: tuple[str, ...] = ()) -> PluginManifest:
    return PluginManifest(
        id=plugin_id,
        version="0.1.0",
        provide=tuple(ProvideDecl(key=key) for key in provide),
        inject=tuple(InjectDecl(key=key) for key in inject),
    )


def _patch_manifests(monkeypatch: pytest.MonkeyPatch, manifests: list[PluginManifest]) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(plugin_cmd, "collect_manifests", lambda root: tuple(manifests))


def test_apply_dry_run_json_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一工作区两次 dry-run 的输出必须逐字节一致（可复现性）。"""
    monkeypatch.chdir(REPO_ROOT)
    first = runner.invoke(cli_app, ["plugin", "apply", "--dry-run", "--json"])
    second = runner.invoke(cli_app, ["plugin", "apply", "--dry-run", "--json"])
    assert first.exit_code == 0, first.output
    assert first.output == second.output

    payload = json.loads(first.output)
    assert payload["status"] == "ok"
    assert payload["order"] == ["xs.commands"]
    assert payload["domains"] == [{"labels": [], "nodes": ["xs.commands"]}]
    assert payload["inactive"] == []
    assert len(payload["provides"]) == 16


def test_apply_refuses_when_not_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    result = runner.invoke(cli_app, ["plugin", "apply"])
    assert result.exit_code == 1
    assert "P1-3" in result.output


def test_apply_dry_run_text_lists_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(plugin_cmd, "console", Console(width=100))
    result = runner.invoke(cli_app, ["plugin", "apply", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "装配顺序" in result.output
    assert "隔离域" in result.output
    assert "默认域" in result.output
    assert "可解性" in result.output


def test_apply_conflict_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "xuan:demo/clash#one"
    _patch_manifests(monkeypatch, [_fake_manifest("a", provide=(key,)), _fake_manifest("b", provide=(key,))])

    first = runner.invoke(cli_app, ["plugin", "apply", "--dry-run", "--json"])
    second = runner.invoke(cli_app, ["plugin", "apply", "--dry-run", "--json"])
    assert first.exit_code == 1
    assert first.output == second.output

    payload = json.loads(first.output)
    assert payload["status"] == "error"
    assert payload["kind"] == "conflict"
    assert payload["details"] == [f"{key}: a, b"]


def test_apply_cycle_reports_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_manifests(
        monkeypatch,
        [
            _fake_manifest("a", provide=("xuan:demo/a#key",), inject=("xuan:demo/b#key",)),
            _fake_manifest("b", provide=("xuan:demo/b#key",), inject=("xuan:demo/a#key",)),
        ],
    )
    result = runner.invoke(cli_app, ["plugin", "apply", "--dry-run", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["kind"] == "cycle"
    assert payload["details"] == ["a -> b -> a"]


def test_apply_missing_dependency_is_reported_as_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "xuan:demo/absent#one"
    _patch_manifests(monkeypatch, [_fake_manifest("a", inject=(key,))])

    result = runner.invoke(cli_app, ["plugin", "apply", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["order"] == []
    assert payload["inactive"] == [{"plugin": "a", "missing": [key]}]
