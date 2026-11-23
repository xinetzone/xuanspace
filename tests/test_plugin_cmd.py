"""``xs plugin`` 命令与内置命令清单的一致性测试（P1-1 验收）。"""

import json
import re
import tomllib
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner
from xs.cli import app as cli_app
from xs.commands import plugin_cmd
from xs.plugins.commands import COMMAND_SPECS, MANIFEST, PLUGIN
from xs.plugins.registry import CommandRegistry
from xuan_plugin.context import Context
from xuan_plugin.manifest import (
    InjectDecl,
    PluginManifest,
    ProvideDecl,
    manifest_from_pyproject,
    merge_manifests,
    parse_manifest,
)
from xuan_plugin.plugin import FiberState

REPO_ROOT = Path(__file__).resolve().parents[1]
XS_PYPROJECT = REPO_ROOT / "tools" / "xs" / "pyproject.toml"
HELP_SNAPSHOT = REPO_ROOT / "tests" / "snapshots" / "xs_help.txt"
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


def test_help_output_matches_snapshot() -> None:
    """P1-3 验收：命令改由插件装配后，``xs --help`` 内容与改造前逐字相同。

    Rich 会按终端能力选择圆角/直角边框与可用宽度（79/80 列），这两处差异
    不属于帮助内容，故比较前统一归一到 ``+``/``-``/``|`` 并去掉行尾填充；
    命令名、顺序、说明与用法文本仍须逐字一致。
    """
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0, result.output
    assert _normalize_help(result.output) == _normalize_help(HELP_SNAPSHOT.read_text(encoding="utf-8"))


_BOX_TRANSLATION = str.maketrans({"╭": "┌", "╮": "┐", "╰": "└", "╯": "┘", "│": "|", "─": "-"})


def _normalize_help(text: str) -> str:
    """归一化终端渲染差异：边框字形、行尾填充与边框长度。"""
    lines: list[str] = []
    for line in text.splitlines():
        line = line.rstrip().translate(_BOX_TRANSLATION)
        line = re.sub(r"\s+\|$", "|", line)
        line = re.sub(r"-{2,}", "--", line)
        lines.append(line)
    return "\n".join(lines).strip("\n")


def test_builtin_manifest_matches_static_declaration() -> None:
    """代码清单与 pyproject 静态清单必须是同一契约（同 id 可合并）。"""
    declared = _declared_manifest()
    assert MANIFEST.contract() == declared.contract()
    assert merge_manifests([declared, MANIFEST]) == (declared,)


def test_command_plugin_registration_is_reversible() -> None:
    """插件登记的命令与能力键必须可全部撤销，卸载后无残留。"""
    registry = CommandRegistry()
    ctx = Context()
    fiber = ctx.plugin(PLUGIN, {"registry": registry})

    assert fiber.state is FiberState.ACTIVE
    assert len(registry.entries()) == len(COMMAND_SPECS) == 16
    assert len(ctx.registered_keys()) == 16

    assert ctx.dispose() == []
    assert registry.entries() == ()
    assert ctx.registered_keys() == ()
    assert ctx.get_effects() == []


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
    assert payload["status"] == "ok"
    assert payload["diagnostics"] == []
    assert "xs.commands" in payload["nodes"]
    assert payload["conflicts"] == []


def test_plugin_graph_dot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    result = runner.invoke(cli_app, ["plugin", "graph", "--dot"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("digraph xuan {")
    assert '"xs.commands";' in result.output
    assert "// 诊断摘要" not in result.output


def test_plugin_graph_text_marks_ambiguous_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """同优先级并列的键在文本图中标为冲突，不再静默指向某个提供者。"""
    key = "xuan:cli/commands#list"
    producers = [
        parse_manifest(
            {
                "id": plugin_id,
                "version": "0.1.0",
                "priority": 100,
                "provide": [{"key": key, "mode": "multi"}],
            }
        )
        for plugin_id in ("a", "b")
    ]
    consumer = parse_manifest({"id": "consumer", "version": "0.1.0", "inject": [{"key": key}]})
    monkeypatch.setattr(plugin_cmd, "collect_manifests", lambda root: [*producers, consumer])
    monkeypatch.chdir(REPO_ROOT)

    result = runner.invoke(cli_app, ["plugin", "graph"])

    # P2-16：冲突没有唯一胜出者，诊断图照常输出，但退出码必须非零
    assert result.exit_code == 1, result.output
    assert "→ 冲突" in result.output
    assert "→ 未解析" not in result.output


def _conflict_and_missing_manifests() -> list[PluginManifest]:
    """同时含「能力键冲突」（全 single 多提供者）与「必需依赖未解析」的清单集合。"""
    key = "xuan:cli/commands#list"
    return [
        _fake_manifest("a", provide=(key,)),
        _fake_manifest("b", provide=(key,)),
        _fake_manifest("consumer", inject=(key, "xuan:cli/commands#missing")),
    ]


def _expected_diagnostics() -> list[str]:
    return [
        "能力键冲突：xuan:cli/commands#list（a, b）",
        "未解析的必需依赖：consumer 需要 xuan:cli/commands#missing",
    ]


def test_plugin_graph_unresolved_required_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """必需依赖无任何提供者时，文本诊断照常输出且退出码为 1。"""
    _patch_manifests(monkeypatch, [_fake_manifest("consumer", inject=("xuan:demo/ext#missing",))])

    result = runner.invoke(cli_app, ["plugin", "graph"])

    assert result.exit_code == 1, result.output
    assert "未解析" in result.output
    assert "consumer 需要 xuan:demo/ext#missing" in result.output


def test_plugin_graph_optional_unresolved_keeps_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """可选依赖未解析不构成错误，退出码仍为 0（避免把可选缺失当成装配失败）。"""
    consumer = parse_manifest(
        {
            "id": "consumer",
            "version": "0.1.0",
            "inject": [{"key": "xuan:demo/ext#missing", "optional": True}],
        }
    )
    _patch_manifests(monkeypatch, [consumer])

    result = runner.invoke(cli_app, ["plugin", "graph"])

    assert result.exit_code == 0, result.output
    assert "诊断摘要" not in result.output


def test_plugin_graph_json_carries_status_and_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON 模式：冲突与未解析进入 status/diagnostics，同时保留结构化数组。"""
    _patch_manifests(monkeypatch, _conflict_and_missing_manifests())

    result = runner.invoke(cli_app, ["plugin", "graph", "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["diagnostics"] == _expected_diagnostics()
    assert len(payload["conflicts"]) == 1
    assert payload["unresolved"] == [{"consumer": "consumer", "key": "xuan:cli/commands#missing"}]


def test_plugin_graph_dot_appends_diagnostics_as_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    """DOT 模式：诊断图仍是合法 DOT，摘要以 ``//`` 注释追加在末尾。"""
    _patch_manifests(monkeypatch, _conflict_and_missing_manifests())

    result = runner.invoke(cli_app, ["plugin", "graph", "--dot"])

    assert result.exit_code == 1, result.output
    assert result.output.startswith("digraph xuan {")
    assert "// 诊断摘要（2 项，退出码 1）" in result.output
    for line in _expected_diagnostics():
        assert f"// {line}" in result.output


def test_plugin_graph_three_modes_share_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """文本/JSON/DOT 三种输出必须携带同一份诊断摘要，且退出码一致。"""
    _patch_manifests(monkeypatch, _conflict_and_missing_manifests())

    text = runner.invoke(cli_app, ["plugin", "graph"])
    json_result = runner.invoke(cli_app, ["plugin", "graph", "--json"])
    dot_result = runner.invoke(cli_app, ["plugin", "graph", "--dot"])

    assert [text.exit_code, json_result.exit_code, dot_result.exit_code] == [1, 1, 1]
    payload = json.loads(json_result.output)
    assert payload["diagnostics"] == _expected_diagnostics()
    for line in payload["diagnostics"]:
        assert line in text.output
        assert f"// {line}" in dot_result.output


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
    assert {manifest.id for manifest in manifests} == {"xs.commands", "xuan.build", "xuan.ffi"}


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
    assert payload["order"] == ["xuan.build", "xs.commands", "xuan.ffi"]
    assert len(payload["domains"]) == 1
    assert set(payload["domains"][0]["nodes"]) == {"xuan.build", "xs.commands", "xuan.ffi"}
    assert payload["inactive"] == []
    assert len(payload["provides"]) == 20


def test_apply_activates_builtin_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实装配：内置插件 ACTIVE、16 条命令登记、逐键对账一致。"""
    monkeypatch.chdir(REPO_ROOT)
    result = runner.invoke(cli_app, ["plugin", "apply", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["applied"] == ["xuan.build", "xs.commands", "xuan.ffi"]
    assert payload["commands"] == [spec.name for spec in COMMAND_SPECS]
    assert payload["reconciled"]["missing"] == []
    assert payload["reconciled"]["unexpected"] == []
    assert len(payload["reconciled"]["matched"]) == 20


def test_apply_reports_missing_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """清单有节点但无运行时实现时显式失败，不伪装成装配成功。"""
    _patch_manifests(monkeypatch, [_fake_manifest("external.plugin", provide=("xuan:demo/ext#one",))])
    result = runner.invoke(cli_app, ["plugin", "apply"])
    assert result.exit_code == 1
    assert "缺少运行时实现" in result.output
    assert "external.plugin" in result.output


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


# ─── P2-18：why 子命令 ─────────────────────────────────────────────────


def test_plugin_why_active_reports_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """已生效插件：给出装配位置与已解析的依赖，退出码 0。"""
    monkeypatch.chdir(REPO_ROOT)

    result = runner.invoke(cli_app, ["plugin", "why", "xs.commands"])

    assert result.exit_code == 0, result.output
    assert "已生效" in result.output
    assert "装配顺序第 2" in result.output
    assert "xuan:build/backend#dispatch" in result.output


def test_plugin_why_unknown_id_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """未知 id 必须显式失败，并列出工作区内可用插件。"""
    monkeypatch.chdir(REPO_ROOT)

    result = runner.invoke(cli_app, ["plugin", "why", "nope"])

    assert result.exit_code == 1, result.output
    assert "未找到插件" in result.output
    assert "xs.commands" in result.output


def test_plugin_why_inactive_names_disabled_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺失键的唯一声明者被 disabled 排除时，必须点名该来源而不是只说“缺失”。"""
    key = "xuan:demo/ext#one"
    consumer = _fake_manifest("consumer", inject=(key,))
    provider = PluginManifest(
        id="provider",
        version="0.1.0",
        provide=(ProvideDecl(key=key),),
        disabled=True,
        source="tools/provider/pyproject.toml",
    )
    _patch_manifests(monkeypatch, [consumer, provider])

    result = runner.invoke(cli_app, ["plugin", "why", "consumer"])

    assert result.exit_code == 1, result.output
    assert "未生效" in result.output
    assert key in result.output
    assert "已禁用来源：provider" in result.output


def test_plugin_why_chain_reaches_root_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """提供者自身失活时，因果链必须下钻到真正缺键的那一层。"""
    upstream = "xuan:demo/a#b"
    root_key = "xuan:demo/x#y"
    consumer = _fake_manifest("a", inject=(upstream,))
    provider = _fake_manifest("b", provide=(upstream,), inject=(root_key,))
    _patch_manifests(monkeypatch, [consumer, provider])

    result = runner.invoke(cli_app, ["plugin", "why", "a"])

    assert result.exit_code == 1, result.output
    assert upstream in result.output
    assert f"b 缺少 {root_key}" in result.output
    assert "无提供者" in result.output


def test_plugin_why_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON 形态：状态、缺失键、候选提供者与因果链结构化输出。"""
    upstream = "xuan:demo/a#b"
    root_key = "xuan:demo/x#y"
    consumer = _fake_manifest("a", inject=(upstream,))
    provider = _fake_manifest("b", provide=(upstream,), inject=(root_key,))
    _patch_manifests(monkeypatch, [consumer, provider])

    result = runner.invoke(cli_app, ["plugin", "why", "a", "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "inactive"
    assert payload["id"] == "a"
    missing = payload["missing"][0]
    assert missing["key"] == upstream
    assert missing["providers"] == [{"id": "b", "state": "inactive"}]
    assert missing["chain"][0]["plugin"] == "b"
    assert missing["chain"][0]["key"] == root_key


def test_plugin_why_optional_missing_is_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """可选依赖未提供不构成失活：状态仍为 ACTIVE，退出码 0。"""
    optional_key = "xuan:demo/opt#one"
    manifest = PluginManifest(
        id="solo",
        version="0.1.0",
        inject=(InjectDecl(key=optional_key, optional=True),),
    )
    _patch_manifests(monkeypatch, [manifest])

    result = runner.invoke(cli_app, ["plugin", "why", "solo"])

    assert result.exit_code == 0, result.output
    assert "已生效" in result.output
    assert optional_key in result.output
    assert "可选" in result.output


def test_plugin_why_reports_unsolvable_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """求解无解（冲突/环）时无法给出单插件因果链，须以 blocked 收口。"""
    key = "xuan:demo/clash#one"
    _patch_manifests(monkeypatch, [_fake_manifest("a", provide=(key,)), _fake_manifest("b", provide=(key,))])

    result = runner.invoke(cli_app, ["plugin", "why", "a", "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert payload["kind"] == "conflict"
