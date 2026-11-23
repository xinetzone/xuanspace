"""``scripts/check_plugin_contract.py`` 的契约断言回归（P3-2 验收）。

覆盖 6 类反例：H1 缺失提供者、H2 同键多 single、H2 multi 同 priority、
H3 模块级可变全局、H4 lock 与重算不一致、H5 constraint 方法缺失；
另含两条正例：无 lock 时 H4 跳过、lock 与声明一致时全部通过。
"""

import importlib.util
import json
from pathlib import Path

import pytest
from xuan_plugin.lock import plugin_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_plugin_contract.py"

PROVIDE = """
[[tool.xuan.plugin.provide]]
key = "xuan:build/backend#dispatch"
"""
PROVIDE_MULTI = """
[[tool.xuan.plugin.provide]]
key = "xuan:build/backend#dispatch"
mode = "multi"
"""
INJECT = """
[[tool.xuan.plugin.inject]]
key = "xuan:build/backend#dispatch"
"""


@pytest.fixture(scope="module")
def checker():
    """按文件路径加载脚本（scripts/ 不是包，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location("check_plugin_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_plugin(
    root: Path,
    dirname: str,
    plugin_id: str,
    *,
    priority: int = 0,
    provide: str = "",
    inject: str = "",
    module: str = "def build():\n    pass\n",
) -> None:
    """在 ``root/dirname`` 造一个最小插件工程：pyproject 清单 + 入口模块。"""
    pyproject = f'''[project]
name = "{plugin_id}"
version = "0.1.0"

[project.entry-points."xuan.plugins"]
"{plugin_id}" = "plugin_mod"

[tool.xuan.plugin]
id = "{plugin_id}"
version = "0.1.0"
priority = {priority}
{provide}{inject}'''
    _write(root / dirname / "pyproject.toml", pyproject)
    _write(root / dirname / "src" / "plugin_mod.py", module)


def _run(checker, root: Path, capsys) -> tuple[int, str]:
    code = checker.main(["--root", str(root)])
    return code, capsys.readouterr().out


def test_h1_missing_provider_is_reported(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "consumer", "a.consumer", inject=INJECT)

    code, out = _run(checker, tmp_path, capsys)

    assert code == 1
    assert "H1 可解性" in out
    assert "a.consumer" in out
    assert "xuan:build/backend#dispatch" in out


def test_h2_multiple_single_providers_is_reported(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "one", "a.one", provide=PROVIDE)
    make_plugin(tmp_path, "two", "b.two", priority=10, provide=PROVIDE)

    code, out = _run(checker, tmp_path, capsys)

    assert code == 1
    h2_lines = [line for line in out.splitlines() if "H2 隔离性" in line]
    assert len(h2_lines) == 1
    assert "single" in h2_lines[0]
    assert "a.one" in h2_lines[0] and "b.two" in h2_lines[0]


def test_h2_multi_with_duplicate_priority_is_reported(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "one", "a.one", priority=20, provide=PROVIDE_MULTI)
    make_plugin(tmp_path, "two", "b.two", priority=20, provide=PROVIDE_MULTI)

    code, out = _run(checker, tmp_path, capsys)

    assert code == 1
    h2_lines = [line for line in out.splitlines() if "H2 隔离性" in line]
    assert len(h2_lines) == 1
    assert "priority" in h2_lines[0] and "并列" in h2_lines[0]
    assert "a.one" in h2_lines[0] and "b.two" in h2_lines[0]


def test_h3_module_level_mutable_global_is_reported(checker, tmp_path, capsys) -> None:
    module = """REGISTRY = {}


def register(name):
    global REGISTRY
    REGISTRY[name] = None
"""
    make_plugin(tmp_path, "solo", "a.solo", provide=PROVIDE, module=module)

    code, out = _run(checker, tmp_path, capsys)

    assert code == 1
    h3_lines = [line for line in out.splitlines() if "H3 可撤销性" in line]
    assert len(h3_lines) == 2
    assert any("plugin_mod.py:1" in line for line in h3_lines)
    assert any("global" in line for line in h3_lines)


def test_h4_lock_digest_mismatch_is_reported(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "solo", "a.solo", provide=PROVIDE)
    _write(
        tmp_path / "xuan.plugins.lock",
        json.dumps({"schema": 1, "plugins": [{"id": "a.solo", "digest": "sha256:deadbeef"}]}, ensure_ascii=False),
    )

    code, out = _run(checker, tmp_path, capsys)

    assert code == 1
    h4_lines = [line for line in out.splitlines() if "H4 视图一致性" in line]
    assert len(h4_lines) == 1
    assert "a.solo" in h4_lines[0] and "digest" in h4_lines[0]


def test_h5_missing_constraint_method_is_reported(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "provider", "p.provider", provide=PROVIDE, module="def dispatch():\n    pass\n")
    make_plugin(
        tmp_path,
        "consumer",
        "c.consumer",
        inject=INJECT + 'constraint = "methods:build,dispatch"\n',
    )

    code, out = _run(checker, tmp_path, capsys)

    assert code == 1
    h5_lines = [line for line in out.splitlines() if "H5 结构兼容" in line]
    assert len(h5_lines) == 1
    assert "build" in h5_lines[0]
    assert "p.provider" in h5_lines[0]
    assert "dispatch" not in h5_lines[0].split("缺失方法：", 1)[1]


def test_valid_workspace_without_lock_passes(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "provider", "p.provider", provide=PROVIDE)
    make_plugin(tmp_path, "consumer", "c.consumer", inject=INJECT)

    code, out = _run(checker, tmp_path, capsys)

    assert code == 0
    assert "跳过" in out
    assert "全部通过" in out


def test_lock_consistent_with_declarations_passes(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "provider", "p.provider", provide=PROVIDE)
    make_plugin(tmp_path, "consumer", "c.consumer", inject=INJECT)
    plugins, issues = checker.load_plugins(tmp_path)
    assert issues == []
    lock = {
        "schema": 1,
        "plugins": [{"id": p.manifest.id, "digest": plugin_digest(p.manifest)} for p in plugins],
    }
    _write(tmp_path / "xuan.plugins.lock", json.dumps(lock, ensure_ascii=False, indent=2))

    code, out = _run(checker, tmp_path, capsys)

    assert code == 0
    assert "与清单重算一致" in out


def test_write_lock_generates_consistent_lock(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "provider", "p.provider", provide=PROVIDE)
    make_plugin(tmp_path, "consumer", "c.consumer", inject=INJECT)

    assert checker.main(["--root", str(tmp_path), "--write-lock"]) == 0
    capsys.readouterr()
    lock = json.loads((tmp_path / "xuan.plugins.lock").read_text(encoding="utf-8"))

    assert lock["schema"] == 1
    assert [item["id"] for item in lock["plugins"]] == ["c.consumer", "p.provider"]

    code, out = _run(checker, tmp_path, capsys)
    assert code == 0
    assert "与清单重算一致（2 个插件）" in out


def test_write_lock_is_idempotent(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "solo", "a.solo", provide=PROVIDE)
    assert checker.main(["--root", str(tmp_path), "--write-lock"]) == 0
    capsys.readouterr()

    assert checker.main(["--root", str(tmp_path), "--write-lock"]) == 0

    assert "已是目标状态" in capsys.readouterr().out


def test_lock_stale_after_manifest_change_is_reported(checker, tmp_path, capsys) -> None:
    make_plugin(tmp_path, "solo", "a.solo", provide=PROVIDE)
    assert checker.main(["--root", str(tmp_path), "--write-lock"]) == 0
    capsys.readouterr()
    make_plugin(tmp_path, "solo", "a.solo", priority=5, provide=PROVIDE)

    code, out = _run(checker, tmp_path, capsys)

    assert code == 1
    assert "H4 视图一致性" in out and "a.solo" in out
