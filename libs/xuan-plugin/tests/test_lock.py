"""``xuan.plugins.lock`` 派生视图测试（P3-1）。

验收要点：同输入同输出、``plugins`` 按 id 排序、来源路径（``source``）不参与
摘要、渲染文本逐字节可重现、差异信息能定位到具体 id。
"""

import json
import subprocess
import sys
from pathlib import Path

from xuan_plugin.lock import build_lock, lock_diff, plugin_digest, render_lock
from xuan_plugin.manifest import parse_manifest

TESTS_DIR = Path(__file__).resolve().parent


def _manifest(plugin_id: str, *, priority: int = 0, source: str = "a/pyproject.toml"):
    return parse_manifest(
        {
            "id": plugin_id,
            "version": "0.1.0",
            "priority": priority,
            "provide": [{"key": "xuan:demo/api#dispatch"}],
        },
        source=source,
    )


def test_digest_ignores_source_and_is_deterministic() -> None:
    """来源路径是诊断标记，不得影响契约摘要；同一清单重复计算相等。"""
    a = _manifest("a.plugin", source="tools/a/pyproject.toml")
    b = _manifest("a.plugin", source="libs/b/pyproject.toml")

    assert plugin_digest(a) == plugin_digest(b)
    assert plugin_digest(a) == plugin_digest(a)
    assert plugin_digest(a).startswith("sha256:")


def test_digest_changes_with_contract_field() -> None:
    """契约字段变化（priority）必须改变摘要，否则 H4 无法发现漂移。"""
    assert plugin_digest(_manifest("a.plugin")) != plugin_digest(_manifest("a.plugin", priority=5))


def test_build_lock_sorted_by_id() -> None:
    """``plugins`` 按 id 排序，保证不同发现顺序产出同一视图。"""
    lock = build_lock([_manifest("b.two"), _manifest("a.one")])

    assert lock["schema"] == 1
    assert [item["id"] for item in lock["plugins"]] == ["a.one", "b.two"]


def test_render_lock_round_trips_and_ends_with_newline() -> None:
    """渲染文本可被 JSON 解析回同一视图，且以单个换行结尾。"""
    manifests = [_manifest("a.one"), _manifest("b.two")]
    text = render_lock(manifests)

    assert json.loads(text) == build_lock(manifests)
    assert text.endswith("}\n")


def test_render_lock_reproducible_across_processes() -> None:
    """两个独立进程对同一输入渲染出的文本逐字节相同（无地址/哈希随机影响）。"""
    snippet = (
        "import sys\n"
        "from xuan_plugin.lock import render_lock\n"
        "from xuan_plugin.manifest import parse_manifest\n"
        "m = parse_manifest({'id': 'a.one', 'version': '0.1.0'}, source='x/pyproject.toml')\n"
        "sys.stdout.write(render_lock([m]))\n"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=TESTS_DIR,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        ).stdout
        for _ in range(2)
    ]

    assert outputs[0] == outputs[1]


def test_lock_diff_reports_missing_and_stale_ids() -> None:
    """差异信息须逐 id 指出 digest 过期与多出插件。"""
    expected = build_lock([_manifest("a.one"), _manifest("b.two")])
    actual = {
        "schema": 1,
        "plugins": [
            {"id": "a.one", "digest": "sha256:deadbeef"},
            {"id": "c.extra", "digest": "sha256:feedface"},
        ],
    }

    diff = lock_diff(expected, actual)

    assert "a.one digest" in diff
    assert "缺少插件 b.two" in diff
    assert "多出插件 c.extra" in diff
