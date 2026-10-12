# SPDX-License-Identifier: GPL-2.0-only
"""TR-9.1：argparse 参数表与上游 pinned 版本零差异对拍（prog 除外）。

快照 ``tests/snapshots/cli_parity.json`` 由
``tests/generate_parity_snapshots.py`` 从 vendor 上游
（containers/podman-compose，commit 记录于快照 meta）程序化生成；
本测试不依赖 vendor 检出，只读快照。

对比维度（AC-4）：option_strings / action / default / nargs /
required，另含 dest/const/choices/metavar/type/help 文案。
"""

import json
import unittest
from pathlib import Path

from tests.parity_lib import serialize_parser_tree
from xuan_compose.cli.parser import SYSTEMD_DESC, build_parser

SNAPSHOT = Path(__file__).parent / "snapshots" / "cli_parity.json"

#: 本库相对上游**有意新增**的全局参数白名单（option 主名 → 差异登记号）。
#: 对拍前从 current 剔除，使上游参数表仍维持逐位置零差异硬约束；
#: ``test_xuan_extensions_actually_exist`` 守卫白名单与代码不漂移。
#: 新增条目必须同时在 doc/dev/parity/differences.md 登记。
XUAN_GLOBAL_EXTENSIONS: dict[str, str] = {
    "--no-machine-start": "D16",
}


def _load_snapshot() -> dict:
    with SNAPSHOT.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _strip_xuan_extensions(options: list[dict]) -> list[dict]:
    return [
        action
        for action in options
        if not any(s in XUAN_GLOBAL_EXTENSIONS for s in action["option_strings"])
    ]


class TestCliParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _load_snapshot()
        cls.current = serialize_parser_tree(
            build_parser({"systemd": SYSTEMD_DESC})
        )

    def test_snapshot_pinned_to_expected_upstream_commit(self) -> None:
        # 防快照被静默换源：基线必须是 T1-T8 全程对照的 e3df104
        assert self.snapshot["meta"]["commit"].startswith("e3df104")
        assert self.snapshot["meta"]["source"] == "containers/podman-compose"

    def test_command_order_matches_upstream(self) -> None:
        assert self.current["command_order"] == self.snapshot["command_order"]

    def test_global_options_zero_diff(self) -> None:
        upstream = self.snapshot["global"]["options"]
        # 剔除 D16 等登记在案的本库扩展后，上游参数表必须逐位置零差异
        current = _strip_xuan_extensions(self.current["global"]["options"])
        self._assert_option_lists_equal(upstream, current, context="<global>")

    def test_xuan_extensions_actually_exist(self) -> None:
        # 防白名单漂移：登记的每个扩展 option 必须真实存在且带 help 文案
        present = {
            option
            for action in self.current["global"]["options"]
            for option in action["option_strings"]
        }
        for option in XUAN_GLOBAL_EXTENSIONS:
            assert option in present, f"白名单扩展 {option} 未在 parser 中找到"
            action = next(
                a
                for a in self.current["global"]["options"]
                if option in a["option_strings"]
            )
            assert action["help"], f"扩展 {option} 缺少 help 文案"

    def test_every_subcommand_options_zero_diff(self) -> None:
        upstream_commands = self.snapshot["commands"]
        current_commands = self.current["commands"]
        assert set(current_commands) == set(upstream_commands)
        for name in self.snapshot["command_order"]:
            with self.subTest(command=name):
                self._assert_option_lists_equal(
                    upstream_commands[name]["options"],
                    current_commands[name]["options"],
                    context=name,
                )

    # ---- helpers -------------------------------------------------------

    def _assert_option_lists_equal(
        self, upstream: list[dict], current: list[dict], context: str
    ) -> None:
        """逐位置对比 action 表（顺序也是 argparse 行为的一部分）。"""
        if len(upstream) != len(current):
            self.fail(
                f"[{context}] action 数量不一致: "
                f"upstream={len(upstream)} current={len(current)}\n"
                f"  upstream only: "
                f"{[a['option_strings'] for a in upstream[len(current):]]}\n"
                f"  current only: "
                f"{[a['option_strings'] for a in current[len(upstream):]]}"
            )
        for idx, (up, cur) in enumerate(zip(upstream, current, strict=True)):
            if up != cur:
                diffs = {
                    key: (up.get(key), cur.get(key))
                    for key in (
                        "option_strings",
                        "dest",
                        "action",
                        "nargs",
                        "const",
                        "default",
                        "choices",
                        "required",
                        "metavar",
                        "type",
                        "help",
                    )
                    if up.get(key) != cur.get(key)
                }
                self.fail(
                    f"[{context}] 第 {idx} 个 action 不一致"
                    f"（upstream={up['option_strings']!r}）:\n"
                    + "\n".join(f"  {k}: upstream={a!r} current={b!r}" for k, (a, b) in diffs.items())
                )
