# SPDX-License-Identifier: GPL-2.0-only
"""TR-9.1 补充：命令集合 / 注册顺序 / help 与 description 快照对等。

``test_cli_parity.py`` 负责参数表层面对拍；本文件负责命令表面
（24 命令名、@cmd_run 装饰器注册顺序、单行 help 与 systemd 多行
description 切分结果），并交叉校验 parser 注册表与 handler 注册表
的键集合严格相等。
"""

import json
import unittest
from pathlib import Path

from tests.parity_lib import serialize_parser_tree
from xuan_compose.cli.parser import COMMAND_HELP, COMMAND_PARSERS, SYSTEMD_DESC, build_parser
from xuan_compose.commands import COMMAND_HANDLERS

SNAPSHOT = Path(__file__).parent / "snapshots" / "cli_parity.json"


class TestCommandSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with SNAPSHOT.open(encoding="utf-8") as fh:
            cls.snapshot = json.load(fh)
        cls.current = serialize_parser_tree(build_parser({"systemd": SYSTEMD_DESC}))

    def test_twenty_four_real_commands_plus_help_pseudo(self) -> None:
        # 沿用 T7/T8 计数勘误口径：spec 文字"22"为笔误，实为 24 + help
        order = self.snapshot["command_order"]
        assert order[0] == "help"
        assert len(order) == 25
        real = order[1:]
        assert len(real) == 24

    def test_command_set_and_order_equal_to_snapshot(self) -> None:
        assert self.current["command_order"] == self.snapshot["command_order"]

    def test_help_and_description_texts_match_snapshot(self) -> None:
        for name in self.snapshot["command_order"]:
            with self.subTest(command=name):
                up = self.snapshot["commands"][name]
                cur = self.current["commands"][name]
                assert cur["help"] == up["help"], name
                assert cur["description"] == up["description"], name

    def test_registry_keys_match_command_surface(self) -> None:
        real_commands = set(self.snapshot["command_order"]) - {"help"}
        assert set(COMMAND_PARSERS) == real_commands
        assert set(COMMAND_HANDLERS) == real_commands
        # help 表 24 键（systemd 为运行时派生，无占位空串）
        assert set(COMMAND_HELP) == real_commands
        assert COMMAND_HELP["systemd"]

    def test_command_order_preserved_in_registry_iteration(self) -> None:
        # 显式注册表迭代序 = 上游 @cmd_run 装饰器出现序（help 输出顺序依赖）
        assert list(COMMAND_PARSERS) == self.snapshot["command_order"][1:]
