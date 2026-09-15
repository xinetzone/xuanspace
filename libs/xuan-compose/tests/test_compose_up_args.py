# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_compose_up_args.py（断言语义零削弱）。

T5 曾登记本文件延 T7，T7 通读后确认被测对象是 argparse（无 parser 不
可落地），改登记延 T8；本批随显式 parser 注册表交付。适配点：
``podman_compose._parse_args(...)`` →
``parse_args(ComposeEngine(), ...)``。
"""

import unittest

from xuan_compose.cli.parser import parse_args
from xuan_compose.engine import ComposeEngine


class TestComposeUpArgs(unittest.TestCase):
    def test_no_attach_can_be_repeated(self) -> None:
        args = parse_args(
            ComposeEngine(),
            [
                "up",
                "--no-attach",
                "db",
                "--no-attach",
                "cache",
            ],
        )

        self.assertEqual(args.no_attach, ["db", "cache"])

    def test_no_attach_defaults_to_empty_list(self) -> None:
        args = parse_args(ComposeEngine(), ["up"])

        self.assertEqual(args.no_attach, [])
