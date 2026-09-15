# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_parse_args.py（断言语义零削弱）。

适配点：上游 ``compose._parse_args([...])``（引擎方法 + 单例注册表）改为
分层后的 ``xuan_compose.cli.parser.parse_args(engine, [...])``——parser
不再是引擎方法，argv 解析在 CLI 层完成，结果回填 ``engine.global_args``。
"""

import os
import unittest
from unittest import mock

from xuan_compose.cli.parser import parse_args
from xuan_compose.engine import ComposeEngine


class TestParseArgs(unittest.TestCase):
    def test_compose_env_files(self) -> None:
        engine = ComposeEngine()

        with mock.patch.dict(
            os.environ,
            {"COMPOSE_ENV_FILES": ".env.default,.env.override"},
        ):
            args = parse_args(engine, ["--version"])

        self.assertEqual(args.env_file, [".env.default", ".env.override"])

    def test_env_file_flag_overrides_compose_env_files(self) -> None:
        engine = ComposeEngine()

        with mock.patch.dict(
            os.environ,
            {"COMPOSE_ENV_FILES": ".env.default,.env.override"},
        ):
            args = parse_args(
                engine, ["--env-file", ".env.cli", "--version"]
            )

        self.assertEqual(args.env_file, [".env.cli"])
