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
from xuan_compose.errors import PodmanComposeError


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

    def test_wait_implies_detach(self) -> None:
        args = parse_args(ComposeEngine(), ["up", "--wait"])

        self.assertTrue(args.wait)
        self.assertTrue(args.detach)

    def test_wait_allows_explicit_detach(self) -> None:
        args = parse_args(ComposeEngine(), ["up", "--detach", "--wait"])

        self.assertTrue(args.detach)

    def test_wait_rejects_attached_exit_options(self) -> None:
        incompatible_options = [
            "--abort-on-container-exit",
            "--abort-on-container-failure",
            "--exit-code-from=app",
        ]

        for option in incompatible_options:
            with self.subTest(option=option):
                with self.assertRaisesRegex(PodmanComposeError, "--wait cannot be combined"):
                    parse_args(ComposeEngine(), ["up", "--wait", option])

    def test_wait_error_lists_all_incompatible_options(self) -> None:
        with self.assertRaises(PodmanComposeError) as error:
            parse_args(
                ComposeEngine(),
                [
                    "up",
                    "--wait",
                    "--abort-on-container-exit",
                    "--abort-on-container-failure",
                    "--exit-code-from=app",
                ],
            )

        self.assertEqual(
            str(error.exception),
            "--wait cannot be combined with --abort-on-container-exit, "
            "--abort-on-container-failure or --exit-code-from",
        )

    def test_wait_timeout_must_be_non_negative(self) -> None:
        for command in ("up", "start"):
            with self.subTest(command=command):
                with self.assertRaisesRegex(PodmanComposeError, "non-negative integer"):
                    parse_args(ComposeEngine(), [command, "--wait-timeout", "-1"])

    def test_zero_wait_timeout_is_allowed(self) -> None:
        args = parse_args(ComposeEngine(), ["up", "--wait", "--wait-timeout", "0"])

        self.assertEqual(args.wait_timeout, 0)
