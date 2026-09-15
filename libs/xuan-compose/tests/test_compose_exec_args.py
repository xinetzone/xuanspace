# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_compose_exec_args.py（2 例，断言零削弱）。

适配点：``podman_compose.compose_exec_args`` →
``xuan_compose.translate.run_args.compose_exec_args``。
"""

import argparse
import unittest

from xuan_compose.translate.run_args import compose_exec_args


class TestComposeExecArgs(unittest.TestCase):
    def test_minimal(self) -> None:
        cnt = get_minimal_container()
        args = get_minimal_args()

        result = compose_exec_args(cnt, "container_name", args)
        expected = ["--interactive", "--tty", "container_name"]
        self.assertEqual(result, expected)

    def test_additional_env_value_equals(self) -> None:
        cnt = get_minimal_container()
        args = get_minimal_args()
        args.env = ["key=valuepart1=valuepart2"]

        result = compose_exec_args(cnt, "container_name", args)
        expected = [
            "--interactive",
            "--tty",
            "--env",
            "key=valuepart1=valuepart2",
            "container_name",
        ]
        self.assertEqual(result, expected)


def get_minimal_container() -> dict:
    return {}


def get_minimal_args() -> argparse.Namespace:
    return argparse.Namespace(
        T=None,
        cnt_command=None,
        env=None,
        privileged=None,
        user=None,
        workdir=None,
    )
