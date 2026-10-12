# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_normalize_service.py（断言零削弱）。

Windows 路径适配：sub_dir 用例的期望值与生产代码一样逐行保留上游
``os.path.join``（build.context 分支另加 ``rstrip("/")``）的本机路径
语义——期望侧改用同一 ``os.path`` 原语表达：POSIX 下与上游硬编码
``./sub_dir/...`` 字符串逐字节一致，Windows 下得到等价的 ``.\\sub_dir\\...``。
"""

import os
import unittest
from typing import Any

from parameterized import parameterized

from xuan_compose.errors import PodmanComposeError
from xuan_compose.normalize import normalize_service


class TestNormalizeService(unittest.TestCase):
    @parameterized.expand(
        [
            ({"test": "test"}, {"test": "test"}),
            ({"build": "."}, {"build": {"context": "."}}),
            ({"build": "./dir-1"}, {"build": {"context": "./dir-1"}}),
            ({"build": {"context": "./dir-1"}}, {"build": {"context": "./dir-1"}}),
            (
                {"build": {"dockerfile": "dockerfile-1"}},
                {"build": {"dockerfile": "dockerfile-1"}},
            ),
            (
                {"build": {"context": "./dir-1", "dockerfile": "dockerfile-1"}},
                {"build": {"context": "./dir-1", "dockerfile": "dockerfile-1"}},
            ),
            (
                {"build": {"additional_contexts": ["ctx=../ctx", "ctx2=../ctx2"]}},
                {"build": {"additional_contexts": ["ctx=../ctx", "ctx2=../ctx2"]}},
            ),
            (
                {"build": {"additional_contexts": {"ctx": "../ctx", "ctx2": "../ctx2"}}},
                {"build": {"additional_contexts": ["ctx=../ctx", "ctx2=../ctx2"]}},
            ),
        ]
    )
    def test_simple(self, input: dict[str, Any], expected: dict[str, Any]) -> None:
        self.assertEqual(normalize_service(input), expected)

    @parameterized.expand(
        [
            ({"test": "test"}, {"test": "test"}),
            (
                {"build": "."},
                {"build": {"context": os.path.join("./sub_dir", ".").rstrip("/")}},
            ),
            (
                {"build": "./dir-1"},
                {"build": {"context": os.path.join("./sub_dir", "dir-1").rstrip("/")}},
            ),
            (
                {"build": {"context": "./dir-1"}},
                {"build": {"context": os.path.join("./sub_dir", "dir-1").rstrip("/")}},
            ),
            (
                {"build": {"dockerfile": "dockerfile-1"}},
                {
                    "build": {
                        "context": os.path.join("./sub_dir", "").rstrip("/"),
                        "dockerfile": "dockerfile-1",
                    }
                },
            ),
            (
                {"build": {"context": "./dir-1", "dockerfile": "dockerfile-1"}},
                {
                    "build": {
                        "context": os.path.join("./sub_dir", "dir-1").rstrip("/"),
                        "dockerfile": "dockerfile-1",
                    }
                },
            ),
            (
                {"volumes": ["./nested/relative:/mnt", "../dir-in-parent:/mnt", "..:/mnt", ".:/mnt"]},
                {
                    "volumes": [
                        os.path.join("./sub_dir", "./nested/relative:/mnt"),
                        os.path.join("./sub_dir", "../dir-in-parent:/mnt"),
                        os.path.join("./sub_dir", "..:/mnt"),
                        os.path.join("./sub_dir", ".:/mnt"),
                    ]
                },
            ),
            (
                {
                    "volumes": [
                        {"type": "bind", "source": "./nested/relative", "target": "/mnt"}
                    ]
                },
                {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": os.path.join("./sub_dir", "./nested/relative"),
                            "target": "/mnt",
                        }
                    ]
                },
            ),
            (
                {"env_file": "./.env"},
                {"env_file": [os.path.join("./sub_dir", "./.env")]},
            ),
            (
                {"env_file": ["./.env", "../shared.env"]},
                {
                    "env_file": [
                        os.path.join("./sub_dir", "./.env"),
                        os.path.join("./sub_dir", "../shared.env"),
                    ]
                },
            ),
            (
                {"env_file": [{"path": "./.env", "required": False}]},
                {
                    "env_file": [
                        {"path": os.path.join("./sub_dir", "./.env"), "required": False}
                    ]
                },
            ),
        ]
    )
    def test_normalize_service_with_sub_dir(
        self, input: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        self.assertEqual(normalize_service(input, sub_dir="./sub_dir"), expected)

    @parameterized.expand(
        [
            ([], []),
            (["sh"], ["sh"]),
            (["sh", "-c", "date"], ["sh", "-c", "date"]),
            ("sh", "sh"),
            ("sleep infinity", "sleep infinity"),
            (
                "bash -c 'sleep infinity'",
                "bash -c 'sleep infinity'",
            ),
        ]
    )
    def test_command_like(
        self, input: list[str] | str, expected: list[str] | str
    ) -> None:
        for key in ["command", "entrypoint"]:
            input_service = {}
            input_service[key] = input

            expected_service = {}
            expected_service[key] = expected
            self.assertEqual(normalize_service(input_service), expected_service)

    @parameterized.expand(
        [
            ("secrets_string", {"secrets": "my_secret"}, {"secrets": ["my_secret"]}),
            ("secrets_list", {"secrets": ["my_secret"]}, {"secrets": ["my_secret"]}),
            (
                "secrets_list_of_dicts",
                {"secrets": [{"source": "my_secret", "target": "ENV_VAR"}]},
                {"secrets": [{"source": "my_secret", "target": "ENV_VAR"}]},
            ),
        ]
    )
    def test_secrets_normalization(
        self, test_name: str, input_service: dict[str, Any], expected_service: dict[str, Any]
    ) -> None:
        self.assertEqual(normalize_service(input_service), expected_service)

    def test_secrets_dict_raises(self) -> None:
        with self.assertRaises(PodmanComposeError) as context:
            normalize_service({"secrets": {"my_secret": {"source": "my_secret"}}})
        self.assertEqual("ERROR: secrets must be a list, not a dict", str(context.exception))

    def test_build_secrets_dict_raises(self) -> None:
        with self.assertRaises(PodmanComposeError) as context:
            normalize_service(
                {"build": {"context": ".", "secrets": {"my_secret": {"source": "my_secret"}}}}
            )
        self.assertEqual(
            "ERROR: build.secrets must be a list, not a dict", str(context.exception)
        )

    def test_volumes_none_raises(self) -> None:
        with self.assertRaises(PodmanComposeError) as context:
            normalize_service({"volumes": None})
        self.assertEqual("ERROR: volumes must be a list", str(context.exception))
