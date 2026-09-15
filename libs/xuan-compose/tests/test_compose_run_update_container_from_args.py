# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_compose_run_update_container_from_args.py（6 例，断言零削弱）。

适配点：
- ``podman_compose.compose_run_update_container_from_args`` →
  ``xuan_compose.translate.run_args.compose_run_update_container_from_args``。
- 上游真实实例化 ``PodmanCompose()``（仅用到 project_name + format_name）；
  引擎层尚未落地（T6），按 tasks.md 允许的"轻量构造改用最小 stub"，
  本文件定义等价 format_name 行为的 MinimalCompose 夹具。
"""

import argparse
import unittest

from xuan_compose.translate.run_args import compose_run_update_container_from_args


class MinimalCompose:
    """只承载 run 参数翻译所需的 project_name/format_name。"""

    def __init__(self, project_name: str = "test_project") -> None:
        self.project_name = project_name

    def format_name(self, *parts: str) -> str:
        return "_".join([self.project_name, *parts])


class TestComposeRunUpdateContainerFromArgs(unittest.TestCase):
    def test_minimal(self) -> None:
        cnt = get_minimal_container()
        compose = get_minimal_compose()
        args = get_minimal_args()

        compose_run_update_container_from_args(compose, cnt, args)

        expected_cnt = {"name": "default_name", "tty": True}
        self.assertEqual(cnt, expected_cnt)

    def test_additional_env_value_equals(self) -> None:
        cnt = get_minimal_container()
        compose = get_minimal_compose()
        args = get_minimal_args()
        args.env = ["key=valuepart1=valuepart2"]

        compose_run_update_container_from_args(compose, cnt, args)

        expected_cnt = {
            "environment": {
                "key": "valuepart1=valuepart2",
            },
            "name": "default_name",
            "tty": True,
        }
        self.assertEqual(cnt, expected_cnt)

    def test_publish_ports(self) -> None:
        cnt = get_minimal_container()
        compose = get_minimal_compose()
        args = get_minimal_args()
        args.publish = ["1111", "2222:2222"]

        compose_run_update_container_from_args(compose, cnt, args)

        expected_cnt = {
            "name": "default_name",
            "ports": ["1111", "2222:2222"],
            "tty": True,
        }
        self.assertEqual(cnt, expected_cnt)

    def test_container_name_from_compose(self) -> None:
        cnt = {"container_name": "compose_custom_name"}
        compose = get_minimal_compose()
        args = get_minimal_args()
        args.name = None

        compose_run_update_container_from_args(compose, cnt, args)

        expected_cnt = {
            "container_name": "compose_custom_name",
            "name": "compose_custom_name",
            "tty": True,
        }
        self.assertEqual(cnt, expected_cnt)

    def test_cli_name_overrides_container_name(self) -> None:
        cnt = {"container_name": "compose_custom_name"}
        compose = get_minimal_compose()
        args = get_minimal_args()
        args.name = "cli_override_name"

        compose_run_update_container_from_args(compose, cnt, args)

        expected_cnt = {
            "container_name": "compose_custom_name",
            "name": "cli_override_name",
            "tty": True,
        }
        self.assertEqual(cnt, expected_cnt)

    def test_fallback_to_generated_name(self) -> None:
        cnt = get_minimal_container()
        compose = get_minimal_compose()
        args = get_minimal_args()
        args.name = None

        compose_run_update_container_from_args(compose, cnt, args)

        self.assertTrue(cnt["name"].startswith("test_project_test_service_tmp"))
        self.assertEqual(cnt["tty"], True)


def get_minimal_container() -> dict:
    return {}


def get_minimal_compose() -> MinimalCompose:
    return MinimalCompose("test_project")


def get_minimal_args() -> argparse.Namespace:
    return argparse.Namespace(
        T=None,
        cnt_command=None,
        entrypoint=None,
        env=None,
        name="default_name",
        rm=None,
        service="test_service",
        publish=None,
        service_ports=None,
        user=None,
        volume=None,
        workdir=None,
    )
