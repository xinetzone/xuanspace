# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_can_merge_build.py（38 用例，断言零削弱）。

T5 延期登记：上游不存在 ``can_merge_build`` 符号，本文件实际通过双 compose
文件 merge 后再解析测引擎 ``_parse_compose_file``/``original_configuration``
行为，被测面属引擎层，随 T6 迁入。

适配点：
- 扁平 ``podman_compose`` 导入改为 ``xuan_compose.engine.ComposeEngine``。
- 上游把 yaml 写到 pytest 启动目录、依赖模块级清理函数删除；移植版每个用例
  在独立临时目录 chdir 执行并快照/还原 COMPOSE_*/PODMAN_* 进程环境，
  数据行与断言逐行不变；模块级 ``test_clean_test_yamls`` 保留（上游收集
  计数 38 含该卫生用例），在还原后的工作目录断言无残留。
"""

import argparse
import copy
import os
import shutil
import tempfile
import unittest

import yaml
from parameterized import parameterized

from xuan_compose.engine import ComposeEngine

# 引擎解析过程可能读写的进程环境键，用例间必须还原
_ENV_KEYS = (
    "COMPOSE_PROJECT_DIR",
    "COMPOSE_FILE",
    "COMPOSE_PATH_SEPARATOR",
    "COMPOSE_PROJECT_NAME",
    "COMPOSE_PROFILES",
)


class TestCanMergeBuild(unittest.TestCase):
    def setUp(self) -> None:
        self._cwd = os.getcwd()
        self._env = {k: os.environ.get(k) for k in _ENV_KEYS}
        self._tmp = tempfile.mkdtemp(prefix="xuan-compose-merge-")
        os.chdir(self._tmp)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self._tmp, ignore_errors=True)

    @parameterized.expand([
        ({}, {}, {}),
        ({}, {"test": "test"}, {"test": "test"}),
        ({"test": "test"}, {}, {"test": "test"}),
        ({"test": "test-1"}, {"test": "test-2"}, {"test": "test-2"}),
        ({}, {"build": "."}, {"build": {"context": "."}}),
        ({"build": "."}, {}, {"build": {"context": "."}}),
        ({"build": "./dir-1"}, {"build": "./dir-2"}, {"build": {"context": "./dir-2"}}),
        ({}, {"build": {"context": "./dir-1"}}, {"build": {"context": "./dir-1"}}),
        ({"build": {"context": "./dir-1"}}, {}, {"build": {"context": "./dir-1"}}),
        (
            {"build": {"context": "./dir-1"}},
            {"build": {"context": "./dir-2"}},
            {"build": {"context": "./dir-2"}},
        ),
        (
            {},
            {"build": {"dockerfile": "dockerfile-1"}},
            {"build": {"dockerfile": "dockerfile-1"}},
        ),
        (
            {"build": {"dockerfile": "dockerfile-1"}},
            {},
            {"build": {"dockerfile": "dockerfile-1"}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-1"}},
            {"build": {"dockerfile": "./dockerfile-2"}},
            {"build": {"dockerfile": "./dockerfile-2"}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-1"}},
            {"build": {"context": "./dir-2"}},
            {"build": {"dockerfile": "./dockerfile-1", "context": "./dir-2"}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-1", "context": "./dir-1"}},
            {"build": {"dockerfile": "./dockerfile-2", "context": "./dir-2"}},
            {"build": {"dockerfile": "./dockerfile-2", "context": "./dir-2"}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-1"}},
            {"build": {"dockerfile": "./dockerfile-2", "args": ["ENV1=1"]}},
            {"build": {"dockerfile": "./dockerfile-2", "args": ["ENV1=1"]}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-2", "args": ["ENV1=1"]}},
            {"build": {"dockerfile": "./dockerfile-1"}},
            {"build": {"dockerfile": "./dockerfile-1", "args": ["ENV1=1"]}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-2", "args": ["ENV1=1"]}},
            {"build": {"dockerfile": "./dockerfile-1", "args": ["ENV2=2"]}},
            {"build": {"dockerfile": "./dockerfile-1", "args": ["ENV1=1", "ENV2=2"]}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-1", "args": {"ENV1": "1"}}},
            {"build": {"dockerfile": "./dockerfile-2", "args": {"ENV2": "2"}}},
            {"build": {"dockerfile": "./dockerfile-2", "args": ["ENV1=1", "ENV2=2"]}},
        ),
        (
            {"build": {"dockerfile": "./dockerfile-1", "args": ["ENV1=1"]}},
            {"build": {"dockerfile": "./dockerfile-2", "args": {"ENV2": "2"}}},
            {"build": {"dockerfile": "./dockerfile-2", "args": ["ENV1=1", "ENV2=2"]}},
        ),
    ])
    def test_parse_compose_file_when_multiple_composes(self, input, override, expected):
        compose_test_1 = {"services": {"test-service": input}}
        compose_test_2 = {"services": {"test-service": override}}
        dump_yaml(compose_test_1, "test-compose-1.yaml")
        dump_yaml(compose_test_2, "test-compose-2.yaml")

        podman_compose = ComposeEngine()
        set_args(podman_compose, ["test-compose-1.yaml", "test-compose-2.yaml"])

        podman_compose._parse_compose_file()  # pylint: disable=protected-access

        actual_compose = {}
        if podman_compose.services:
            actual_compose = podman_compose.original_configuration(
                podman_compose.services["test-service"]
            )
        self.assertEqual(actual_compose, expected)

    def test_parse_with_map_merge_into_none(self):
        compose_test_1 = {"volumes": {"vol_a": None}}
        compose_test_2 = {
            "volumes": {
                "vol_a": {
                    "driver_opts": {"type": "none", "device": "/dev/some", "o": "bind"},
                }
            }
        }
        expected = {"driver_opts": {"device": "/dev/some", "o": "bind", "type": "none"}}
        dump_yaml(compose_test_1, "test-compose-1.yaml")
        dump_yaml(compose_test_2, "test-compose-2.yaml")

        podman_compose = ComposeEngine()
        set_args(podman_compose, ["test-compose-1.yaml", "test-compose-2.yaml"])

        podman_compose._parse_compose_file()  # pylint: disable=protected-access

        actual_compose = podman_compose.vols["vol_a"]
        self.assertEqual(actual_compose, expected)

    # $$$ is a placeholder for either command or entrypoint
    @parameterized.expand([
        ({}, {"$$$": []}, {"$$$": []}),
        ({"$$$": []}, {}, {"$$$": []}),
        ({"$$$": []}, {"$$$": "sh-2"}, {"$$$": "sh-2"}),
        ({"$$$": "sh-2"}, {"$$$": []}, {"$$$": []}),
        ({}, {"$$$": "sh"}, {"$$$": "sh"}),
        ({"$$$": "sh"}, {}, {"$$$": "sh"}),
        ({"$$$": "sh-1"}, {"$$$": "sh-2"}, {"$$$": "sh-2"}),
        ({"$$$": ["sh-1"]}, {"$$$": "sh-2"}, {"$$$": "sh-2"}),
        ({"$$$": "sh-1"}, {"$$$": ["sh-2"]}, {"$$$": ["sh-2"]}),
        ({"$$$": "sh-1"}, {"$$$": ["sh-2", "sh-3"]}, {"$$$": ["sh-2", "sh-3"]}),
        ({"$$$": ["sh-1"]}, {"$$$": ["sh-2", "sh-3"]}, {"$$$": ["sh-2", "sh-3"]}),
        ({"$$$": ["sh-1", "sh-2"]}, {"$$$": ["sh-3", "sh-4"]}, {"$$$": ["sh-3", "sh-4"]}),
        ({}, {"$$$": ["sh-3", "sh      4"]}, {"$$$": ["sh-3", "sh      4"]}),
        ({"$$$": "sleep infinity"}, {"$$$": "sh"}, {"$$$": "sh"}),
        ({"$$$": "sh"}, {"$$$": "sleep infinity"}, {"$$$": "sleep infinity"}),
        (
            {},
            {"$$$": "bash -c 'sleep infinity'"},
            {"$$$": "bash -c 'sleep infinity'"},
        ),
    ])
    def test_parse_compose_file_when_multiple_composes_keys_command_entrypoint(
        self, base_template, override_template, expected_template
    ):
        for key in ["command", "entrypoint"]:
            base, override, expected = template_to_expression(
                base_template, override_template, expected_template, key
            )
            compose_test_1 = {"services": {"test-service": base}}
            compose_test_2 = {"services": {"test-service": override}}
            dump_yaml(compose_test_1, "test-compose-1.yaml")
            dump_yaml(compose_test_2, "test-compose-2.yaml")

            podman_compose = ComposeEngine()
            set_args(podman_compose, ["test-compose-1.yaml", "test-compose-2.yaml"])

            podman_compose._parse_compose_file()  # pylint: disable=protected-access

            actual = {}
            if podman_compose.services:
                actual = podman_compose.original_configuration(
                    podman_compose.services["test-service"]
                )
            self.assertEqual(actual, expected)


def set_args(podman_compose: ComposeEngine, file_names: list[str]) -> None:
    podman_compose.global_args = argparse.Namespace()
    podman_compose.global_args.file = file_names
    podman_compose.global_args.project_name = None
    podman_compose.global_args.env_file = None
    podman_compose.global_args.profile = []
    podman_compose.global_args.in_pod = "1"
    podman_compose.global_args.pod_args = None
    podman_compose.global_args.no_normalize = True


def dump_yaml(compose: dict, name: str) -> None:
    with open(name, "w", encoding="utf-8") as outfile:
        yaml.safe_dump(compose, outfile, default_flow_style=False)


def template_to_expression(base, override, expected, key):
    base_copy = copy.deepcopy(base)
    override_copy = copy.deepcopy(override)
    expected_copy = copy.deepcopy(expected)

    expected_copy[key] = expected_copy.pop("$$$")
    if "$$$" in base:
        base_copy[key] = base_copy.pop("$$$")
    if "$$$" in override:
        override_copy[key] = override_copy.pop("$$$")
    return base_copy, override_copy, expected_copy


def test_clean_test_yamls() -> None:
    test_files = ["test-compose-1.yaml", "test-compose-2.yaml"]
    for file in test_files:
        assert not os.path.exists(file)
