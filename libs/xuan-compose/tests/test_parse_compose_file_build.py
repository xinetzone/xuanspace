# SPDX-License-Identifier: GPL-2.0-only
# pylint: disable=protected-access
"""build.context 经引擎解析后的最终规范化测试（引擎部分，24 用例）。

逐用例移植自上游 tests/unit/test_normalize_final_build.py 中依赖
``PodmanCompose._parse_compose_file`` 的两个参数化方法（T3 延期登记）：

- ``test_parse_compose_file_when_single_compose``：9 用例；
- ``test_parse_when_multiple_composes``：14 用例；
- ``test_clean_test_yamls``：1 个卫生用例（上游收集计数包含它）。

适配点：
- 导入符号由 ``podman_compose`` 改为
  ``xuan_compose.engine.ComposeEngine``；
- 上游 ``cwd = os.path.abspath(".")`` 取 pytest 启动目录并把 yaml 直接
  落在该目录；移植版在模块导入时创建独占临时根目录（与 T5
  test_container_to_args 的 REPO_ROOT 同模式），用例在其中 chdir 执行，
  并快照/还原 COMPOSE_*/PODMAN_* 进程环境；数据表与断言逐行不变。
"""

import argparse
import os
import shutil
import tempfile
import unittest

import yaml
from parameterized import parameterized

from xuan_compose.engine import ComposeEngine

# 本模块独占的等价"仓库根"：全部 yaml 落在此处，cwd 期望值锚定它
_PROJECT_ROOT = os.path.realpath(tempfile.mkdtemp(prefix="xuan-compose-final-build-"))
cwd = _PROJECT_ROOT

# 引擎解析过程可能读写的进程环境键，用例间必须还原
_ENV_KEYS = (
    "COMPOSE_PROJECT_DIR",
    "COMPOSE_FILE",
    "COMPOSE_PATH_SEPARATOR",
    "COMPOSE_PROJECT_NAME",
    "COMPOSE_PROFILES",
)


class TestParseComposeFileBuild(unittest.TestCase):
    def setUp(self) -> None:
        self._cwd = os.getcwd()
        self._env = {k: os.environ.get(k) for k in _ENV_KEYS}
        os.chdir(cwd)

    def tearDown(self) -> None:
        for name in ("test-compose.yaml", "test-compose-1.yaml", "test-compose-2.yaml"):
            path = os.path.join(cwd, name)
            if os.path.exists(path):
                os.remove(path)
        os.chdir(self._cwd)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    cases_simple_normalization = [
        ({"image": "test-image"}, {"image": "test-image"}),
        (
            {"build": "."},
            {
                "build": {"context": cwd},
            },
        ),
        (
            {"build": "../relative"},
            {
                "build": {
                    "context": os.path.normpath(os.path.join(cwd, "../relative")),
                },
            },
        ),
        (
            {"build": "./relative"},
            {
                "build": {
                    "context": os.path.normpath(os.path.join(cwd, "./relative")),
                },
            },
        ),
        (
            {"build": "/workspace/absolute"},
            {
                "build": {
                    "context": "/workspace/absolute",
                },
            },
        ),
        (
            {
                "build": {
                    "dockerfile": "Dockerfile",
                },
            },
            {
                "build": {
                    "context": cwd,
                    "dockerfile": "Dockerfile",
                },
            },
        ),
        (
            {
                "build": {
                    "context": ".",
                },
            },
            {
                "build": {
                    "context": cwd,
                },
            },
        ),
        (
            {
                "build": {"context": "../", "dockerfile": "test-dockerfile"},
            },
            {
                "build": {
                    "context": os.path.normpath(os.path.join(cwd, "../")),
                    "dockerfile": "test-dockerfile",
                },
            },
        ),
        (
            {
                "build": {"context": ".", "dockerfile": "./dev/test-dockerfile"},
            },
            {
                "build": {
                    "context": cwd,
                    "dockerfile": "./dev/test-dockerfile",
                },
            },
        ),
    ]

    @parameterized.expand(cases_simple_normalization)
    def test_parse_compose_file_when_single_compose(self, input, expected):
        compose_test = {"services": {"test-service": input}}
        dump_yaml(compose_test, "test-compose.yaml")

        podman_compose = ComposeEngine()
        set_args(podman_compose, ["test-compose.yaml"], no_normalize=None)

        podman_compose._parse_compose_file()

        actual_compose = {}
        if podman_compose.services:
            actual_compose = podman_compose.original_configuration(
                podman_compose.services["test-service"]
            )
        self.assertEqual(actual_compose, expected)

    @parameterized.expand([
        (
            {},
            {"build": "."},
            {"build": {"context": cwd}},
        ),
        (
            {"build": "."},
            {},
            {"build": {"context": cwd}},
        ),
        (
            {"build": "/workspace/absolute"},
            {"build": "./relative"},
            {
                "build": {
                    "context": os.path.normpath(os.path.join(cwd, "./relative")),
                }
            },
        ),
        (
            {"build": "./relative"},
            {"build": "/workspace/absolute"},
            {"build": {"context": "/workspace/absolute"}},
        ),
        (
            {"build": "./relative"},
            {"build": "/workspace/absolute"},
            {"build": {"context": "/workspace/absolute"}},
        ),
        (
            {"build": {"dockerfile": "test-dockerfile"}},
            {},
            {"build": {"context": cwd, "dockerfile": "test-dockerfile"}},
        ),
        (
            {},
            {"build": {"dockerfile": "test-dockerfile"}},
            {"build": {"context": cwd, "dockerfile": "test-dockerfile"}},
        ),
        (
            {},
            {"build": {"dockerfile": "test-dockerfile"}},
            {"build": {"context": cwd, "dockerfile": "test-dockerfile"}},
        ),
        (
            {"build": {"dockerfile": "test-dockerfile-1"}},
            {"build": {"dockerfile": "test-dockerfile-2"}},
            {"build": {"context": cwd, "dockerfile": "test-dockerfile-2"}},
        ),
        (
            {"build": "/workspace/absolute"},
            {"build": {"dockerfile": "test-dockerfile"}},
            {"build": {"context": "/workspace/absolute", "dockerfile": "test-dockerfile"}},
        ),
        (
            {"build": {"dockerfile": "test-dockerfile"}},
            {"build": "/workspace/absolute"},
            {"build": {"context": "/workspace/absolute", "dockerfile": "test-dockerfile"}},
        ),
        (
            {"build": {"dockerfile": "./test-dockerfile-1"}},
            {"build": {"dockerfile": "./test-dockerfile-2", "args": ["ENV1=1"]}},
            {
                "build": {
                    "context": cwd,
                    "dockerfile": "./test-dockerfile-2",
                    "args": ["ENV1=1"],
                }
            },
        ),
        (
            {"build": {"dockerfile": "./test-dockerfile-1", "args": ["ENV1=1"]}},
            {"build": {"dockerfile": "./test-dockerfile-2"}},
            {
                "build": {
                    "context": cwd,
                    "dockerfile": "./test-dockerfile-2",
                    "args": ["ENV1=1"],
                }
            },
        ),
        (
            {"build": {"dockerfile": "./test-dockerfile-1", "args": ["ENV1=1"]}},
            {"build": {"dockerfile": "./test-dockerfile-2", "args": ["ENV2=2"]}},
            {
                "build": {
                    "context": cwd,
                    "dockerfile": "./test-dockerfile-2",
                    "args": ["ENV1=1", "ENV2=2"],
                }
            },
        ),
    ])
    def test_parse_when_multiple_composes(self, input, override, expected):
        compose_test_1 = {"services": {"test-service": input}}
        compose_test_2 = {"services": {"test-service": override}}
        dump_yaml(compose_test_1, "test-compose-1.yaml")
        dump_yaml(compose_test_2, "test-compose-2.yaml")

        podman_compose = ComposeEngine()
        set_args(
            podman_compose,
            ["test-compose-1.yaml", "test-compose-2.yaml"],
            no_normalize=None,
        )

        podman_compose._parse_compose_file()

        actual_compose = {}
        if podman_compose.services:
            actual_compose = podman_compose.original_configuration(
                podman_compose.services["test-service"]
            )
        self.assertEqual(actual_compose, expected)


def set_args(podman_compose: ComposeEngine, file_names: list[str], no_normalize: bool) -> None:
    podman_compose.global_args = argparse.Namespace()
    podman_compose.global_args.file = file_names
    podman_compose.global_args.project_name = None
    podman_compose.global_args.env_file = None
    podman_compose.global_args.profile = []
    podman_compose.global_args.in_pod = "1"
    podman_compose.global_args.pod_args = None
    podman_compose.global_args.no_normalize = no_normalize


def dump_yaml(compose: dict, name: str) -> None:
    # Path(Path.cwd()/"subdirectory").mkdir(parents=True, exist_ok=True)
    with open(name, "w", encoding="utf-8") as outfile:
        yaml.safe_dump(compose, outfile, default_flow_style=False)


def test_clean_test_yamls() -> None:
    test_files = ["test-compose-1.yaml", "test-compose-2.yaml", "test-compose.yaml"]
    for file in test_files:
        assert not os.path.exists(os.path.join(cwd, file))
    # 临时根只应保留为目录本身，引擎产物不得残留于仓库工作区
    assert not os.path.exists(os.path.join(os.getcwd(), "test-compose.yaml"))
    shutil.rmtree(_PROJECT_ROOT, ignore_errors=True)
