# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_override_environment.py（fd64dc2，断言零削弱）。

适配点：
- 扁平 ``from podman_compose import ...`` 改为分层导入（各符号实际所在模块）。
- ``PodmanCompose`` → ``xuan_compose.engine.ComposeEngine``；
  ``podman_compose._parse_compose_file/services/containers`` 对应
  ``engine._parse_compose_file``/``engine.services``/``engine.containers``。
- 上游直接构造 ``global_args = argparse.Namespace(...)``；本库同构
  （``_parse_compose_file`` 只读取相同的 Namespace 字段）。为用例间隔离，
  setUp/tearDown 快照并还原 COMPOSE_* 进程环境，并切到独立临时工作目录
  （上游依赖 pytest 启动目录）。
"""

import argparse
import os
import shutil
import tempfile
import unittest

import yaml

from xuan_compose.engine import ComposeEngine
from xuan_compose.merge import OverrideTag, rec_merge
from xuan_compose.normalize import normalize_service, rec_subs

_ENV_KEYS = (
    "COMPOSE_PROJECT_DIR",
    "COMPOSE_FILE",
    "COMPOSE_PATH_SEPARATOR",
    "COMPOSE_PROJECT_NAME",
    "COMPOSE_PROFILES",
)


class TestOverrideEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self._cwd = os.getcwd()
        self._env = {k: os.environ.get(k) for k in _ENV_KEYS}
        self._tmp = tempfile.mkdtemp(prefix="xuan-compose-override-")
        os.chdir(self._tmp)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _parse(self, files: list[str]) -> ComposeEngine:
        engine = ComposeEngine()
        engine.global_args = argparse.Namespace(
            file=files,
            project_name="test_proj",
            env_file=None,
            profile=[],
            in_pod="false",
            pod_args=None,
            no_normalize=False,
        )
        engine._parse_compose_file()  # pylint: disable=protected-access
        return engine

    def test_yaml_load_mapping_override(self) -> None:
        yaml_content = """
services:
    app:
        image: busybox
        environment: !override
            VAR1: value1
            VAR2: value2
"""
        loaded = yaml.safe_load(yaml_content)
        env = loaded["services"]["app"]["environment"]
        self.assertIsInstance(env, OverrideTag)
        self.assertEqual(env.value, {"VAR1": "value1", "VAR2": "value2"})

    def test_yaml_load_sequence_override(self) -> None:
        yaml_content = """
services:
    app:
        image: busybox
        environment: !override
            - VAR1=value1
            - VAR2=value2
"""
        loaded = yaml.safe_load(yaml_content)
        env = loaded["services"]["app"]["environment"]
        self.assertIsInstance(env, OverrideTag)
        self.assertEqual(env.value, ["VAR1=value1", "VAR2=value2"])

    def test_normalize_service_with_override_mapping(self) -> None:
        service = {
            "image": "busybox",
            "environment": OverrideTag({"VAR1": "value1", "VAR2": "value2"}),
        }
        normalized = normalize_service(service)
        self.assertIsInstance(normalized["environment"], OverrideTag)
        self.assertEqual(
            normalized["environment"].value,
            {"VAR1": "value1", "VAR2": "value2"},
        )

    def test_normalize_service_with_override_sequence(self) -> None:
        service = {
            "image": "busybox",
            "environment": OverrideTag(["VAR1=value1", "VAR2=value2", "VAR3"]),
        }
        normalized = normalize_service(service)
        self.assertIsInstance(normalized["environment"], OverrideTag)
        self.assertEqual(
            normalized["environment"].value,
            {"VAR1": "value1", "VAR2": "value2", "VAR3": None},
        )

    def test_rec_subs_with_override_mapping(self) -> None:
        service = {
            "environment": OverrideTag({"V1": "$HOST_V1", "V2": "prefix_${HOST_V1}"}),
            "image": "busybox",
        }
        sub_dict = {"HOST_V1": "interpolated"}
        result = rec_subs(service, sub_dict)
        self.assertIsInstance(result["environment"], OverrideTag)
        self.assertEqual(
            result["environment"].value,
            {"V1": "interpolated", "V2": "prefix_interpolated"},
        )

    def test_rec_subs_service_env_override_used_in_other_fields(self) -> None:
        service = {
            "environment": OverrideTag({"V100": "v1.0.0"}),
            "image": "abc:$V100",
        }
        sub_dict: dict[str, str] = {}
        result = rec_subs(service, sub_dict)
        self.assertEqual(result["image"], "abc:v1.0.0")

    def test_rec_subs_short_form_with_override(self) -> None:
        service = {
            "environment": OverrideTag({"SHORT_VAR": None, "NORMAL_VAR": "val"}),
            "image": "busybox",
        }
        sub_dict = {"SHORT_VAR": "resolved_from_env"}
        result = rec_subs(service, sub_dict)
        self.assertIsInstance(result["environment"], OverrideTag)
        self.assertEqual(
            result["environment"].value,
            {"SHORT_VAR": "resolved_from_env", "NORMAL_VAR": "val"},
        )

    def test_rec_merge_replaces_environment_on_override(self) -> None:
        base_service = {
            "image": "busybox",
            "environment": {"OLD_A": "1", "COMMON": "old"},
        }
        override_service = {
            "environment": OverrideTag({"NEW_B": "2", "COMMON": "new"}),
        }
        merged = rec_merge({}, base_service, override_service)
        # With !override, COMMON is updated and OLD_A is removed
        self.assertEqual(merged["environment"], {"NEW_B": "2", "COMMON": "new"})

    def test_extends_with_environment_override(self) -> None:
        common_yaml = """
services:
    webapp:
        image: busybox
        environment:
            VAR_BASE: "base"
            VAR_COMMON: "common"
"""
        app_yaml = """
services:
    web:
        extends:
            file: common.yaml
            service: webapp
        environment: !override
            VAR_OVERRIDE: "override"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_common = os.path.join(tmp_dir, "common.yaml")
            file_app = os.path.join(tmp_dir, "docker-compose.yaml")
            with open(file_common, "w", encoding="utf-8") as f:
                f.write(common_yaml)
            with open(file_app, "w", encoding="utf-8") as f:
                f.write(app_yaml)

            podman_compose = self._parse([file_app])

            web_service = podman_compose.services["web"]
            self.assertEqual(
                web_service["environment"],
                {"VAR_OVERRIDE": "override"},
            )

    def test_parse_compose_file_multiple_with_environment_override(self) -> None:
        base_yaml = """
services:
    app:
        image: busybox
        environment:
            BASE_VAR: "base"
            SHARED_VAR: "from_base"
"""
        override_yaml = """
services:
    app:
        environment: !override
            SHARED_VAR: "from_override"
            OVERRIDE_VAR: "override"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "docker-compose.yaml")
            file2 = os.path.join(tmp_dir, "docker-compose.override.yaml")
            with open(file1, "w", encoding="utf-8") as f:
                f.write(base_yaml)
            with open(file2, "w", encoding="utf-8") as f:
                f.write(override_yaml)

            podman_compose = self._parse([file1, file2])

            app_service = podman_compose.services["app"]
            # BASE_VAR should be removed because !override replaced the entire environment
            self.assertEqual(
                app_service["environment"],
                {"SHARED_VAR": "from_override", "OVERRIDE_VAR": "override"},
            )
            # Ensure containers also have the unwrapped overridden environment
            self.assertEqual(
                podman_compose.containers[0]["environment"],
                {"SHARED_VAR": "from_override", "OVERRIDE_VAR": "override"},
            )

    def test_parse_compose_file_sequence_environment_override(self) -> None:
        base_yaml = """
services:
    app:
        image: busybox
        environment:
            - BASE_VAR=base
            - SHARED_VAR=from_base
"""
        override_yaml = """
services:
    app:
        environment: !override
            - SHARED_VAR=from_override
            - OVERRIDE_VAR=override
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "docker-compose.yaml")
            file2 = os.path.join(tmp_dir, "docker-compose.override.yaml")
            with open(file1, "w", encoding="utf-8") as f:
                f.write(base_yaml)
            with open(file2, "w", encoding="utf-8") as f:
                f.write(override_yaml)

            podman_compose = self._parse([file1, file2])

            app_service = podman_compose.services["app"]
            self.assertEqual(
                app_service["environment"],
                {"SHARED_VAR": "from_override", "OVERRIDE_VAR": "override"},
            )
