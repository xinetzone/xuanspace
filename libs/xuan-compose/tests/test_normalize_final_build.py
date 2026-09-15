# SPDX-License-Identifier: GPL-2.0-only
# pylint: disable=protected-access
"""build.context 最终规范化测试（纯函数部分）。

逐用例移植自上游 tests/unit/test_normalize_final_build.py 的前两个参数化方法
（cases_simple_normalization × 2，共 18 用例，断言零削弱）。

拆分说明：上游同文件中依赖 ``PodmanCompose._parse_compose_file`` 的两个方法
（test_parse_compose_file_when_single_compose 9 用例、
test_parse_when_multiple_composes 14 用例，及其 set_args/dump_yaml/清理函数）
需要引擎层装配，按 tasks.md 分层在 T6 整体迁入
tests/test_parse_compose_file_build.py，用例数据与断言原样保留。
"""

import os
import unittest

from parameterized import parameterized

from xuan_compose.normalize import normalize_final, normalize_service_final

cwd = os.path.abspath(".")


class TestNormalizeFinalBuild(unittest.TestCase):
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
    def test_normalize_service_final_returns_absolute_path_in_context(self, input, expected):
        # Tests that [service.build] is normalized after merges
        project_dir = cwd
        self.assertEqual(normalize_service_final(input, project_dir), expected)

    @parameterized.expand(cases_simple_normalization)
    def test_normalize_returns_absolute_path_in_context(self, input, expected):
        project_dir = cwd
        compose_test = {"services": {"test-service": input}}
        compose_expected = {"services": {"test-service": expected}}
        self.assertEqual(normalize_final(compose_test, project_dir), compose_expected)
