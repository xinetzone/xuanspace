# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_normalize_depends_on.py（断言零削弱）。"""

import copy
import unittest
from typing import Any

from parameterized import parameterized

from xuan_compose.normalize import normalize_service


class TestNormalizeServicesSimple(unittest.TestCase):
    @parameterized.expand(
        [
            (
                {"depends_on": "my_service"},
                {"depends_on": {"my_service": {"condition": "service_started"}}},
            ),
            (
                {"depends_on": ["my_service"]},
                {"depends_on": {"my_service": {"condition": "service_started"}}},
            ),
            (
                {"depends_on": ["my_service1", "my_service2"]},
                {
                    "depends_on": {
                        "my_service1": {"condition": "service_started"},
                        "my_service2": {"condition": "service_started"},
                    }
                },
            ),
            (
                {"depends_on": {"my_service": {"condition": "service_started"}}},
                {"depends_on": {"my_service": {"condition": "service_started"}}},
            ),
            (
                {"depends_on": {"my_service": {"condition": "service_healthy"}}},
                {"depends_on": {"my_service": {"condition": "service_healthy"}}},
            ),
        ]
    )
    def test_normalize_service_simple(
        self, test_case: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        copy.deepcopy(test_case)
        result = normalize_service(test_case)
        self.assertEqual(result, expected)
