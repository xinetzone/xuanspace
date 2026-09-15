# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_service_dependency_condition.py（断言零削弱）。"""

import unittest

from xuan_compose.model import ServiceDependencyCondition


class TestServiceDependencyCondition(unittest.TestCase):
    def test_service_completed_successfully_maps_to_service_completed_successfully(self) -> None:
        condition = ServiceDependencyCondition.from_value("service_completed_successfully")
        self.assertEqual(condition, ServiceDependencyCondition.SERVICE_COMPLETED_SUCCESSFULLY)

    def test_service_healthy_maps_correctly(self) -> None:
        condition = ServiceDependencyCondition.from_value("service_healthy")
        self.assertEqual(condition, ServiceDependencyCondition.HEALTHY)

    def test_service_started_maps_to_running(self) -> None:
        condition = ServiceDependencyCondition.from_value("service_started")
        self.assertEqual(condition, ServiceDependencyCondition.RUNNING)

    def test_direct_condition_values(self) -> None:
        self.assertEqual(
            ServiceDependencyCondition.from_value("stopped"),
            ServiceDependencyCondition.STOPPED,
        )
        self.assertEqual(
            ServiceDependencyCondition.from_value("healthy"),
            ServiceDependencyCondition.HEALTHY,
        )
        self.assertEqual(
            ServiceDependencyCondition.from_value("running"),
            ServiceDependencyCondition.RUNNING,
        )

    def test_invalid_condition_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            ServiceDependencyCondition.from_value("invalid_condition")
