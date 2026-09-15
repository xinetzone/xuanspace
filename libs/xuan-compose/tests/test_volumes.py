# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_volumes.py（断言零削弱）。

适配点：``podman_compose.parse_short_mount`` →
``xuan_compose.translate.mounts.parse_short_mount``。
"""

# pylint: disable=redefined-outer-name
import unittest

from xuan_compose.translate.mounts import parse_short_mount


class ParseShortMountTests(unittest.TestCase):
    def test_multi_propagation(self) -> None:
        self.assertEqual(
            parse_short_mount("/foo/bar:/baz:U,Z", "/"),
            {
                "type": "bind",
                "source": "/foo/bar",
                "target": "/baz",
                "bind": {
                    "propagation": "U,Z",
                },
            },
        )
