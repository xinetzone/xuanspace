# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_volumes.py（断言零削弱）。

适配点：``podman_compose.parse_short_mount`` →
``xuan_compose.translate.mounts.parse_short_mount``。

Windows 路径适配：bind 源路径与生产代码一样逐行保留上游
``os.path.abspath(os.path.join(basedir, ...))`` 本机语义——POSIX 下
``/foo/bar`` 恒等，Windows 下按当前盘符解析为 ``<盘符>:\\foo\\bar``；
容器内目标 ``/baz`` 恒为 POSIX 串不变。
"""

# pylint: disable=redefined-builtin
import os
import unittest

from xuan_compose.translate.mounts import parse_short_mount


class ParseShortMountTests(unittest.TestCase):
    def test_multi_propagation(self) -> None:
        self.assertEqual(
            parse_short_mount("/foo/bar:/baz:U,Z", "/"),
            {
                "type": "bind",
                "source": os.path.abspath("/foo/bar"),
                "target": "/baz",
                "bind": {
                    "propagation": "U,Z",
                },
            },
        )
