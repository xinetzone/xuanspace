# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_main.py（断言语义零削弱）。

适配点：
- ``from podman_compose import PodmanComposeError, main`` →
  ``xuan_compose.errors`` / ``xuan_compose.cli.main``；
- mock 目标 ``podman_compose.asyncio.run`` →
  ``xuan_compose.cli.main.asyncio.run``（main 内 ``asyncio.run`` 绑定
  在 cli.main 模块命名空间，补丁该绑定即等价上游）。
"""

import contextlib
import io
import runpy
import unittest
from unittest import mock

from xuan_compose.cli.main import main
from xuan_compose.errors import PodmanComposeError


class TestMain(unittest.TestCase):
    @mock.patch("xuan_compose.cli.main.asyncio.run")
    def test_main_shows_clean_error_for_podman_compose_error(self, mocked_run: mock.Mock) -> None:
        def fake_asyncio_run(coro: object) -> None:
            if hasattr(coro, "close"):
                coro.close()  # prevent un-awaited coroutine warnings in this unit test
            raise PodmanComposeError("External network [missing-net] does not exist")

        mocked_run.side_effect = fake_asyncio_run
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ex:
            main()

        self.assertEqual(ex.exception.code, 1)
        self.assertEqual(
            stderr.getvalue().strip(),
            "Error: External network [missing-net] does not exist",
        )

    def test_python_m_entrypoint_invokes_main(self) -> None:
        # python -m xuan_compose → 包根 __main__.py 调用 cli.main.main 一次
        with mock.patch("xuan_compose.cli.main.main") as mocked:
            runpy.run_module("xuan_compose.__main__", run_name="__main__")
        mocked.assert_called_once_with()
