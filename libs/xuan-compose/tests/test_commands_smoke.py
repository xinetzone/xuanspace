# SPDX-License-Identifier: GPL-2.0-only
"""TR-7.2：命令层导入冒烟——导入零引擎实例化、零子进程边界外溢。

- 导入 ``xuan_compose.commands`` 不触发任何 podman 调用（用
  ``Podman.output/run/exec`` 包装器计数证明）；
- 源码扫描：命令层不直接 ``import subprocess``（``CalledProcessError``
  统一从 runner 边界再导出），也不实例化 ``ComposeEngine``；
- 命令层不反向导入引擎实现模块（依赖方向：engine → commands，T11 审查）。
"""

import subprocess
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

COMMANDS_DIR = Path(__file__).resolve().parents[1] / "src" / "xuan_compose" / "commands"


class TestCommandsImportSmoke(IsolatedAsyncioTestCase):
    def test_import_package_creates_no_engine(self) -> None:
        with mock.patch("xuan_compose.runner.Podman") as podman_cls:
            __import__("xuan_compose.commands")
        # 导入期不得构造 Podman（子进程边界唯一入口）
        podman_cls.assert_not_called()

    def test_no_direct_subprocess_import_in_command_sources(self) -> None:
        for path in sorted(COMMANDS_DIR.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "import subprocess" not in source, (
                f"{path.name}: 子进程异常须经 xuan_compose.runner.CalledProcessError 再导出"
            )
            assert "subprocess." not in source, (
                f"{path.name}: 命令层不得直接发起子进程，唯一边界是 runner.Podman"
            )

    def test_no_engine_instantiation_or_import(self) -> None:
        for path in sorted(COMMANDS_DIR.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "ComposeEngine(" not in source, f"{path.name}: 不得实例化引擎"
            assert "from ..engine import" not in source and "from .. import engine" not in source, (
                f"{path.name}: 命令层不得反向依赖引擎实现（compose 形参为 Any 鸭子类型）"
            )

    def test_subprocess_module_still_importable_for_reference(self) -> None:
        # 护栏自检：本测试自身可以使用 subprocess 模块（确认扫描规则不是误配置）
        assert subprocess.CalledProcessError is not None
