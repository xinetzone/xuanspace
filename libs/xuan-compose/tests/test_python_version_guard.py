# SPDX-License-Identifier: GPL-2.0-only
"""P2-10 回归测试：低版本 Python 误用须得到友好 ``RuntimeError``。

背景（findings-ledger §C P2-10）：M1 移除 ``from __future__ import
annotations`` 后，merge.py 类体内自引用注解（``-> OverrideTag`` 等）
在 PEP 649 之前的解释器上定义时即求值，Python 3.13 直接
``import xuan_compose.cli.main`` 会抛晦涩的 ``NameError: OverrideTag``。
pyproject 的 ``requires-python>=3.14`` 只挡 pip 安装，挡不住 PYTHONPATH
直跑/IDE 误用。守卫位于包 ``__init__.py`` 最早执行点，先于一切第三方
依赖导入，故在未装 PyYAML 的解释器上也能给出清晰诊断。
"""

import sys

import pytest

from xuan_compose import MINIMUM_PYTHON_VERSION, require_python_version


class TestRequirePythonVersion:
    def test_floor_is_3_14(self) -> None:
        assert MINIMUM_PYTHON_VERSION == (3, 14)

    def test_current_interpreter_passes(self) -> None:
        # 包导入时守卫已执行过一次；在受支持解释器上显式再调必须静默通过
        assert require_python_version() is None
        assert sys.version_info[:2] >= MINIMUM_PYTHON_VERSION

    @pytest.mark.parametrize("current", [(3, 13), (3, 13, 5), (3, 13, 99), (2, 7, 18)])
    def test_older_interpreter_raises_friendly_runtime_error(self, current: tuple) -> None:
        with pytest.raises(RuntimeError) as exc_info:
            require_python_version(current=current)

        message = str(exc_info.value)
        # 最低版本、实际版本（major.minor）、升级与 requires-python 引导缺一不可
        assert "requires Python 3.14" in message
        assert f"Python {current[0]}.{current[1]}" in message
        assert "requires-python" in message

    @pytest.mark.parametrize("current", [(3, 14), (3, 14, 0), (3, 15, 1), (4, 0, 0)])
    def test_supported_versions_pass_silently(self, current: tuple) -> None:
        assert require_python_version(current=current) is None
