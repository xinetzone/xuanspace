# SPDX-License-Identifier: GPL-2.0-only
"""interpolation 错误与字面量分支补测（移植的 test_var_interpolate
覆盖全部合法语法；这里补未闭合括号、非法操作符与结尾裸 ``$`` 三类
真实报错/字面量分支）。"""

import pytest

from xuan_compose.interpolation import var_interpolate


def test_unclosed_brace_raises() -> None:
    with pytest.raises(ValueError, match="No closing brace"):
        var_interpolate("${X", {})


def test_unknown_operator_syntax_raises() -> None:
    with pytest.raises(ValueError, match="Invalid variable interpolation syntax"):
        var_interpolate("${X!default}", {})


def test_trailing_dollar_is_literal() -> None:
    # 末尾 $ 后无字符：无 lookahead，按字面量保留
    assert var_interpolate("price: 5$", {}) == "price: 5$"
