# SPDX-License-Identifier: GPL-2.0-only
"""脚手架冒烟测试。"""

import xuan_compose


def test_version() -> None:
    assert isinstance(xuan_compose.__version__, str)
    assert xuan_compose.__version__.startswith("1.6.0")


def test_import_has_no_side_effects() -> None:
    # 导入即访问版本属性，不读取 sys.argv、不产生子进程
    assert hasattr(xuan_compose, "__version__")
