"""
{{package_name}} 基础测试
"""

from {{package_name}} import add, add_f, __version__


def test_version():
    """测试版本号格式"""
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_add_integers():
    """测试整数加法"""
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(100, 200) == 300


def test_add_floats():
    """测试浮点数加法"""
    assert add_f(1.5, 2.5) == 4.0
    assert abs(add_f(0.1, 0.2) - 0.3) < 1e-10


def test_import():
    """测试模块导入"""
    import {{package_name}}
    assert hasattr({{package_name}}, "add")
    assert callable({{package_name}}.add)
