"""
{{package_name}} 基础测试
"""

from {{package_name}} import __version__


def test_version():
    """测试版本号格式"""
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_import():
    """测试模块导入"""
    import {{package_name}}
    assert hasattr({{package_name}}, "__version__")
