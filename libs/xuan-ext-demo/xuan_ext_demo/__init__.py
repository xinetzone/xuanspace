"""
Xuan-Ext-Demo: C/C++ 原生扩展示例库

本包展示如何使用 scikit-build-core + CMake + Ninja 构建 Python 原生扩展。
实际的 C++ 扩展模块需要在 CMakeLists.txt 中配置并编译后才能使用。
"""

__version__ = "0.1.0"


def add(a: int, b: int) -> int:
    """
    加法函数（Python实现占位）

    注意：此函数为纯Python实现占位，实际C++扩展编译后会被同名原生函数覆盖。

    Args:
        a: 第一个整数
        b: 第二个整数

    Returns:
        a + b 的结果
    """
    return a + b


__all__ = ["__version__", "add"]
