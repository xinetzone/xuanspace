"""
路径工具模块
提供项目根路径查找等功能
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    获取 Xuanspace monorepo 根目录路径

    从当前文件位置向上追溯5级：
    path.py → xuan_core/ → src/ → xuan-core/ → libs/ → xuanspace/

    Returns:
        Path: xuanspace 项目根目录的 Path 对象
    """
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def get_subproject_root() -> Path:
    """
    获取当前子项目（xuan-core）的根目录路径

    Returns:
        Path: xuan-core 子项目根目录的 Path 对象
    """
    return Path(__file__).resolve().parent.parent.parent
