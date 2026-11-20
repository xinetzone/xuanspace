"""统一错误类型：调用方据此把工具库失败映射为自己的 CLI 异常。"""


class XuanCmdError(Exception):
    """命令装配或路径解析失败。"""
