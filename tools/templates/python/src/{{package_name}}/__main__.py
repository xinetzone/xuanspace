"""
{{package_name}} 命令行入口
使用方式: python -m {{package_name}}
"""

from . import __version__


def main() -> None:
    """主入口函数"""
    print(f"{{name}} v{__version__}")
    print("Xuanspace（玄境）Python monorepo 子项目")


if __name__ == "__main__":
    main()
