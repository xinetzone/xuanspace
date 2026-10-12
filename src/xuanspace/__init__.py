"""Xuanspace（玄境）monorepo 枢纽包。

根发行版 ``xuanspace`` 只承载工作区元信息与开发依赖聚合，本身不含业务代码：
CLI 由独立发行版 *xs-cli*（``tools/xs``）提供，可复用库位于 ``libs/*``，
各子项目以独立发行版方式构建与安装。PDM 用户经工作区配置一次性安装全部成员；
pip/uv 用户需额外执行 ``pip install -e tools/xs``（与 CI 一致）。
"""

from __future__ import annotations

__version__ = "0.1.0"
