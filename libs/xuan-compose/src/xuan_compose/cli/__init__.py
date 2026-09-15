# SPDX-License-Identifier: GPL-2.0-only
"""CLI 层（T8）——argparse parser、显式装配与进程入口。

本包是分层架构的最薄适配层（spec 公理 4）：
argv → Namespace → 装配 Engine/Runner → 查表分发 handler。
包内模块是 ``sys.argv`` 的唯一合法读取处（TR-8.3）。
"""

from .main import async_main, main
from .parser import (
    COMMAND_HELP,
    COMMAND_PARSERS,
    PODMAN_CMDS,
    PullPolicyAction,
    build_parser,
    parse_args,
)

__all__ = [
    "COMMAND_HELP",
    "COMMAND_PARSERS",
    "PODMAN_CMDS",
    "PullPolicyAction",
    "async_main",
    "build_parser",
    "main",
    "parse_args",
]
