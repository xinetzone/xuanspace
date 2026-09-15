# SPDX-License-Identifier: GPL-2.0-only
"""``version`` 命令（翻译自 podman_compose.py 第 3378-3388 行）。"""

import argparse
import json
from typing import Any

from .. import __version__

__all__ = ["compose_version"]


async def compose_version(compose: Any, args: argparse.Namespace) -> None:
    if getattr(args, "short", False):
        print(__version__)
        return
    if getattr(args, "format", "pretty") == "json":
        res = {"version": __version__}
        print(json.dumps(res))
        return
    print("podman-compose version", __version__)
    await compose.podman.run(["--version"], "", [])
