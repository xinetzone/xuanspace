# SPDX-License-Identifier: GPL-2.0-only
"""生成 argparse 对等快照（TR-9.1）——**非 pytest 收集文件**（命名无 test_ 前缀）。

用途
----
从 vendor 只读子模块 ``vendor/podman-compose``（pinned commit 见快照
``meta`` 字段）加载上游单文件模块，程序化构造其完整 argparse parser
树（复刻上游 ``PodmanCompose._parse_args`` 第 3149-3160 行的构造序：
``_init_global_parser`` → title="command" subparsers → help 伪命令 →
逐命令 ``cmd._parse_args`` append），序列化为
``tests/snapshots/cli_parity.json``。

``tests/test_cli_parity.py`` 在**不依赖 vendor 路径**的前提下读取该
快照与新库 ``build_parser()`` 深度对拍；日常 CI/门禁只需快照文件，
vendor 检出与否不影响测试。

何时重新生成
------------
仅当主动升级上游基线（gitlink bump 到新的 podman-compose release）
时才重新运行；生成后必须：

1. ``git -C vendor/podman-compose rev-parse HEAD`` 核对 meta.commit；
2. 运行 ``pytest tests/test_cli_parity.py tests/test_command_surface.py``，
   出现差异时先人工研判是上游有意变更还是本地翻译漂移，禁止为让测试
   变绿而无判据地刷新快照。

运行方式（仓库根 = SpecWeave/）
-------------------------------
``PYTHONPATH=src python tests/generate_parity_snapshots.py``

可用环境变量 ``PODMAN_COMPOSE_SRC`` 指向上游源码目录以覆盖默认锚定
路径（默认: 本文件上溯 6 级的 ``vendor/podman-compose``）。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
LIB_ROOT = THIS_FILE.parents[1]  # libs/xuan-compose
DEFAULT_VENDOR = THIS_FILE.parents[5] / "vendor" / "podman-compose"
SNAPSHOT_PATH = THIS_FILE.parent / "snapshots" / "cli_parity.json"

# 使 import tests.parity_lib 可用（独立脚本运行时无 pytest 的路径注入）
sys.path.insert(0, str(LIB_ROOT))

from tests.parity_lib import serialize_parser_tree  # noqa: E402


def _vendor_dir() -> Path:
    path = Path(os.environ.get("PODMAN_COMPOSE_SRC", DEFAULT_VENDOR)).resolve()
    entry = path / "podman_compose.py"
    if not entry.is_file():
        raise SystemExit(
            f"找不到上游源码目录（期望含 podman_compose.py）: {path}\n"
            "可设置 PODMAN_COMPOSE_SRC 指向上游 podman-compose 源码根目录。"
        )
    return path


def _git_commit(vendor: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_upstream_parser(vendor: Path) -> argparse.ArgumentParser:
    """加载上游模块并复刻其 _parse_args 的 parser 构造段（不解析 argv）。"""
    sys.path.insert(0, str(vendor))
    # 排除环境变量对 default 的干扰，保证快照与运行机环境无关
    os.environ.pop("COMPOSE_PARALLEL_LIMIT", None)
    import podman_compose  # type: ignore[import-not-found]

    compose = podman_compose.podman_compose  # 模块级单例，装饰器已注册 24 命令
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    compose._init_global_parser(parser)
    subparsers = parser.add_subparsers(title="command", dest="command")
    subparsers.add_parser("help", help="show help")
    for cmd_name, cmd in compose.commands.items():
        subparser = subparsers.add_parser(cmd_name, help=cmd.help, description=cmd.desc)
        for cmd_parser in cmd._parse_args:
            cmd_parser(subparser)
    return parser


def main() -> None:
    vendor = _vendor_dir()
    parser = build_upstream_parser(vendor)
    snapshot = serialize_parser_tree(parser)
    snapshot["meta"] = {
        "source": "containers/podman-compose",
        "source_file": "podman_compose.py",
        "commit": _git_commit(vendor),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "note": "由 tests/generate_parity_snapshots.py 生成；仅上游基线升级时重新生成",
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {SNAPSHOT_PATH}")
    print(f"commands: {len(snapshot['commands'])} (含 help 伪命令)")
    print(f"upstream commit: {snapshot['meta']['commit']}")


if __name__ == "__main__":
    main()
