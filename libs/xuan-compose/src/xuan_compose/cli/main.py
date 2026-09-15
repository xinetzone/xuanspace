# SPDX-License-Identifier: GPL-2.0-only
"""CLI 装配与分发（翻译自 podman_compose.py 第 2480-2523、5528-5537 行）。

上游 ``PodmanCompose.run()`` 是引擎方法、``async_main/main`` 操作模块级
单例；分层后装配全部显式化且集中于本模块（包内唯一读取 ``sys.argv``
的运行时入口，TR-8.3）：

1. 构造 :class:`~xuan_compose.engine.ComposeEngine` 并安装命令注册表；
2. 经 :func:`~xuan_compose.cli.parser.parse_args` 解析 argv；
3. 按参数装配 :class:`~xuan_compose.runner.Podman`（路径校验、信号量）；
4. 非 dry-run 探测 podman 版本（失败给上游等价的 fatal + 退出码 1）；
5. 除 ``version`` 与 ``systemd create-unit`` 外加载 compose 文件；
6. 注入可执行路径并经 ``engine.commands`` 查表分发 handler，int 返回码
   转 ``SystemExit``。
"""

import asyncio
import os
import sys

from .. import __version__
from ..commands import install_handlers
from ..engine import ComposeEngine
from ..errors import PodmanComposeError
from ..logging_utils import log
from ..runner import CalledProcessError, Podman
from .parser import parse_args

__all__ = ["async_main", "main"]


async def async_main(argv: list[str] | None = None) -> None:
    engine = ComposeEngine()
    install_handlers(engine)

    log.info("podman-compose version: %s", __version__)
    args = parse_args(engine, argv)
    podman_path = args.podman_path
    if podman_path != "podman":
        if os.path.isfile(podman_path) and os.access(podman_path, os.X_OK):
            podman_path = os.path.realpath(podman_path)
        else:
            # this also works if podman hasn't been installed now
            if args.dry_run is False:
                log.fatal("Binary %s has not been found.", podman_path)
                sys.exit(1)
    engine.podman = Podman(engine, podman_path, args.dry_run, asyncio.Semaphore(args.parallel))
    # 上游模块级 script（第 56 行）的分层落点：仅 CLI 装配期读取一次
    # 解释器启动参数，供 systemd create-unit 模板使用（T10 差异表）。
    engine.executable = os.path.realpath(sys.argv[0])

    if not args.dry_run:
        # just to make sure podman is running
        try:
            assert engine.podman is not None
            engine.podman_version = (
                await engine.podman.output(["--version"], "", [])
            ).decode("utf-8").strip() or ""
            engine.podman_version = (engine.podman_version.split() or [""])[-1]
        except (CalledProcessError, FileNotFoundError) as e:
            msg = str(e)
            if isinstance(e, CalledProcessError) and e.output:
                msg += f": {e.output.decode('utf-8')}"
            log.error("failed to check if podman is installed: %s", msg)
            engine.podman_version = None
        if not engine.podman_version:
            log.fatal(
                "It seems that you either do not have `podman` installed "
                "or the `podman version` command failed."
            )
            sys.exit(1)
        log.info("using podman version: %s", engine.podman_version)
    cmd_name = args.command
    compose_required = cmd_name != "version" and (
        cmd_name != "systemd" or args.action != "create-unit"
    )
    if compose_required:
        engine._parse_compose_file()
    cmd = engine.commands[cmd_name]
    retcode = await cmd(engine, args)
    if isinstance(retcode, int):
        sys.exit(retcode)


def main(argv: list[str] | None = None) -> None:
    try:
        asyncio.run(async_main(argv))
    except PodmanComposeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
