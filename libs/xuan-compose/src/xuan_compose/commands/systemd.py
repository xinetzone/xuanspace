# SPDX-License-Identifier: GPL-2.0-only
"""``systemd`` 命令（翻译自 podman_compose.py 第 3411-3513 行）。

与上游的唯一结构性差异（T10 差异表）：unit 模板中的可执行路径上游取自模块级
``script``（第 56 行，导入期读取解释器启动参数的副作用）；分层后
改为读取引擎显式字段 ``compose.executable``，由 CLI 层装配时注入，本模块
导入期不读取解释器启动参数（FR-5/TR-8.3）。
"""

import argparse
import getpass
import glob
import os
import sys
from typing import Any

from ..logging_utils import log

__all__ = ["compose_systemd"]


async def compose_systemd(compose: Any, args: argparse.Namespace) -> None:
    """
    create systemd unit file and register its compose stacks

    When first installed type `sudo podman-compose systemd -a create-unit`
    later you can add a compose stack by running `podman-compose systemd -a register`
    then you can start/stop your stack with `systemctl --user start podman-compose@<PROJ>`
    """
    stacks_dir = ".config/containers/compose/projects"
    if args.action == "register":
        proj_name = compose.project_name
        fn = os.path.expanduser(f"~/{stacks_dir}/{proj_name}.env")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        log.debug("writing [%s]: ...", fn)
        with open(fn, "w", encoding="utf-8") as f:
            for k, v in compose.environ.items():
                if k.startswith("COMPOSE_") or k.startswith("PODMAN_"):
                    f.write(f"{k}={v}\n")
        log.debug("writing [%s]: done.", fn)
        log.info("\n\ncreating the pod without starting it: ...\n\n")
        username = getpass.getuser()
        print(
            f"""
you can use systemd commands like enable, start, stop, status, cat
all without `sudo` like this:

\t\tsystemctl --user enable --now 'podman-compose@{proj_name}'
\t\tsystemctl --user status 'podman-compose@{proj_name}'
\t\tjournalctl --user -xeu 'podman-compose@{proj_name}'

and for that to work outside a session
you might need to run the following command *once*

\t\tsudo loginctl enable-linger '{username}'

you can use podman commands like:

\t\tpodman pod ps
\t\tpodman pod stats 'pod_{proj_name}'
\t\tpodman pod logs --tail=10 -f 'pod_{proj_name}'
"""
        )
    elif args.action == "unregister":
        proj_name = compose.project_name
        fn = os.path.expanduser(f"~/{stacks_dir}/{proj_name}.env")
        if os.path.exists(fn):
            try:
                log.debug("removing [%s]: ...", fn)
                os.remove(fn)
                log.debug("removing [%s]: done.", fn)
                print(
                    f"""
project '{proj_name}' successfully unregistered

you can stop and disable the service with:

\t\tsystemctl --user disable --now 'podman-compose@{proj_name}'
"""
                )
            except OSError as e:
                log.error("failed to remove file %s: %s", fn, e)
                print(f"Failed to remove registration file for project '{proj_name}'")
                sys.exit(1)
        else:
            log.warning("registration file not found: %s", fn)
            print(f"Project '{proj_name}' is not registered")
    elif args.action in ("list", "ls"):
        ls = glob.glob(os.path.expanduser(f"~/{stacks_dir}/*.env"))
        for i in ls:
            print(os.path.basename(i[:-4]))
    elif args.action == "create-unit":
        fn = "/etc/systemd/user/podman-compose@.service"
        script = compose.executable
        out = f"""\
# {fn}

[Unit]
Description=%i rootless pod (podman-compose)

[Service]
Type=simple
EnvironmentFile=%h/{stacks_dir}/%i.env
ExecStartPre=-{script} up --no-start
ExecStartPre=/usr/bin/env podman pod start pod_%i
ExecStart={script} wait
ExecStop=/usr/bin/env podman pod stop pod_%i

[Install]
WantedBy=default.target
"""
        if os.access(os.path.dirname(fn), os.W_OK):
            log.debug("writing [%s]: ...", fn)
            with open(fn, "w", encoding="utf-8") as f:
                f.write(out)
            log.debug("writing [%s]: done.", fn)
            print(
                """
while in your project type `podman-compose systemd -a register`
"""
            )
        else:
            print(out)
            log.warning("Could not write to [%s], use 'sudo'", fn)
