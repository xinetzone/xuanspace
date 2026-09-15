# SPDX-License-Identifier: GPL-2.0-only
"""secrets 翻译（翻译自 podman_compose.py 第 838-954 行）。

环境 secret 翻译为 ``--secret id=...,env=...``（build）或
``--secret project_name``（run）；文件 secret 在 run 时以只读 bind
卷形式挂入；external/name secret 透传 uid/gid/mode/type/target 选项。
"""

import os
from typing import Any

from ..logging_utils import log

__all__ = [
    "get_secret_args",
]


def get_secret_args(
    compose: Any,
    cnt: dict[str, Any],
    secret: str | dict[str, Any],
    podman_is_building: bool = False,
) -> list[str]:
    """
    podman_is_building: True if we are preparing arguments for an invocation of "podman build"
                        False if we are preparing for something else like "podman run"
    """
    assert compose.declared_secrets is not None

    secret_name = secret if isinstance(secret, str) else secret.get("source")
    if not secret_name or secret_name not in compose.declared_secrets.keys():
        raise ValueError(f'ERROR: undeclared secret: "{secret}", service: {cnt["_service"]}')
    declared_secret = compose.declared_secrets[secret_name]

    source_file = declared_secret.get("file")
    x_podman_relabel = declared_secret.get("x-podman.relabel")
    dest_file = ""
    secret_opts = ""

    secret_target = None
    secret_uid = None
    secret_gid = None
    secret_mode = None
    secret_type = None
    if isinstance(secret, dict):
        secret_target = secret.get("target")
        secret_uid = secret.get("uid")
        secret_gid = secret.get("gid")
        secret_mode = secret.get("mode")
        secret_type = secret.get("type")

    source_env = declared_secret.get("environment")
    if source_env:
        if podman_is_building:
            secret_id = secret_target if secret_target else secret_name
            return ["--secret", f"id={secret_id},env={source_env}"]

        assert compose.project_name is not None
        log.debug("mounting secret '%s'", secret_name)
        return ["--secret", f"{compose.project_name}_{secret_name}"]

    if source_file:
        # assemble path for source file first, because we need it for all cases
        basedir = compose.dirname
        source_file = os.path.realpath(os.path.join(basedir, os.path.expanduser(source_file)))

        if podman_is_building:
            # pass file secrets to "podman build" with param --secret
            if not secret_target:
                secret_id = secret_name
            elif "/" in secret_target:
                raise ValueError(
                    f'ERROR: Build secret "{secret_name}" has invalid target "{secret_target}". '
                    + "(Expected plain filename without directory as target.)"
                )
            else:
                secret_id = secret_target
            volume_ref = ["--secret", f"id={secret_id},src={source_file}"]
        else:
            # pass file secrets to "podman run" as volumes
            if not secret_target:
                dest_file = f"/run/secrets/{secret_name}"
            elif not secret_target.startswith("/"):
                sec = secret_target if secret_target else secret_name
                dest_file = f"/run/secrets/{sec}"
            else:
                dest_file = secret_target

            mount_options = "ro,rprivate,rbind"

            selinux_relabel_to_mount_option_map = {None: "", "z": ",z", "Z": ",Z"}
            try:
                mount_options += selinux_relabel_to_mount_option_map[x_podman_relabel]
            except KeyError as exc:
                raise ValueError(
                    f'ERROR: Run secret "{secret_name}" has invalid "relabel" option related '
                    + f' to SELinux "{x_podman_relabel}". Expected "z" "Z" or nothing.'
                ) from exc
            volume_ref = ["--volume", f"{source_file}:{dest_file}:{mount_options}"]

        if secret_uid or secret_gid or secret_mode:
            sec = secret_target if secret_target else secret_name
            log.warning(
                "WARNING: Service %s uses secret %s with uid, gid, or mode."
                + " These fields are not supported by this implementation of the Compose file",
                cnt["_service"],
                sec,
            )
        return volume_ref
    # v3.5 and up added external flag, earlier the spec
    # only required a name to be specified.
    # docker-compose does not support external secrets outside of swarm mode.
    # However accessing these via podman is trivial
    # since these commands are directly translated to
    # podman-create commands, albeit we can only support a 1:1 mapping
    # at the moment
    if declared_secret.get("external", False) or declared_secret.get("name"):
        secret_opts += f",uid={secret_uid}" if secret_uid else ""
        secret_opts += f",gid={secret_gid}" if secret_gid else ""
        secret_opts += f",mode={secret_mode}" if secret_mode else ""
        secret_opts += f",type={secret_type}" if secret_type else ""
        secret_opts += f",target={secret_target}" if secret_target else ""
        # having a custom name for the external secret is not supported
        ext_name = declared_secret.get("name")
        if ext_name and ext_name != secret_name:
            raise ValueError(
                f'ERROR: Custom name/target reference "{secret_name}" '
                f'for mounted external secret "{ext_name}" is not supported'
            )
        return ["--secret", f"{secret_name}{secret_opts}"]

    raise ValueError(
        'ERROR: unparsable secret: "{}", service: "{}"'.format(secret_name, cnt["_service"])
    )
