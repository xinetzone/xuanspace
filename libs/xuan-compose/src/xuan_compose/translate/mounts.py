# SPDX-License-Identifier: GPL-2.0-only
"""挂载点翻译（翻译自 podman_compose.py 第 160-257、582-799 行）。

包含短挂载语法解析、长格式修正、bind/volume/tmpfs/image/glob 各类型到
``podman --mount/--volume/--tmpfs`` 参数的翻译，以及卷存在性断言
（``assert_volume``：bind 源目录按需创建，命名卷 inspect 失败时 create）。

``compose`` 形参一律 ``Any`` 鸭子类型：翻译层不反向依赖 T6 引擎层，
调用方按上游运行时契约提供 ``project_name`` / ``dirname`` / ``vols`` /
``format_name`` / ``podman.output`` / ``prefer_volume_over_mount``。
"""

import hashlib
import os
import re
from typing import Any

from ..compat import filteri
from ..errors import PodmanComposeError
from ..logging_utils import log
from ..normalize import norm_as_list
from ..runner import CalledProcessError

__all__ = [
    "parse_short_mount",
    "fix_mount_dict",
    "assert_volume",
    "mount_desc_to_mount_args",
    "mount_desc_to_volume_args",
    "get_mnt_dict",
    "get_mount_args",
]

dir_re = re.compile(r"^[~/\.]")
propagation_re = re.compile(
    "^(?:z|Z|O|U|r?shared|r?slave|r?private|r?unbindable|r?bind|(?:no)?(?:exec|dev|suid))$"
)


def parse_short_mount(mount_str: str, basedir: str) -> dict[str, Any]:
    mount_a = mount_str.split(":")
    mount_opt_dict: dict[str, Any] = {}
    mount_opt = None
    if len(mount_a) == 1:
        # Anonymous: Just specify a path and let the engine create the volume
        # - /var/lib/mysql
        mount_src, mount_dst = None, mount_str
    elif len(mount_a) == 2:
        mount_src, mount_dst = mount_a
        # dest must start with / like /foo:/var/lib/mysql
        # otherwise it's option like /var/lib/mysql:rw
        if not mount_dst.startswith("/"):
            mount_dst, mount_opt = mount_a
            mount_src = None
    elif len(mount_a) == 3:
        mount_src, mount_dst, mount_opt = mount_a
    else:
        raise ValueError("could not parse mount " + mount_str)
    if mount_src and dir_re.match(mount_src):
        # Specify an absolute path mapping
        # - /opt/data:/var/lib/mysql
        # Path on the host, relative to the Compose file
        # - ./cache:/tmp/cache
        # User-relative path
        # - ~/configs:/etc/configs/:ro
        mount_type = "bind"
        if os.name != "nt" or (os.name == "nt" and ".sock" not in mount_src):
            mount_src = os.path.abspath(os.path.join(basedir, os.path.expanduser(mount_src)))
    else:
        # Named volume
        # - datavolume:/var/lib/mysql
        mount_type = "volume"
    mount_opts = filteri((mount_opt or "").split(","))
    propagation_opts = []
    for opt in mount_opts:
        if opt == "ro":
            mount_opt_dict["read_only"] = True
        elif opt == "rw":
            mount_opt_dict["read_only"] = False
        elif opt in ("consistent", "delegated", "cached"):
            mount_opt_dict["consistency"] = opt
        elif propagation_re.match(opt):
            propagation_opts.append(opt)
        else:
            # TODO: ignore
            raise ValueError("unknown mount option " + opt)
    mount_opt_dict["bind"] = {"propagation": ",".join(propagation_opts)}
    return {
        "type": mount_type,
        "source": mount_src,
        "target": mount_dst,
        **mount_opt_dict,
    }


# NOTE: if a named volume is used but not defined it
# gives ERROR: Named volume "abc" is used in service "xyz"
#   but no declaration was found in the volumes section.
# unless it's anonymous-volume


def fix_mount_dict(compose: Any, mount_dict: dict[str, Any], srv_name: str) -> dict[str, Any]:
    """
    in-place fix mount dictionary to:
    - define _vol to be the corresponding top-level volume
    - if name is missing it would be source prefixed with project
    - if no source it would be generated
    """
    # if already applied nothing to do
    assert compose.project_name is not None

    if "_vol" in mount_dict:
        return mount_dict
    if mount_dict["type"] == "volume":
        vols = compose.vols
        source = mount_dict.get("source")
        vol = (vols.get(source, {}) or {}) if source else {}  # type: ignore[union-attr]
        name = vol.get("name")
        mount_dict["_vol"] = vol
        # handle anonymous or implied volume
        if not source:
            # missing source
            vol["name"] = compose.format_name(
                srv_name,
                hashlib.sha256(mount_dict["target"].encode("utf-8")).hexdigest(),
            )
        elif not name:
            external = vol.get("external")
            if isinstance(external, dict):
                vol["name"] = external.get("name", f"{source}")
            elif external:
                vol["name"] = f"{source}"
            else:
                vol["name"] = f"{compose.project_name}_{source}"
    return mount_dict


async def assert_volume(compose: Any, mount_dict: dict[str, Any]) -> None:
    """
    inspect volume to get directory
    create volume if needed
    """
    vol = mount_dict.get("_vol")
    if mount_dict["type"] == "bind":
        basedir = os.path.realpath(compose.dirname)
        mount_src = mount_dict["source"]
        mount_src = os.path.abspath(os.path.join(basedir, os.path.expanduser(mount_src)))
        if not os.path.exists(mount_src):
            bind_opts = mount_dict.get("bind", {})
            if "create_host_path" in bind_opts and not bind_opts["create_host_path"]:
                raise ValueError(
                    "invalid mount config for type 'bind': bind source path does not exist: "
                    f"{mount_src}"
                )
            try:
                os.makedirs(mount_src, exist_ok=True)
            except OSError:
                pass
        mount_dict["source"] = mount_src
        return
    if mount_dict["type"] != "volume" or not vol or not vol.get("name"):
        return
    vol_name = vol["name"]
    is_ext = vol.get("external")
    log.debug("podman volume inspect %s || podman volume create %s", vol_name, vol_name)
    # TODO: might move to using "volume list"
    # podman volume list --format '{{.Name}}\t{{.MountPoint}}' \
    #     -f 'label=io.podman.compose.project=HERE'
    try:
        await compose.podman.output([], "volume", ["inspect", vol_name])

    except CalledProcessError as e:
        if is_ext:
            raise PodmanComposeError(
                f"External volume [{vol_name}] does not exist. "
                f"Create it first with: podman volume create '{vol_name}'"
            ) from e
        labels = vol.get("labels", [])
        args = [
            "create",
            "--label",
            f"io.podman.compose.project={compose.project_name}",
            "--label",
            f"com.docker.compose.project={compose.project_name}",
        ]
        for item in norm_as_list(labels):
            args.extend(["--label", item])
        driver = vol.get("driver")
        if driver:
            args.extend(["--driver", driver])
        driver_opts = vol.get("driver_opts", {})
        for opt, value in driver_opts.items():
            args.extend(["--opt", f"{opt}={value}"])
        args.append(vol_name)
        await compose.podman.output([], "volume", args)
        await compose.podman.output([], "volume", ["inspect", vol_name])


def mount_desc_to_mount_args(mount_desc: dict[str, Any]) -> str:
    mount_type: str | None = mount_desc.get("type")
    assert mount_type is not None
    vol = mount_desc.get("_vol") if mount_type == "volume" else None
    source = vol["name"] if vol else mount_desc.get("source")
    target = mount_desc["target"]
    opts = []
    if mount_desc.get(mount_type, None):
        # TODO: we might need to add mount_dict[mount_type]["propagation"] = "z"
        mount_prop = mount_desc.get(mount_type, {}).get("propagation")
        if mount_prop:
            opts.append(f"{mount_type}-propagation={mount_prop}")
    if mount_desc.get("read_only", False):
        opts.append("ro")
    if mount_type == "tmpfs":
        tmpfs_opts = mount_desc.get("tmpfs", {})
        tmpfs_size = tmpfs_opts.get("size")
        if tmpfs_size:
            opts.append(f"tmpfs-size={tmpfs_size}")
        tmpfs_mode = tmpfs_opts.get("mode")
        if tmpfs_mode:
            opts.append(f"tmpfs-mode={tmpfs_mode}")
    if mount_type == "bind":
        bind_opts = mount_desc.get("bind", {})
        selinux = bind_opts.get("selinux")
        if selinux is not None:
            opts.append(selinux)

    # According to compose specifications https://docs.docker.com/reference/compose-file/services/#volumes
    # subpath can be used in image and volume mount type
    if mount_type in ["volume", "image"] and mount_desc.get(mount_type):
        subpath = mount_desc.get(mount_type, {}).get("subpath")
        if subpath is not None:
            opts.append(f"subpath={subpath}")

    opts_str = ",".join(opts)
    if mount_type == "bind":
        return f"type=bind,source={source},destination={target},{opts_str}".rstrip(",")
    if mount_type == "glob":
        return f"type=glob,source={source},destination={target},{opts_str}".rstrip(",")
    if mount_type == "volume":
        return f"type=volume,source={source},destination={target},{opts_str}".rstrip(",")
    if mount_type == "tmpfs":
        return f"type=tmpfs,destination={target},{opts_str}".rstrip(",")
    if mount_type == "image":
        return f"type=image,source={source},destination={target},{opts_str}".rstrip(",")
    raise ValueError("unknown mount type:" + mount_type)


def mount_desc_to_volume_args(mount_desc: dict[str, Any], srv_name: str) -> str:
    mount_type = mount_desc["type"]
    if mount_type not in ("bind", "volume", "glob"):
        raise ValueError("unknown mount type:" + mount_type)
    vol = mount_desc.get("_vol") if mount_type == "volume" else None
    source = vol["name"] if vol else mount_desc.get("source")
    if not source:
        raise ValueError(f"missing mount source for {mount_type} on {srv_name}")
    target = mount_desc["target"]
    opts: list[str] = []

    propagations = set(filteri(mount_desc.get(mount_type, {}).get("propagation", "").split(",")))
    if mount_type != "bind":
        propagations.update(filteri(mount_desc.get("bind", {}).get("propagation", "").split(",")))
    opts.extend(propagations)
    # --volume, -v[=[[SOURCE-VOLUME|HOST-DIR:]CONTAINER-DIR[:OPTIONS]]]
    # [rw|ro]
    # [z|Z]
    # [[r]shared|[r]slave|[r]private]|[r]unbindable
    # [[r]bind]
    # [noexec|exec]
    # [nodev|dev]
    # [nosuid|suid]
    # [O]
    # [U]
    read_only = mount_desc.get("read_only")
    if read_only is not None:
        opts.append("ro" if read_only else "rw")
    if mount_type == "bind":
        bind_opts = mount_desc.get("bind", {})
        selinux = bind_opts.get("selinux")
        if selinux is not None:
            opts.append(selinux)

    args = f"{source}:{target}"
    if opts:
        args += ":" + ",".join(opts)
    return args


def get_mnt_dict(compose: Any, cnt: dict[str, Any], volume: str | dict[str, Any]) -> dict[str, Any]:
    srv_name = cnt["_service"]
    basedir = compose.dirname
    if isinstance(volume, str):
        volume = parse_short_mount(volume, basedir)
    return fix_mount_dict(compose, volume, srv_name)


async def get_mount_args(
    compose: Any, cnt: dict[str, Any], volume: str | dict[str, Any]
) -> list[str]:
    volume = get_mnt_dict(compose, cnt, volume)
    srv_name = cnt["_service"]
    mount_type = volume["type"]
    # By default, mount using -v is actually preferred over --mount.
    # In some case, options can only be set using --mount.
    # --mount is forced for type set in mount_over_volume_needed var.
    #
    mount_over_volume_needed = {"image", "glob", "volume"}
    await assert_volume(compose, volume)
    if compose.prefer_volume_over_mount and mount_type not in mount_over_volume_needed:
        if mount_type == "tmpfs":
            # TODO: --tmpfs /tmp:rw,size=787448k,mode=1777
            args = volume["target"]
            tmpfs_opts = volume.get("tmpfs", {})
            opts = []
            size = tmpfs_opts.get("size")
            if size:
                opts.append(f"size={size}")
            mode = tmpfs_opts.get("mode")
            if mode:
                opts.append(f"mode={mode}")
            if opts:
                args += ":" + ",".join(opts)
            return ["--tmpfs", args]
        args = mount_desc_to_volume_args(volume, srv_name)
        return ["-v", args]
    args = mount_desc_to_mount_args(volume)
    return ["--mount", args]
