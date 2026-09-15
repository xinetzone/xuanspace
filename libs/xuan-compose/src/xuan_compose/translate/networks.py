# SPDX-License-Identifier: GPL-2.0-only
"""网络参数翻译（翻译自 podman_compose.py 第 551-562、1110-1341 行）。

包含网络创建参数、默认网络命名（含 ``default_net_name_compat`` 兼容模式）、
network_mode/network 两种形态到 ``podman --network`` 的翻译，以及
``assert_cnt_nets``——网络不存在时按描述自动创建，外部网络缺失则抛
用户友好错误。``compose`` 形参为 ``Any`` 鸭子类型（见 mounts 模块说明）。
"""

import sys
from typing import Any

from ..compat import is_list
from ..errors import PodmanComposeError
from ..logging_utils import log
from ..model import XPodmanSettingKey
from ..normalize import norm_as_list
from ..runner import CalledProcessError

__all__ = [
    "default_network_name_for_project",
    "get_network_create_args",
    "assert_cnt_nets",
    "get_net_args_from_network_mode",
    "get_net_args",
    "get_net_args_from_networks",
]


def default_network_name_for_project(compose: Any, net: str, is_ext: Any) -> str:
    if is_ext:
        return net

    assert compose.project_name is not None

    default_net_name_compat = compose.x_podman.get(
        XPodmanSettingKey.DEFAULT_NET_NAME_COMPAT, False
    )
    if default_net_name_compat is True:
        return compose.join_name_parts(compose.project_name.replace("-", ""), net)
    return compose.format_name(net)


def get_network_create_args(net_desc: dict[str, Any], proj_name: str, net_name: str) -> list[str]:
    args = [
        "create",
        "--label",
        f"io.podman.compose.project={proj_name}",
        "--label",
        f"com.docker.compose.project={proj_name}",
    ]
    # TODO: add more options here, like dns, ipv6, etc.
    labels = net_desc.get("labels", [])
    for item in norm_as_list(labels):
        args.extend(["--label", item])
    if net_desc.get("internal"):
        args.append("--internal")
    driver = net_desc.get("driver")
    if driver:
        args.extend(("--driver", driver))
    driver_opts = net_desc.get("driver_opts", {})
    for key, value in driver_opts.items():
        args.extend(("--opt", f"{key}={value}"))
    ipam = net_desc.get("ipam", {})
    ipam_driver = ipam.get("driver")
    if ipam_driver and ipam_driver != "default":
        args.extend(("--ipam-driver", ipam_driver))
    ipam_config_ls = ipam.get("config", [])
    if net_desc.get("enable_ipv6"):
        args.append("--ipv6")
    if net_desc.get("x-podman.disable_dns"):
        args.append("--disable-dns")
    if net_desc.get("x-podman.dns"):
        args.extend((
            "--dns",
            ",".join(norm_as_list(net_desc.get("x-podman.dns"))),
        ))
    if net_desc.get("x-podman.routes"):
        routes = norm_as_list(net_desc.get("x-podman.routes"))
        for route in routes:
            args.extend(["--route", route])

    if isinstance(ipam_config_ls, dict):
        ipam_config_ls = [ipam_config_ls]
    for ipam_config in ipam_config_ls:
        subnet = ipam_config.get("subnet")
        ip_range = ipam_config.get("ip_range")
        gateway = ipam_config.get("gateway")
        if subnet:
            args.extend(("--subnet", subnet))
        if ip_range:
            args.extend(("--ip-range", ip_range))
        if gateway:
            args.extend(("--gateway", gateway))
    args.append(net_name)

    return args


async def assert_cnt_nets(compose: Any, cnt: dict[str, Any]) -> None:
    """
    create missing networks
    """
    net = cnt.get("network_mode")
    if net:
        return

    assert compose.project_name is not None

    cnt_nets = cnt.get("networks")
    if cnt_nets and isinstance(cnt_nets, dict):
        cnt_nets = list(cnt_nets.keys())
    cnt_nets = norm_as_list(cnt_nets or compose.default_net)  # type: ignore[arg-type]
    for net in cnt_nets:
        net_desc = compose.networks[net] or {}
        is_ext = net_desc.get("external")
        ext_desc = is_ext if isinstance(is_ext, dict) else {}
        default_net_name = default_network_name_for_project(compose, net, is_ext)
        net_name = ext_desc.get("name") or net_desc.get("name") or default_net_name
        try:
            await compose.podman.output([], "network", ["exists", net_name])
        except CalledProcessError as e:
            if is_ext:
                raise PodmanComposeError(
                    f"External network [{net_name}] does not exist. "
                    f"Create it first with: podman network create '{net_name}'"
                ) from e
            args = get_network_create_args(net_desc, compose.project_name, net_name)
            await compose.podman.output([], "network", args)
            await compose.podman.output([], "network", ["exists", net_name])


def get_net_args_from_network_mode(compose: Any, cnt: dict[str, Any]) -> list[str]:
    net_args = []
    net = cnt.get("network_mode")
    assert isinstance(net, str)
    service_name = cnt["service_name"]

    if "networks" in cnt:
        raise ValueError(
            f"networks and network_mode must not be present in the same service [{service_name}]"
        )

    if net == "none":
        net_args.append("--network=none")
    elif net == "host":
        net_args.append(f"--network={net}")
    elif net.startswith("slirp4netns"):  # Note: podman-specific network mode
        net_args.append(f"--network={net}")
    elif net == "private":  # Note: podman-specific network mode
        net_args.append("--network=private")
    elif net.startswith("pasta"):  # Note: podman-specific network mode
        net_args.append(f"--network={net}")
    elif net.startswith("ns:"):  # Note: podman-specific network mode
        net_args.append(f"--network={net}")
    elif net.startswith("service:"):
        other_srv = net.split(":", 1)[1].strip()
        other_cnt = compose.container_names_by_service[other_srv][0]
        net_args.append(f"--network=container:{other_cnt}")
    elif net.startswith("container:"):
        other_cnt = net.split(":", 1)[1].strip()
        net_args.append(f"--network=container:{other_cnt}")
    elif net.startswith("bridge"):
        aliases_on_container = [service_name]
        if cnt.get("_aliases"):
            aliases_on_container.extend(cnt.get("_aliases"))  # type: ignore[arg-type]
        net_options = [f"alias={alias}" for alias in aliases_on_container]
        mac_address = cnt.get("mac_address")
        if mac_address:
            net_options.append(f"mac={mac_address}")

        net = f"{net}," if ":" in net else f"{net}:"  # type: ignore[operator]
        net_args.append(f"--network={net}{','.join(net_options)}")
    else:
        log.fatal("unknown network_mode [%s]", net)
        sys.exit(1)

    return net_args


def get_net_args(compose: Any, cnt: dict[str, Any]) -> list[str]:
    net = cnt.get("network_mode")
    if net:
        return get_net_args_from_network_mode(compose, cnt)

    return get_net_args_from_networks(compose, cnt)


def get_net_args_from_networks(compose: Any, cnt: dict[str, Any]) -> list[str]:
    net_args = []
    mac_address = cnt.get("mac_address")
    service_name = cnt["service_name"]

    aliases_on_container = [service_name]
    aliases_on_container.extend(cnt.get("_aliases", []))

    multiple_nets = cnt.get("networks", {})
    if not multiple_nets:
        if not compose.default_net:
            # The bridge mode in podman is using the `podman` network.
            # It seems weird, but we should keep this behavior to avoid
            # breaking changes.
            net_options = [f"alias={alias}" for alias in aliases_on_container]
            if mac_address:
                net_options.append(f"mac={mac_address}")
            net_args.append(f"--network=bridge:{','.join(net_options)}")
            return net_args

        multiple_nets = {compose.default_net: {}}

    # networks can be specified as a dict with config per network or as a plain list without
    # config.  Support both cases by converting the plain list to a dict with empty config.
    if is_list(multiple_nets):
        multiple_nets = {net: {} for net in multiple_nets}
    else:
        multiple_nets = {net: net_config or {} for net, net_config in multiple_nets.items()}

    # if a mac_address was specified on the container level, we need to check that it is not
    # specified on the network level as well
    if mac_address is not None:
        for net_config in multiple_nets.values():
            network_mac = net_config.get("mac_address", net_config.get("x-podman.mac_address"))
            if network_mac is not None:
                raise RuntimeError(
                    f"conflicting mac addresses {mac_address} and {network_mac}:"
                    "specifying mac_address on both container and network level "
                    "is not supported"
                )

    for net_, net_config_ in multiple_nets.items():
        net_desc = compose.networks.get(net_) or {}
        is_ext = net_desc.get("external")
        ext_desc: dict[str, Any] = is_ext if isinstance(is_ext, str) else {}  # type: ignore[assignment]
        default_net_name = default_network_name_for_project(compose, net_, is_ext)  # type: ignore[arg-type]
        net_name = ext_desc.get("name") or net_desc.get("name") or default_net_name

        interface_name = net_config_.get("x-podman.interface_name")
        ipv4 = net_config_.get("ipv4_address")
        ipv6 = net_config_.get("ipv6_address")
        # Note: mac_address is supported by compose spec now, and x-podman.mac_address
        # is only for backward compatibility
        # https://github.com/compose-spec/compose-spec/blob/main/05-services.md#mac_address
        mac = net_config_.get("mac_address", net_config_.get("x-podman.mac_address"))
        aliases_on_net = norm_as_list(net_config_.get("aliases", []))

        # if a mac_address was specified on the container level, apply it to the first network
        # This works for Python > 3.6, because dict insert ordering is preserved, so we are
        # sure that the first network we encounter here is also the first one specified by
        # the user
        if mac is None and mac_address is not None:
            mac = mac_address
            mac_address = None

        net_options = []
        if interface_name:
            net_options.append(f"interface_name={interface_name}")
        if ipv4:
            net_options.append(f"ip={ipv4}")
        if ipv6:
            net_options.append(f"ip6={ipv6}")
        if mac:
            net_options.append(f"mac={mac}")

        # Container level service aliases
        net_options.extend([f"alias={alias}" for alias in aliases_on_container])
        # network level service aliases
        if aliases_on_net:
            net_options.extend([f"alias={alias}" for alias in aliases_on_net])

        if net_options:
            net_args.append(f"--network={net_name}:" + ",".join(net_options))
        else:
            net_args.append(f"--network={net_name}")

    return net_args
