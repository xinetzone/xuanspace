# SPDX-License-Identifier: GPL-2.0-only
"""端口描述翻译（翻译自 podman_compose.py 第 1073-1107 行）。"""

from typing import Any

__all__ = [
    "port_dict_to_str",
    "norm_ports",
]


def port_dict_to_str(port_desc: dict[str, Any]) -> str:
    # NOTE: `mode: host|ingress` is ignored
    cnt_port = port_desc.get("target")
    published = port_desc.get("published", "")
    host_ip = port_desc.get("host_ip")
    protocol = port_desc.get("protocol", "tcp")
    if not cnt_port:
        raise ValueError("target container port must be specified")
    if host_ip:
        ret = f"{host_ip}:{published}:{cnt_port}"
    else:
        ret = f"{published}:{cnt_port}" if published else f"{cnt_port}"
    if protocol != "tcp":
        ret += f"/{protocol}"
    return ret


def norm_ports(
    ports_in: None | str | list[str | dict[str, Any] | int] | dict[str, Any] | int,
) -> list[str]:
    if not ports_in:
        ports_in = []
    if isinstance(ports_in, str):
        ports_in = [ports_in]
    assert isinstance(ports_in, list)
    ports_out = []
    for port in ports_in:
        if isinstance(port, dict):
            port = port_dict_to_str(port)
        elif isinstance(port, int):
            port = str(port)
        elif not isinstance(port, str):
            raise TypeError("port should be either string or dict")
        ports_out.append(port)
    return ports_out
