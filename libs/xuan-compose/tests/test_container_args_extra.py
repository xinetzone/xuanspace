# SPDX-License-Identifier: GPL-2.0-only
"""container_to_args 分支补测（移植的 78 例覆盖镜像/网络/挂载主路径；
本文件补齐安全/设备/DNS 等简单开关全字段、env_file 三种结局、裸环境名
回填、ipc 四分支、healthcheck 全形态、x-podman.* 扩展与类型错误——
上游 container_to_args 第 1344-1622 行真实分支，零真实 podman。"""

# pylint: disable=protected-access

import os
import tempfile
import unittest
from typing import Any

from xuan_compose.translate.container_args import container_to_args

from .test_container_to_args import create_compose_mock, get_minimal_container


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


def _pairs(out: list[str]) -> list[tuple[str, str]]:
    return [(out[i], out[i + 1]) for i in range(len(out) - 1)]


class TestSimpleFlags(unittest.IsolatedAsyncioTestCase):
    async def test_full_simple_flag_surface(self) -> None:
        c = create_compose_mock()
        cnt: dict[str, Any] = get_minimal_container()
        cnt.update(
            {
                "pod": "pod_x",
                "security_opt": ["seccomp=unconfined"],
                "annotations": ["a=1"],
                "read_only": True,
                "http_proxy": False,
                "labels": ["l=1"],
                "cap_add": ["NET_ADMIN"],
                "cap_drop": ["MKNOD"],
                "group_add": ["1000"],
                "devices": ["/dev/fuse"],
                "device_cgroup_rules": ["c 1:1 rw"],
                "dns": ["1.1.1.1"],
                "dns_opt": ["ndots:2"],
                "dns_search": ["example.com"],
                "tmpfs": ["/tmp"],
                "extra_hosts": ["host:1.2.3.4"],
                "expose": ["80"],
                "publishall": True,
                "userns_mode": "keep-id",
                "user": "1000",
                "working_dir": "/app",
                "hostname": "node-1",
                "shm_size": "256m",
                "stdin_open": True,
                "stop_signal": "SIGTERM",
                "tty": True,
                "privileged": True,
                "pid": "host",
                "pull_policy": "always",
                "restart": "on-failure",
                "init": True,
                "init-path": "/sbin/init",
                "platform": "linux/amd64",
                "runtime": "crun",
                "cpuset": "0-1",
            }
        )
        out = await container_to_args(c, cnt, detached=False)
        assert "-d" not in out
        for flag in (
            "--pod=pod_x",
            "--read-only",
            "--http-proxy=false",
            "-P",
            "--privileged",
            "--init",
            "--tty",
            "-i",
        ):
            assert flag in out, flag
        pairs = dict(_pairs(out))
        assert pairs["--security-opt"] == "seccomp=unconfined"
        assert pairs["--annotation"] == "a=1"
        assert pairs["--cap-add"] == "NET_ADMIN"
        assert pairs["--cap-drop"] == "MKNOD"
        assert pairs["--group-add"] == "1000"
        assert pairs["--device"] == "/dev/fuse"
        assert pairs["--dns"] == "1.1.1.1"
        assert pairs["--hostname"] == "node-1"
        assert pairs["--shm-size"] == "256m"
        assert pairs["--stop-signal"] == "SIGTERM"
        assert pairs["--pid"] == "host"
        assert "--pull=always" in out  # 单参数等号形式
        assert pairs["--restart"] == "on-failure"
        assert pairs["--init-path"] == "/sbin/init"
        assert pairs["--platform"] == "linux/amd64"
        assert pairs["--runtime"] == "crun"
        assert pairs["--cpuset-cpus"] == "0-1"
        assert pairs["--userns"] == "keep-id"

    async def test_dependency_requires_and_no_deps(self) -> None:
        c = create_compose_mock()
        c.container_names_by_service = {"db": ["proj_db_1", "proj_db_2"]}
        cnt = get_minimal_container()
        cnt["_deps"] = [_Named("db")]
        out = await container_to_args(c, cnt)
        assert "--requires=proj_db_1,proj_db_2" in out
        out_nd = await container_to_args(c, cnt, no_deps=True)
        assert not any(x.startswith("--requires") for x in out_nd)

    async def test_sysctls_dict_list_and_invalid(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["sysctls"] = {"net.core.somaxconn": "1024"}
        out = await container_to_args(c, cnt)
        assert dict(_pairs(out))["--sysctl"] == "net.core.somaxconn=1024"
        cnt2 = get_minimal_container()
        cnt2["sysctls"] = ["a=b"]
        assert dict(_pairs(await container_to_args(c, cnt2)))["--sysctl"] == "a=b"
        cnt3 = get_minimal_container()
        cnt3["sysctls"] = 5
        with self.assertRaises(TypeError):
            await container_to_args(c, cnt3)

    async def test_pull_policy_build_is_skipped(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["pull_policy"] = "build"
        out = await container_to_args(c, cnt)
        assert not any(x.startswith("--pull") for x in out)

    async def test_stop_grace_invalid_omitted(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["stop_grace_period"] = "garbage"
        out = await container_to_args(c, cnt)
        assert not any(x.startswith("--stop-timeout") for x in out)
        cnt["stop_grace_period"] = "30s"
        assert dict(_pairs(await container_to_args(c, cnt)))["--stop-timeout"] == "30"

    async def test_entrypoint_string_split_and_command_forms(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["entrypoint"] = "/bin/sh -c"
        cnt["command"] = "echo hi"
        out = await container_to_args(c, cnt)
        pairs = dict(_pairs(out))
        assert pairs["--entrypoint"] == '["/bin/sh", "-c"]'
        assert out[-2:] == ["echo", "hi"]
        cnt2 = get_minimal_container()
        cnt2["command"] = ["echo", "list"]
        assert (await container_to_args(c, cnt2))[-2:] == ["echo", "list"]


class TestEnvFilesAndEnvironment(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.base = tempfile.mkdtemp(prefix="xuan-envfiles-")

    async def test_env_file_loaded_required_optional_missing(self) -> None:
        with open(os.path.join(self.base, "real.env"), "w", encoding="utf-8") as fh:
            fh.write("FROM_ENV=present\n")
        c = create_compose_mock()
        c.dirname = self.base
        cnt = get_minimal_container()
        cnt["env_file"] = [
            "real.env",
            {"path": "missing.env", "required": False},
        ]
        out = await container_to_args(c, cnt)
        assert ("-e", "FROM_ENV=present") in _pairs(out)

    async def test_missing_required_env_file_raises(self) -> None:
        c = create_compose_mock()
        c.dirname = self.base
        cnt = get_minimal_container()
        cnt["env_file"] = "nope.env"
        with self.assertRaises(ValueError):
            await container_to_args(c, cnt)

    async def test_bare_env_name_resolved_from_compose_environ(self) -> None:
        c = create_compose_mock()
        c.environ = {"INHERITED": "from-host"}
        cnt = get_minimal_container()
        cnt["environment"] = {"A": "1", "INHERITED": None, "UNSET": None}
        out = await container_to_args(c, cnt)
        pairs = dict(_pairs(out))
        assert pairs["-e"] in {"A=1", "INHERITED=from-host"}
        joined = [x for pair in _pairs(out) if pair[0] == "-e" for x in pair]
        assert "-e" in joined and "INHERITED=from-host" in joined
        # UNSET 在 compose.environ 中无值 → 不透传
        assert all("UNSET" not in x for x in out)


class TestIpc(unittest.IsolatedAsyncioTestCase):
    async def test_valid_plain_modes(self) -> None:
        for mode in ("", "host", "none", "private", "shareable"):
            c = create_compose_mock()
            cnt = get_minimal_container()
            cnt["ipc"] = mode
            out = await container_to_args(c, cnt)
            assert dict(_pairs(out))["--ipc"] == mode

    async def test_ipc_container_service_resolved(self) -> None:
        c = create_compose_mock()
        c.container_names_by_service = {"db": ["proj_db_1"]}
        cnt = get_minimal_container()
        cnt["ipc"] = "service:db"
        assert dict(_pairs(await container_to_args(c, cnt)))["--ipc"] == (
            "container:proj_db_1"
        )

    async def test_ipc_unknown_service_raises(self) -> None:
        c = create_compose_mock()
        c.container_names_by_service = {}
        cnt = get_minimal_container()
        cnt["ipc"] = "service:ghost"
        with self.assertRaises(ValueError):
            await container_to_args(c, cnt)

    async def test_ipc_invalid_strings(self) -> None:
        c = create_compose_mock()
        for bad in ("bogus", "container:", "ns:"):
            cnt = get_minimal_container()
            cnt["ipc"] = bad
            with self.assertRaises(ValueError):
                await container_to_args(c, cnt)

    async def test_ipc_non_string_raises(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["ipc"] = 5
        with self.assertRaises(ValueError):
            await container_to_args(c, cnt)


class TestHealthcheck(unittest.IsolatedAsyncioTestCase):
    async def _args_with(self, hc: dict | list | str | int) -> list[str]:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["healthcheck"] = hc
        return await container_to_args(c, cnt)

    async def test_disable_sets_none(self) -> None:
        out = await self._args_with({"disable": True})
        assert "--no-healthcheck" in out

    async def test_string_test_becomes_cmd_shell(self) -> None:
        out = await self._args_with({"test": "curl -f http://localhost"})
        pairs = dict(_pairs(out))
        assert pairs["--health-cmd"] == '["CMD-SHELL", "curl -f http://localhost"]'

    async def test_list_forms(self) -> None:
        assert "--no-healthcheck" in await self._args_with({"test": ["NONE"]})
        pairs = dict(_pairs(await self._args_with({"test": ["CMD", "curl", "x"]})))
        assert pairs["--health-cmd"] == '["curl", "x"]'
        pairs = dict(_pairs(await self._args_with({"test": ["CMD-SHELL", "echo x"]})))
        assert pairs["--health-cmd"] == '["echo x"]'

    async def test_cmd_shell_requires_single_arg(self) -> None:
        with self.assertRaises(ValueError):
            await self._args_with({"test": ["CMD-SHELL", "a", "b"]})

    async def test_unknown_test_type(self) -> None:
        with self.assertRaises(ValueError):
            await self._args_with({"test": ["BOGUS", "x"]})

    async def test_test_must_be_str_or_list(self) -> None:
        with self.assertRaises(ValueError):
            await self._args_with({"test": 5})

    async def test_timing_fields_and_retries(self) -> None:
        out = await self._args_with(
            {
                "interval": "10s",
                "timeout": "2s",
                "start_period": "30s",
                "start_interval": "5s",
                "retries": 3,
            }
        )
        pairs = dict(_pairs(out))
        assert pairs["--health-interval"] == "10s"
        assert pairs["--health-timeout"] == "2s"
        assert pairs["--health-start-period"] == "30s"
        assert pairs["--health-startup-interval"] == "5s"
        assert pairs["--health-retries"] == "3"

    async def test_healthcheck_must_be_mapping(self) -> None:
        with self.assertRaises(ValueError):
            await self._args_with(["not", "mapping"])  # type: ignore[arg-type]


class TestXPodmanExtensions(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_x_podman_section_rejected(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["x-podman"] = {"uidmaps": []}
        with self.assertRaisesRegex(ValueError, "migrated"):
            await container_to_args(c, cnt)

    async def test_uidmaps_gidmaps_no_hosts_passwd(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt.update(
            {
                "x-podman.uidmaps": ["0:0:1"],
                "x-podman.gidmaps": ["0:0:1"],
                "x-podman.no_hosts": True,
                "x-podman.passwd": False,
            }
        )
        out = await container_to_args(c, cnt)
        pairs = dict(_pairs(out))
        assert pairs["--uidmap"] == "0:0:1"
        assert pairs["--gidmap"] == "0:0:1"
        assert "--no-hosts" in out
        assert "--passwd=false" in out  # 等号单参数形式

    async def test_rootfs_mode_drops_image(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        cnt["x-podman.rootfs"] = "/path/to/rootfs"
        out = await container_to_args(c, cnt)
        assert dict(_pairs(out))["--rootfs"] == "/path/to/rootfs"
        assert "busybox" not in out  # rootfs 模式不追加镜像名
