# SPDX-License-Identifier: GPL-2.0-only
"""translate.resources 分支补测（ulimit 三形态、GPU 设备过滤四分支、
count/device_ids/all 三路径、CPU/内存/pids 的 v2/v3 取值与冲突）。"""

import pytest

from xuan_compose.translate.resources import (
    container_to_cpu_res_args,
    container_to_gpu_res_args,
    ulimit_to_ulimit_args,
)


class TestUlimitArgs:
    def test_none_is_noop(self) -> None:
        out: list[str] = []
        ulimit_to_ulimit_args(None, out)
        assert out == []

    def test_string_passthrough(self) -> None:
        out: list[str] = []
        ulimit_to_ulimit_args("host", out)
        assert out == ["--ulimit", "host"]

    def test_dict_maps_soft_hard(self) -> None:
        out: list[str] = []
        ulimit_to_ulimit_args({"nofile": {"soft": 1024, "hard": 2048}}, out)
        assert out == ["--ulimit", "nofile=1024:2048"]

    def test_list_of_assignments(self) -> None:
        out: list[str] = []
        ulimit_to_ulimit_args(["nofile=65536"], out)
        assert out == ["--ulimit", "nofile=65536"]


class TestGpuResArgs:
    @staticmethod
    def _gpu_args(devices: list[dict]) -> list[str]:
        out: list[str] = []
        cnt = {"deploy": {"resources": {"reservations": {"devices": devices}}}}
        container_to_gpu_res_args(cnt, out)
        return out

    def test_empty_devices_noop(self) -> None:
        assert self._gpu_args([]) == []

    def test_device_without_driver_skipped(self) -> None:
        assert self._gpu_args([{"capabilities": ["gpu"]}]) == []

    def test_device_without_capabilities_skipped(self) -> None:
        assert self._gpu_args([{"driver": "nvidia"}]) == []

    def test_non_nvidia_driver_skipped(self) -> None:
        assert self._gpu_args(
            [{"driver": "amd", "capabilities": ["gpu"], "count": "all"}]
        ) == []

    def test_nvidia_without_gpu_capability_skipped(self) -> None:
        assert self._gpu_args(
            [{"driver": "nvidia", "capabilities": ["compute"], "count": "all"}]
        ) == []

    def test_all_devices(self) -> None:
        out = self._gpu_args(
            [{"driver": "nvidia", "capabilities": ["gpu"], "count": "all"}]
        )
        assert out == ["--device", "nvidia.com/gpu=all", "--security-opt=label=disable"]

    def test_integer_count_expands(self) -> None:
        out = self._gpu_args(
            [{"driver": "nvidia", "capabilities": ["gpu"], "count": 2}]
        )
        assert out == [
            "--device",
            "nvidia.com/gpu=0",
            "--device",
            "nvidia.com/gpu=1",
            "--security-opt=label=disable",
        ]

    def test_explicit_device_ids(self) -> None:
        out = self._gpu_args(
            [{
                "driver": "nvidia",
                "capabilities": ["gpu"],
                "device_ids": ["0", "3"],
                "count": "all",
            }]
        )
        assert out == [
            "--device",
            "nvidia.com/gpu=0",
            "--device",
            "nvidia.com/gpu=3",
            "--security-opt=label=disable",
        ]

    def test_empty_device_ids_falls_back_to_count(self) -> None:
        out = self._gpu_args(
            [{
                "driver": "nvidia",
                "capabilities": ["gpu"],
                "device_ids": [],
                "count": 1,
            }]
        )
        assert out == ["--device", "nvidia.com/gpu=0", "--security-opt=label=disable"]


class TestCpuResArgs:
    def test_v2_fields(self) -> None:
        out: list[str] = []
        container_to_cpu_res_args(
            {"cpus": 1.5, "cpu_shares": 100, "mem_limit": "1G", "mem_reservation": "256M"},
            out,
        )
        assert ["--cpus", "1.5"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["--cpu-shares", "100"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["-m", "1g"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["--memory-reservation", "256m"] in [
            out[i : i + 2] for i in range(len(out) - 1)
        ]

    def test_v3_deploy_fields_take_effect(self) -> None:
        out: list[str] = []
        container_to_cpu_res_args(
            {
                "deploy": {
                    "resources": {
                        "limits": {"cpus": 2.0, "memory": "2G", "pids": 100},
                        "reservations": {"memory": "512M"},
                    }
                }
            },
            out,
        )
        assert "--cpus" in out and "2.0" in out
        assert "-m" in out and "2g" in out
        assert "--memory-reservation" in out and "512m" in out
        assert out[-2:] == ["--pids-limit", "100"]

    def test_pids_limit_standalone(self) -> None:
        out: list[str] = []
        container_to_cpu_res_args({"pids_limit": 42}, out)
        assert out == ["--pids-limit", "42"]

    def test_inconsistent_pids_limits_raises(self) -> None:
        with pytest.raises(ValueError, match="Inconsistent PIDs limit"):
            container_to_cpu_res_args(
                {"pids_limit": 42, "deploy": {"resources": {"limits": {"pids": 99}}}},
                [],
            )

    def test_consistent_dual_pids_declarations(self) -> None:
        out: list[str] = []
        container_to_cpu_res_args(
            {"pids_limit": 42, "deploy": {"resources": {"limits": {"pids": 42}}}},
            out,
        )
        assert out == ["--pids-limit", "42"]

    def test_no_resources_noop(self) -> None:
        out: list[str] = []
        container_to_cpu_res_args({}, out)
        assert out == []
