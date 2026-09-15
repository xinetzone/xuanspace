# SPDX-License-Identifier: GPL-2.0-only
"""T7 新增：命令层可纯 mock 分支的 handler 行为测试（上半）。

全部夹具为 fake podman + mock compose，零真实子进程，覆盖
version/pull/push/ps/config/port/pause/unpause/kill/stats/images/
wait/logs/ls/systemd 十四组命令。断言对齐上游 podman_compose.py
对应 handler 的逐行行为，不顺手修正上游怪癖（如 ls 的 except:break）。
"""

import asyncio
import io
import json
from argparse import Namespace
from contextlib import redirect_stdout
from unittest import IsolatedAsyncioTestCase, mock

from xuan_compose.commands import lifecycle
from xuan_compose.commands.inspect import (
    compose_config,
    compose_images,
    compose_port,
    compose_ps,
    compose_stats,
    list_running_projects,
)
from xuan_compose.commands.logs import compose_logs
from xuan_compose.commands.pullpush import compose_pull, compose_push
from xuan_compose.commands.systemd import compose_systemd
from xuan_compose.commands.version import compose_version


def _podman(run_return: int = 0) -> mock.Mock:
    podman = mock.Mock()
    podman.run = mock.AsyncMock(return_value=run_return)
    podman.output = mock.AsyncMock(return_value=b"")
    podman.exec = mock.Mock()
    return podman


class TestVersion(IsolatedAsyncioTestCase):
    async def test_short_prints_package_version_only(self) -> None:
        from xuan_compose import __version__

        compose = mock.Mock(podman=_podman())
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_version(compose, Namespace(short=True, format="pretty"))
        assert buf.getvalue().strip() == __version__
        compose.podman.run.assert_not_called()

    async def test_json_format(self) -> None:
        from xuan_compose import __version__

        compose = mock.Mock(podman=_podman())
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_version(compose, Namespace(short=False, format="json"))
        assert json.loads(buf.getvalue()) == {"version": __version__}
        compose.podman.run.assert_not_called()

    async def test_default_runs_podman_version(self) -> None:
        compose = mock.Mock(podman=_podman())
        await compose_version(compose, Namespace(short=False, format="pretty"))
        compose.podman.run.assert_awaited_once_with(["--version"], "", [])


class TestPullPush(IsolatedAsyncioTestCase):
    def _compose(self) -> mock.Mock:
        compose = mock.Mock()
        compose.podman = _podman()
        compose.containers = [
            {"_service": "a", "image": "ghcr.io/a:latest"},
            {"_service": "a2", "image": "ghcr.io/a:latest"},
            {"_service": "b", "image": "localhost/b:latest"},
            {"_service": "c", "image": "cimg"},  # 无 build 的普通远端名
        ]
        return compose

    async def test_pull_dedupes_and_skips_local_images(self) -> None:
        compose = self._compose()
        result = await compose_pull(compose, Namespace(services=[], force_local=False))
        assert result == 0
        pulled = {call.args[2][0] for call in compose.podman.run.await_args_list}
        assert pulled == {"ghcr.io/a:latest", "cimg"}
        assert compose.podman.run.await_count == 2

    async def test_pull_force_local_includes_local_images(self) -> None:
        compose = self._compose()
        await compose_pull(compose, Namespace(services=[], force_local=True))
        pulled = {call.args[2][0] for call in compose.podman.run.await_args_list}
        assert "localhost/b:latest" in pulled

    async def test_pull_services_filter(self) -> None:
        compose = self._compose()
        await compose_pull(compose, Namespace(services=["a"], force_local=True))
        pulled = [call.args[2][0] for call in compose.podman.run.await_args_list]
        assert pulled == ["ghcr.io/a:latest"]

    async def test_pull_propagates_nonzero_status(self) -> None:
        compose = self._compose()
        compose.podman.run = mock.AsyncMock(return_value=1)
        assert await compose_pull(compose, Namespace(services=[], force_local=True)) == 1

    async def test_push_only_services_with_build(self) -> None:
        compose = self._compose()
        compose.containers[0]["build"] = {"context": "."}
        await compose_push(compose, Namespace(services=[]))
        pushed = [call.args[2][0] for call in compose.podman.run.await_args_list]
        assert pushed == ["ghcr.io/a:latest"]

    async def test_push_respects_services_filter(self) -> None:
        compose = self._compose()
        compose.containers[0]["build"] = {}
        compose.containers[1]["build"] = {}
        await compose_push(compose, Namespace(services=["a2"]))
        pushed = [call.args[2][0] for call in compose.podman.run.await_args_list]
        assert pushed == ["ghcr.io/a:latest"]
        assert len(pushed) == 1


class TestPs(IsolatedAsyncioTestCase):
    async def test_quiet_uses_id_format(self) -> None:
        compose = mock.Mock(podman=_podman(), project_name="proj")
        await compose_ps(compose, Namespace(quiet=True, format=None))
        args = compose.podman.run.await_args.args[2]
        assert args[:3] == ["-a", "--filter", "label=io.podman.compose.project=proj"]
        assert args[-2:] == ["--format", "{{.ID}}"]

    async def test_custom_format_ignored_when_quiet(self) -> None:
        compose = mock.Mock(podman=_podman(), project_name="proj")
        await compose_ps(compose, Namespace(quiet=True, format="{{.Names}}"))
        args = compose.podman.run.await_args.args[2]
        assert "{{.ID}}" in args

    async def test_custom_format_passthrough(self) -> None:
        compose = mock.Mock(podman=_podman(), project_name="proj")
        await compose_ps(compose, Namespace(quiet=False, format="{{.Names}}"))
        args = compose.podman.run.await_args.args[2]
        assert args[-2:] == ["--format", "{{.Names}}"]


class TestConfig(IsolatedAsyncioTestCase):
    async def test_lists_service_names(self) -> None:
        compose = mock.Mock(services={"a": {}, "b": {}}, merged_yaml="YAML")
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_config(compose, Namespace(services=True, quiet=False))
        assert buf.getvalue().split() == ["a", "b"]

    async def test_quiet_services_lists_nothing(self) -> None:
        compose = mock.Mock(services={"a": {}}, merged_yaml="YAML")
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_config(compose, Namespace(services=True, quiet=True))
        assert buf.getvalue() == ""

    async def test_prints_merged_yaml(self) -> None:
        compose = mock.Mock(services={"a": {}}, merged_yaml="YAML")
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_config(compose, Namespace(services=False, quiet=False))
        assert buf.getvalue().strip() == "YAML"


class TestPort(IsolatedAsyncioTestCase):
    async def test_extracts_host_port(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "NetworkSettings": {
                            "Ports": {"8080/tcp": [{"HostPort": "18080"}]}
                        }
                    }
                ]
            ).encode()
        )
        compose = mock.Mock(podman=podman, container_names_by_service={"a": ["proj_a_1"]})
        compose.assert_services = mock.Mock()
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_port(compose, Namespace(service="a", index=1, private_port=8080, protocol="tcp"))
        assert buf.getvalue().strip() == "18080"


class TestPauseUnpause(IsolatedAsyncioTestCase):
    def _compose(self) -> mock.Mock:
        compose = mock.Mock()
        compose.podman = _podman()
        compose.container_names_by_service = {"a": ["proj_a_1", "proj_a_2"], "b": ["proj_b_1"]}
        return compose

    async def test_pause_defaults_to_all_services(self) -> None:
        from xuan_compose.commands.lifecycle import compose_pause

        compose = self._compose()
        await compose_pause(compose, Namespace(services=[]))
        targets = compose.podman.run.await_args.args[2]
        assert set(targets) == {"proj_a_1", "proj_a_2", "proj_b_1"}

    async def test_unpause_selected_services(self) -> None:
        from xuan_compose.commands.lifecycle import compose_unpause

        compose = self._compose()
        await compose_unpause(compose, Namespace(services=["b"]))
        assert compose.podman.run.await_args.args[2] == ["proj_b_1"]


class TestKill(IsolatedAsyncioTestCase):
    def _compose(self) -> mock.Mock:
        compose = mock.Mock()
        compose.podman = _podman()
        compose.container_names_by_service = {"a": ["proj_a_1"], "b": ["proj_b_1"]}
        return compose

    async def test_fatal_without_services_or_all(self) -> None:
        compose = self._compose()
        # 上游 log.fatal + sys.exit()（无退出码）
        with self.assertRaises(SystemExit):
            await lifecycle.compose_kill(compose, Namespace(services=[], all=False, signal=None))
        compose.podman.run.assert_not_called()

    async def test_kill_all_targets_every_container(self) -> None:
        compose = self._compose()
        await lifecycle.compose_kill(
            compose, Namespace(services=[], all=True, signal=None)
        )
        args = compose.podman.run.await_args.args[2]
        assert set(args) == {"proj_a_1", "proj_b_1"}

    async def test_kill_selected_with_signal(self) -> None:
        compose = self._compose()
        await lifecycle.compose_kill(
            compose, Namespace(services=["a"], all=False, signal="HUP")
        )
        args = compose.podman.run.await_args.args[2]
        assert args[:2] == ["--signal", "HUP"]
        assert args[-1] == "proj_a_1"


class TestStats(IsolatedAsyncioTestCase):
    def _compose(self) -> mock.Mock:
        compose = mock.Mock()
        compose.podman = _podman()
        compose.container_names_by_service = {"a": ["proj_a_1"], "b": ["proj_b_1"]}
        return compose

    async def test_assembles_streaming_args(self) -> None:
        compose = self._compose()
        await compose_stats(
            compose,
            Namespace(
                services=[],
                interval="2s",
                format="json",
                no_reset=True,
                no_stream=True,
            ),
        )
        args = compose.podman.run.await_args.args[2]
        assert args == [
            "--interval", "2s",
            "--format", "json",
            "--no-reset",
            "--no-stream",
            "proj_a_1",
            "proj_b_1",
        ]

    async def test_keyboard_interrupt_swallowed(self) -> None:
        compose = self._compose()
        compose.podman.run = mock.AsyncMock(side_effect=KeyboardInterrupt)
        # 上游吞掉 Ctrl+C 静默返回
        await compose_stats(compose, Namespace(services=["a"], interval=None, format=None,
                                               no_reset=False, no_stream=False))


class TestImages(IsolatedAsyncioTestCase):
    async def test_quiet_lists_image_ids(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(return_value=b"deadbeef\n")
        compose = mock.Mock(podman=podman)
        compose.containers = [{"name": "proj_a_1", "image": "ghcr.io/a:latest"}]
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_images(compose, Namespace(quiet=True))
        assert "deadbeef" in buf.getvalue()
        assert podman.output.await_args.args[2] == ["--quiet", "ghcr.io/a:latest"]

    async def test_table_output_has_header(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(
            return_value=b"ghcr.io/a latest deadbeef 10MB\n"
        )
        compose = mock.Mock(podman=podman)
        compose.containers = [{"name": "proj_a_1", "image": "ghcr.io/a:latest"}]
        buf = io.StringIO()
        with redirect_stdout(buf):
            await compose_images(compose, Namespace(quiet=False))
        header = buf.getvalue().splitlines()[0]
        assert header.split("\t")[0].strip() == "CONTAINER"
        fmt_arg = podman.output.await_args.args[2]
        assert fmt_arg[1].startswith("table proj_a_1 ")


class TestWait(IsolatedAsyncioTestCase):
    async def test_uses_synchronous_podman_exec(self) -> None:
        compose = mock.Mock(podman=_podman())
        compose.containers = [{"name": "proj_a_1"}, {"name": "proj_b_1"}]
        await lifecycle.compose_wait(compose, Namespace())
        # 上游 os.execlp 路径：exec 为同步调用、不 await，参数无 -a 等前缀
        compose.podman.exec.assert_called_once()
        args = compose.podman.exec.call_args.args
        assert args[0] == []
        assert args[1] == "wait"
        assert args[2] == ["--", "proj_a_1", "proj_b_1"]


class TestLogs(IsolatedAsyncioTestCase):
    async def test_assembles_args_and_creates_per_service_tasks(self) -> None:
        compose = mock.Mock()
        compose.container_names_by_service = {"a": ["proj_a_1"], "b": ["proj_b_1"]}
        compose.assert_services = mock.Mock()

        created: list[tuple[str, list, int]] = []

        def fake_create_task(_compose: object, _args: object, service: str,
                             podman_args: list, max_len: int) -> asyncio.Future:
            created.append((service, list(podman_args), max_len))
            done: asyncio.Future = asyncio.Future()
            done.set_result(0)
            return done

        with mock.patch(
            "xuan_compose.commands.logs.create_format_logs_task",
            side_effect=fake_create_task,
        ):
            await compose_logs(
                compose,
                Namespace(
                    services=["a", "b"],
                    latest=False,
                    follow=True,
                    names=True,
                    no_color=False,
                    since="1h",
                    tail="10",
                    timestamps=True,
                    until="5m",
                ),
            )
        services = {item[0] for item in created}
        assert services == {"a", "b"}
        podman_args = created[0][1]
        assert podman_args == ["-f", "-n", "--color", "--since", "1h", "--tail", "10", "-t",
                               "--until", "5m"]
        assert created[0][2] == 1

    async def test_tail_all_omits_tail_flag(self) -> None:
        compose = mock.Mock()
        compose.container_names_by_service = {"svc": ["proj_svc_1"]}
        compose.assert_services = mock.Mock()
        captured: list = []

        def fake_create_task(_c: object, _a: object, _s: str, podman_args: list,
                             _m: int) -> asyncio.Future:
            captured.extend(podman_args)
            done: asyncio.Future = asyncio.Future()
            done.set_result(0)
            return done

        with mock.patch(
            "xuan_compose.commands.logs.create_format_logs_task",
            side_effect=fake_create_task,
        ):
            await compose_logs(
                compose,
                Namespace(services=["svc"], latest=False, follow=False, names=False,
                          no_color=True, since=None, tail="all", timestamps=False, until=None),
            )
        assert "--tail" not in captured
        assert "--color" not in captured


class TestLs(IsolatedAsyncioTestCase):
    def _inspect_bytes(self, status: str, running: bool, workdir: str, cfg: str) -> bytes:
        return f"{status}\n{'true' if running else 'false'}\n{workdir}\n{cfg}\n".encode()

    async def test_table_row_contains_project_status(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(
            return_value=self._inspect_bytes("running", True, "/work", "compose.yml")
        )
        compose = mock.Mock(podman=podman)
        compose.containers = [{"name": "proj_a_1", "image": "ghcr.io/a:latest"}]
        buf = io.StringIO()
        with redirect_stdout(buf):
            await list_running_projects(compose, Namespace(format="table"))
        out = buf.getvalue()
        assert "NAME" in out
        assert "proj_a_1" in out
        assert "running(1)" in out

    async def test_json_format_emits_dict_rows(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(
            return_value=self._inspect_bytes("exited", False, "/work", "compose.yml")
        )
        compose = mock.Mock(podman=podman)
        compose.containers = [{"name": "proj_a_1", "image": "ghcr.io/a:latest"}]
        buf = io.StringIO()
        with redirect_stdout(buf):
            await list_running_projects(compose, Namespace(format="json"))
        # 上游 print(data)（Python repr，非标准 JSON），逐行保留该怪癖
        printed = buf.getvalue().strip()
        assert "'Name': 'proj_a_1'" in printed
        assert "'Status': 'exited(0)'" in printed
        assert printed.startswith("[{") and printed.endswith("}]")

    async def test_inspect_failure_breaks_loop(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(side_effect=RuntimeError("gone"))
        compose = mock.Mock(podman=podman)
        compose.containers = [
            {"name": "proj_a_1", "image": "ghcr.io/a:latest"},
            {"name": "proj_b_1", "image": "ghcr.io/b:latest"},
        ]
        # table 无数据行时上游 zip(*data) 仅剩表头，仍可格式化（表头一行）
        buf = io.StringIO()
        with redirect_stdout(buf):
            await list_running_projects(compose, Namespace(format="table"))
        # 异常即 break：第二个容器绝不被 inspect
        assert podman.output.await_count == 1


class TestSystemd(IsolatedAsyncioTestCase):
    async def test_list_prints_registered_projects(self) -> None:
        compose = mock.Mock(project_name="proj")
        with mock.patch(
            "xuan_compose.commands.systemd.glob.glob",
            return_value=["/home/u/.config/containers/compose/projects/alpha.env",
                          "/home/u/.config/containers/compose/projects/beta.env"],
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                await compose_systemd(compose, Namespace(action="ls"))
        assert buf.getvalue().split() == ["alpha", "beta"]

    async def test_create_unit_uses_injected_executable(self) -> None:
        compose = mock.Mock(project_name="proj", executable="/opt/bin/podman-compose")
        # 非 root：os.access 为 False，走 print(out) 分支，不落盘 /etc
        with mock.patch("xuan_compose.commands.systemd.os.access", return_value=False):
            buf = io.StringIO()
            with redirect_stdout(buf):
                await compose_systemd(compose, Namespace(action="create-unit"))
        unit = buf.getvalue()
        assert "/opt/bin/podman-compose" in unit
        assert "ExecStartPre=-/opt/bin/podman-compose up --no-start" in unit
