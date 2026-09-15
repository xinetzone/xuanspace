# SPDX-License-Identifier: GPL-2.0-only
"""T7 新增：命令层编排型 handler 行为测试（下半）。

覆盖 build_one/compose_build 的构建拓扑、create_pods、环境密钥创建、
check_dep_conditions 版本门控、down 的逆序停止/卷网清理、up 的
detach happy path 与互斥参数校验，以及 run/cp/exec/start/stop/restart。
全部 fake podman + mock compose，零真实子进程；算法分支逐行对齐上游，
不做顺手优化。
"""

import os
from argparse import Namespace
from unittest import IsolatedAsyncioTestCase, mock

from xuan_compose.commands.build import build_one, compose_build
from xuan_compose.commands.lifecycle import (
    compose_restart,
    compose_start,
    compose_stop,
)
from xuan_compose.commands.runexec import compose_cp, compose_exec, compose_run
from xuan_compose.commands.updown import (
    check_dep_conditions,
    compose_down,
    compose_up,
    create_pods,
    create_secrets_from_environment,
)
from xuan_compose.model import ServiceDependency
from xuan_compose.runner import CalledProcessError


def _podman(run_return: int = 0) -> mock.Mock:
    podman = mock.Mock()
    podman.run = mock.AsyncMock(return_value=run_return)
    podman.output = mock.AsyncMock(return_value=b"")
    podman.exec = mock.Mock()
    podman.volume_ls = mock.AsyncMock(return_value=[])
    podman.network_ls = mock.AsyncMock(return_value=[])
    podman.existing_containers = mock.AsyncMock(return_value={})
    return podman


def _down_args(**overrides: object) -> Namespace:
    base: dict[str, object] = dict(
        services=[],
        no_deps=False,
        volumes=False,
        rmi=None,
        remove_orphans=False,
        timeout=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TestBuildOne(IsolatedAsyncioTestCase):
    async def test_container_without_build_is_skipped(self) -> None:
        compose = mock.Mock(podman=_podman())
        result = await build_one(compose, Namespace(if_not_exists=False), {})
        assert result is None
        compose.podman.run.assert_not_called()

    async def test_if_not_exists_skips_present_image(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(return_value=b"sha256:abc\n")
        compose = mock.Mock(podman=podman)
        cnt = {"image": "ghcr.io/a:latest", "build": {"context": "."}}
        result = await build_one(compose, Namespace(if_not_exists=True), cnt)
        assert result is None
        podman.run.assert_not_called()

    async def test_if_not_exists_inspect_error_falls_through_to_build(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(side_effect=CalledProcessError(1, ["podman"]))
        compose = mock.Mock(podman=podman)
        cnt = {"image": "ghcr.io/a:latest", "build": {"context": "."}}
        cleanup_marker = mock.Mock()
        def fake_build_args(_c: object, _cnt: object, _a: object, _exists: object,
                            cleanup_callbacks: list) -> list[str]:
            cleanup_callbacks.append(cleanup_marker)
            return ["ctx"]

        with mock.patch(
            "xuan_compose.commands.build.container_to_build_args",
            side_effect=fake_build_args,
        ):
            result = await build_one(compose, Namespace(if_not_exists=True), cnt)
        assert result == 0
        podman.run.assert_awaited_once_with([], "build", ["ctx"])
        cleanup_marker.assert_called_once()


class TestComposeBuild(IsolatedAsyncioTestCase):
    def _compose(self, services: dict, containers: list[dict]) -> mock.Mock:
        compose = mock.Mock()
        compose.podman = _podman()
        compose.services = services
        compose.containers = containers
        compose.container_names_by_service = {
            cnt["service_name"]: [cnt["name"]] for cnt in containers
        }
        compose.container_by_name = {cnt["name"]: cnt for cnt in containers}
        compose.assert_services = mock.Mock()
        return compose

    async def test_builds_named_service_via_registry_path(self) -> None:
        cnt = {"name": "p_a_1", "service_name": "a", "build": {"context": "."}}
        compose = self._compose({"a": {"build": {"context": "."}}}, [cnt])
        with mock.patch(
            "xuan_compose.commands.build.build_one", mock.AsyncMock(return_value=0)
        ) as b1:
            result = await compose_build(compose, Namespace(services=["a"], if_not_exists=False))
        assert result == 0
        b1.assert_awaited_once()
        assert b1.await_args.args[2] is cnt

    async def test_build_deps_topological_order(self) -> None:
        # a 依赖 additional_contexts 的 b：第一批只能建 b
        services = {
            "a": {"build": {"context": "./a", "build_deps": {"b"}}},
            "b": {"build": {"context": "./b", "build_deps": set()}},
        }
        containers = [
            {"name": "p_a_1", "service_name": "a", "build": {}},
            {"name": "p_b_1", "service_name": "b", "build": {}},
        ]
        compose = self._compose(services, containers)
        built_order: list[str] = []

        async def fake_build_one(_c: object, _a: object, cnt: dict) -> int:
            built_order.append(cnt["service_name"])
            return 0

        with mock.patch(
            "xuan_compose.commands.build.build_one", side_effect=fake_build_one
        ):
            result = await compose_build(
                compose, Namespace(services=[], if_not_exists=False)
            )
        assert result == 0
        assert built_order == ["b", "a"]

    async def test_failed_build_aborts_batch(self) -> None:
        cnt = {"name": "p_a_1", "service_name": "a", "build": {}}
        compose = self._compose({"a": {"build": {"build_deps": set()}}}, [cnt])
        with mock.patch(
            "xuan_compose.commands.build.build_one", mock.AsyncMock(return_value=2)
        ):
            result = await compose_build(compose, Namespace(services=[], if_not_exists=False))
        assert result == 2


class TestCreatePods(IsolatedAsyncioTestCase):
    async def test_existing_pod_skipped_new_pod_created(self) -> None:
        podman = _podman()
        compose = mock.Mock(podman=podman)
        compose.pods = [
            {"name": "pod_existing", "ports": []},
            {"name": "pod_new", "ports": ["8080:80"]},
        ]
        compose.resolve_pod_args = mock.Mock(return_value=["--infra=false"])

        with mock.patch(
            "xuan_compose.commands.updown.pod_exists",
            mock.AsyncMock(side_effect=[True, False]),
        ):
            await create_pods(compose)

        podman.run.assert_awaited_once()
        args = podman.run.await_args.args[2]
        assert args[:2] == ["create", "--name=pod_new"]
        assert "--infra=false" in args
        assert args[-2:] == ["-p", "8080:80"]


class TestCreateSecrets(IsolatedAsyncioTestCase):
    async def test_no_declared_secrets_is_noop(self) -> None:
        compose = mock.Mock(podman=_podman(), declared_secrets={})
        await create_secrets_from_environment(compose)
        compose.podman.run.assert_not_called()

    async def test_missing_environment_variable_raises(self) -> None:
        compose = mock.Mock(
            podman=_podman(),
            declared_secrets={"tok": {"environment": "MY_TOKEN"}},
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                await create_secrets_from_environment(compose)

    async def test_creates_named_secret_from_environment(self) -> None:
        podman = _podman()
        compose = mock.Mock(
            podman=podman,
            project_name="proj",
            declared_secrets={"tok": {"environment": "MY_TOKEN"}},
        )
        with mock.patch.dict(os.environ, {"MY_TOKEN": "secret-value"}):
            await create_secrets_from_environment(compose)
        args = podman.run.await_args.args[2]
        assert args == [
            "create",
            "--label",
            "io.podman.compose.project=proj",
            "--env",
            "proj_tok",
            "MY_TOKEN",
        ]


class TestCheckDepConditions(IsolatedAsyncioTestCase):
    async def test_empty_deps_makes_no_calls(self) -> None:
        compose = mock.Mock(podman=_podman())
        await check_dep_conditions(compose, set())
        compose.podman.output.assert_not_called()

    async def test_healthy_condition_ignored_below_podman_4_6(self) -> None:
        compose = mock.Mock(podman=_podman(), podman_version="4.5.0")
        compose.container_names_by_service = {"a": ["p_a_1"]}
        deps = {ServiceDependency("a", "service_healthy")}
        await check_dep_conditions(compose, deps)
        compose.podman.output.assert_not_called()

    async def test_service_started_waits_per_container(self) -> None:
        podman = _podman()
        podman.output = mock.AsyncMock(return_value=b"")
        compose = mock.Mock(podman=podman, podman_version="5.0.0")
        compose.container_names_by_service = {"a": ["p_a_1"]}
        deps = {ServiceDependency("a", "service_started")}
        await check_dep_conditions(compose, deps)
        args = podman.output.await_args.args[2]
        # 枚举 SERVICE_STARTED 的 value 对齐 podman wait 字面量 "running"
        assert args == ["--condition=running", "p_a_1"]


class TestDown(IsolatedAsyncioTestCase):
    def _compose(self, containers: list[dict]) -> mock.Mock:
        compose = mock.Mock(podman=_podman(), project_name="proj")
        compose.containers = containers
        compose.services = {cnt["_service"] for cnt in containers}
        compose.pods = []
        return compose

    async def test_stops_in_reverse_then_removes(self) -> None:
        compose = self._compose(
            [
                {"_service": "a", "name": "p_a_1"},
                {"_service": "b", "name": "p_b_1"},
            ]
        )
        await compose_down(compose, _down_args())
        stop_targets = [
            call.args[2][-1]
            for call in compose.podman.run.await_args_list
            if call.args[1] == "stop"
        ]
        # 上游逆序停止（含默认 stop_grace_period 的 -t 参数）
        assert stop_targets == ["p_b_1", "p_a_1"]
        stop_full = compose.podman.run.await_args_list[0].args[2]
        assert stop_full[:2] == ["-t", "10"]
        rm_cmds = [c for c in compose.podman.run.await_args_list if c.args[1] == "rm"]
        assert {c.args[2][0] for c in rm_cmds} == {"p_a_1", "p_b_1"}
        # excluded 为空时收尾：pods 空但仍查网络并删除
        compose.podman.network_ls.assert_awaited()

    async def test_rmi_local_removes_volumes_and_networks(self) -> None:
        # 上游语义（L4514）：--rmi local 只删 is_local（localhost/ 前缀或无
        # 仓库前缀的本地构建）镜像，远端拉取镜像保留——逐行保留。
        compose = self._compose(
            [
                {"_service": "a", "name": "p_a_1", "image": "localhost/a:latest"},
                {"_service": "b", "name": "p_b_1", "image": "ghcr.io/b:latest"},
            ]
        )
        compose.podman.volume_ls = mock.AsyncMock(return_value=["proj_vol1"])
        compose.podman.network_ls = mock.AsyncMock(return_value=["proj_net"])
        await compose_down(compose, _down_args(volumes=True, rmi="local"))
        rmi_call = next(
            c for c in compose.podman.run.await_args_list if c.args[1] == "rmi"
        )
        rmi_images = rmi_call.args[2]
        assert "localhost/a:latest" in rmi_images
        assert "ghcr.io/b:latest" not in rmi_images
        vol_call = next(
            c for c in compose.podman.run.await_args_list
            if c.args[1] == "volume" and c.args[2][0] == "rm"
        )
        assert vol_call.args[2] == ["rm", "proj_vol1"]
        net_call = next(
            c for c in compose.podman.run.await_args_list
            if c.args[1] == "network" and c.args[2][0] == "rm"
        )
        assert net_call.args[2] == ["rm", "proj_net"]


def _up_args(**overrides: object) -> Namespace:
    base: dict[str, object] = dict(
        services=[],
        no_deps=False,
        no_attach=[],
        dry_run=False,
        force_recreate=False,
        no_recreate=False,
        no_start=False,
        detach=True,
        wait=False,
        wait_timeout=None,
        no_hosts=False,
    )
    base.update(overrides)
    return Namespace(**base)


class TestUpDetached(IsolatedAsyncioTestCase):
    def _compose(self) -> mock.Mock:
        compose = mock.Mock(podman=_podman(), project_name="proj")
        compose.containers = [
            {"_service": "a", "name": "p_a_1", "_deps": set()},
        ]
        compose.services = {"a": {}}
        compose.pods = []
        compose.declared_secrets = {}
        compose.commands = {"down": mock.AsyncMock()}
        compose.config_hash = mock.Mock(return_value="hash")
        return compose

    async def test_detached_creates_then_starts(self) -> None:
        compose = self._compose()
        with (
            mock.patch(
                "xuan_compose.commands.updown.prepare_images",
                mock.AsyncMock(return_value=0),
            ),
            mock.patch(
                "xuan_compose.commands.updown.container_to_args",
                mock.AsyncMock(return_value=["img", "cmd"]),
            ),
        ):
            result = await compose_up(compose, _up_args())
        assert result == 0
        commands = [(c.args[1], c.args[2]) for c in compose.podman.run.await_args_list]
        assert commands[0] == ("create", ["img", "cmd"])
        assert commands[1] == ("start", ["p_a_1"])

    async def test_unknown_no_attach_service_returns_1(self) -> None:
        compose = self._compose()
        with mock.patch("xuan_compose.commands.updown.prepare_images") as prepare:
            result = await compose_up(compose, _up_args(no_attach=["ghost"]))
        assert result == 1
        prepare.assert_not_called()
        compose.podman.run.assert_not_called()

    async def test_force_and_no_recreate_are_mutually_exclusive(self) -> None:
        compose = self._compose()
        compose.podman.existing_containers = mock.AsyncMock(
            return_value={
                "p_a_1": mock.Mock(
                    service_name="a",
                    exited=False,
                    image_id="img-id",
                    config_hash="hash",
                )
            }
        )
        with (
            mock.patch(
                "xuan_compose.commands.updown.prepare_images",
                mock.AsyncMock(return_value=0),
            ),
            mock.patch(
                "xuan_compose.commands.updown.container_to_args",
                mock.AsyncMock(return_value=[]),
            ),
        ):
            result = await compose_up(
                compose, _up_args(force_recreate=True, no_recreate=True)
            )
        assert result == 1
        compose.commands["down"].assert_not_called()

    async def test_dry_run_returns_after_create_phase(self) -> None:
        compose = self._compose()
        with (
            mock.patch(
                "xuan_compose.commands.updown.prepare_images",
                mock.AsyncMock(return_value=0),
            ),
            mock.patch(
                "xuan_compose.commands.updown.container_to_args",
                mock.AsyncMock(return_value=["img"]),
            ),
        ):
            result = await compose_up(compose, _up_args(dry_run=True))
        assert result is None
        assert all(c.args[1] != "start" for c in compose.podman.run.await_args_list)


class TestRunCpExec(IsolatedAsyncioTestCase):
    async def test_run_detached_without_deps_builds_then_runs(self) -> None:
        podman = _podman(0)
        cnt = {"_deps": set()}
        compose = mock.Mock(podman=podman, pods=[])
        compose.container_names_by_service = {"svc": ["p_svc_1"]}
        compose.container_by_name = {"p_svc_1": cnt}
        compose.assert_services = mock.Mock()
        compose.commands = {
            "up": mock.AsyncMock(),
            "build": mock.AsyncMock(return_value=0),
        }
        # 注：上游 compose_run_parse 不定义 --build-arg，真实 args 无 build_arg
        # 属性（handler 内 Namespace(build_arg=[], **args.__dict__) 依赖这点，
        # T8 parser 装配注意事项已登记 tasks.md）。
        args = Namespace(
            service="svc",
            no_deps=True,
            detach=True,
            rm=False,
            build=False,
            extra_args=[],
        )
        with (
            mock.patch(
                "xuan_compose.commands.runexec.compose_run_update_container_from_args"
            ) as update,
            mock.patch(
                "xuan_compose.commands.runexec.container_to_args",
                mock.AsyncMock(return_value=["img"]),
            ),
        ):
            with self.assertRaises(SystemExit) as ctx:
                await compose_run(compose, args)
        assert ctx.exception.code == 0
        update.assert_called_once()
        compose.commands["up"].assert_not_called()
        compose.commands["build"].assert_awaited_once()
        run_args = podman.run.await_args.args[2]
        assert run_args == ["img"]  # detach：不插入 -i/--rm

    async def test_run_attached_inserts_interactive_and_rm(self) -> None:
        cnt = {"_deps": set()}
        compose = mock.Mock(podman=_podman(0), pods=[])
        compose.container_names_by_service = {"svc": ["p_svc_1"]}
        compose.container_by_name = {"p_svc_1": cnt}
        compose.assert_services = mock.Mock()
        compose.commands = {"up": mock.AsyncMock(), "build": mock.AsyncMock(return_value=0)}
        args = Namespace(
            service="svc", no_deps=True, detach=False, rm=True, build=False,
            extra_args=[],
        )
        with (
            mock.patch(
                "xuan_compose.commands.runexec.compose_run_update_container_from_args"
            ),
            mock.patch(
                "xuan_compose.commands.runexec.container_to_args",
                mock.AsyncMock(return_value=["img"]),
            ),
        ):
            with self.assertRaises(SystemExit):
                await compose_run(compose, args)
        run_args = compose.podman.run.await_args.args[2]
        # 上游 insert(1, ...) 顺序：先 --rm 再 -i
        assert run_args[1:3] == ["--rm", "-i"]

    async def test_cp_resolves_service_from_source_colon(self) -> None:
        compose = mock.Mock(podman=_podman(7))
        compose.container_names_by_service = {"svc": ["p_svc_1"]}
        compose.assert_services = mock.Mock()
        args = Namespace(src="svc:/in", dst="/out", archive=False)
        with mock.patch(
            "xuan_compose.commands.runexec.compose_cp_args", return_value=["x", "y"]
        ):
            with self.assertRaises(SystemExit) as ctx:
                await compose_cp(compose, args)
        assert ctx.exception.code == 7
        compose.podman.run.assert_awaited_once_with([], "cp", ["x", "y"])

    async def test_cp_without_colon_raises_value_error(self) -> None:
        compose = mock.Mock(podman=_podman())
        args = Namespace(src="plain", dst="plain2", archive=False)
        with self.assertRaises(ValueError):
            await compose_cp(compose, args)

    async def test_exec_uses_index_to_pick_container(self) -> None:
        cnt = {"name": "p_svc_2"}
        compose = mock.Mock(podman=_podman(0))
        compose.container_names_by_service = {"svc": ["p_svc_1", "p_svc_2"]}
        compose.container_by_name = {"p_svc_2": cnt}
        compose.assert_services = mock.Mock()
        args = Namespace(service="svc", index=2, command="sh", tty=True, interactive=True,
                        privileged=False, user=None, detach=False, env=[], workdir=None,
                        extra_args=[])
        with mock.patch(
            "xuan_compose.commands.runexec.compose_exec_args", return_value=["-it", "p_svc_2"]
        ):
            with self.assertRaises(SystemExit):
                await compose_exec(compose, args)
        compose.podman.run.assert_awaited_once_with([], "exec", ["-it", "p_svc_2"])


class TestStartStopRestart(IsolatedAsyncioTestCase):
    def _compose(self) -> mock.Mock:
        compose = mock.Mock(podman=_podman())
        compose.container_names_by_service = {"a": ["p_a_1"], "b": ["p_b_1"]}
        compose.container_by_name = {
            "p_a_1": {},
            "p_b_1": {},
        }
        compose.assert_services = mock.Mock()
        return compose

    async def test_stop_reverses_targets_with_grace_period(self) -> None:
        compose = self._compose()
        await compose_stop(compose, Namespace(services=[], timeout=None))
        stop_args = [
            c.args[2] for c in compose.podman.run.await_args_list if c.args[1] == "stop"
        ]
        assert stop_args[0][-1] == "p_b_1"
        assert stop_args[1][-1] == "p_a_1"
        assert stop_args[0][:2] == ["-t", "10"]

    async def test_explicit_timeout_overrides_grace_period(self) -> None:
        compose = self._compose()
        await compose_stop(compose, Namespace(services=["a"], timeout=3))
        args = compose.podman.run.await_args.args[2]
        assert args == ["-t", "3", "p_a_1"]

    async def test_restart_reverses_targets(self) -> None:
        compose = self._compose()
        await compose_restart(compose, Namespace(services=[], timeout=None))
        actions = [c.args[1] for c in compose.podman.run.await_args_list]
        assert actions == ["restart", "restart"]
        targets = [c.args[2][-1] for c in compose.podman.run.await_args_list]
        assert targets == ["p_b_1", "p_a_1"]

    async def test_start_with_wait_invokes_health_wait(self) -> None:
        compose = self._compose()
        with mock.patch(
            "xuan_compose.commands.lifecycle.wait_for_container_running_healthy",
            mock.AsyncMock(),
        ) as wait_mock:
            await compose_start(compose, Namespace(services=[], wait=True, wait_timeout=None))
        wait_mock.assert_awaited_once()
        starts = [c for c in compose.podman.run.await_args_list if c.args[1] == "start"]
        assert {c.args[2][0] for c in starts} == {"p_a_1", "p_b_1"}
        for c in starts:
            assert "-t" not in c.args[2]  # start 不带停止宽限参数
