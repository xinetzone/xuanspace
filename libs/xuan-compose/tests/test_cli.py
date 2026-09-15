# SPDX-License-Identifier: GPL-2.0-only
"""T8 新增：CLI parser 注册表、argparse 行为与 main 装配测试。

- TR-8.1 的程序化证据：24 命令逐一 ``<cmd> --help`` 均 SystemExit(0)；
- parser 覆盖矩阵与挂载顺序（精确复刻 @cmd_parse append 序）；
- PullPolicyAction、无子命令/--version/COMPOSE_PARALLEL_LIMIT 等分支；
- async_main 装配：dry-run/version/systemd 的 compose 加载豁免、podman
  版本探测成败、handler int 返回码、executable 注入。
全部 fake Podman + 真实 ComposeEngine，零真实 podman 守护。
"""

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from unittest import IsolatedAsyncioTestCase, mock

from xuan_compose.cli import parser as cli_parser
from xuan_compose.cli.main import async_main
from xuan_compose.cli.parser import COMMAND_PARSERS, build_parser, parse_args
from xuan_compose.commands import COMMAND_HANDLERS
from xuan_compose.engine import ComposeEngine
from xuan_compose.runner import CalledProcessError

ALL_COMMANDS = set(COMMAND_HANDLERS)


def _parse(argv: list[str]) -> object:
    return parse_args(ComposeEngine(), argv)


class TestParserRegistry(IsolatedAsyncioTestCase):
    def test_parser_keys_equal_handler_keys(self) -> None:
        assert set(COMMAND_PARSERS) == ALL_COMMANDS
        assert len(COMMAND_PARSERS) == 24

    def test_wait_has_empty_parser_tuple(self) -> None:
        # 上游 wait 命令无任何 @cmd_parse，subparser 仍需创建
        assert COMMAND_PARSERS["wait"] == ()

    def test_all_24_subparsers_present_in_help(self) -> None:
        p = build_parser()
        # subparser 名集合（含上游的 help 伪命令）
        got = set(p._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
        assert got == ALL_COMMANDS | {"help"}

    def test_parser_mount_order_matches_decorator_append_order(self) -> None:
        # up: compose_up_parse → compose_build_up_parse → compose_build_parse
        #     → compose_up_start_parse（L4959 < L5355 < L5387 < L5398）
        fns = COMMAND_PARSERS["up"]
        assert [f.__name__ for f in fns] == [
            "compose_up_parse",
            "compose_build_up_parse",
            "compose_build_parse",
            "compose_up_start_parse",
        ]
        # build: pull 策略组先于 services 位置参
        assert [f.__name__ for f in COMMAND_PARSERS["build"]] == [
            "compose_build_up_parse",
            "compose_build_parse",
        ]
        # down: down 专属 → timeout 组 → services 组
        assert [f.__name__ for f in COMMAND_PARSERS["down"]] == [
            "compose_down_parse",
            "compose_parse_timeout",
            "compose_build_parse",
        ]
        # start/stop/restart
        assert [f.__name__ for f in COMMAND_PARSERS["start"]] == [
            "compose_build_parse",
            "compose_up_start_parse",
        ]
        assert [f.__name__ for f in COMMAND_PARSERS["stop"]] == [
            "compose_parse_timeout",
            "compose_build_parse",
        ]
        # pause/unpause 共享、ps/stats 各有 format
        assert COMMAND_PARSERS["pause"] == COMMAND_PARSERS["unpause"]
        assert [f.__name__ for f in COMMAND_PARSERS["ps"]] == [
            "compose_ps_parse",
            "compose_format_parse",
        ]
        assert [f.__name__ for f in COMMAND_PARSERS["stats"]] == [
            "compose_stats_parse",
            "compose_format_parse",
        ]

    def test_systemd_help_derived_from_handler_docstring(self) -> None:
        from xuan_compose.cli.parser import COMMAND_HELP

        assert COMMAND_HELP["systemd"] == (
            "create systemd unit file and register its compose stacks"
        )


class TestHelpEveryCommand(IsolatedAsyncioTestCase):
    """TR-8.1 程序化证据：24 命令 --help 全部退出码 0。"""

    def test_every_command_help_exits_zero(self) -> None:
        for cmd in sorted(ALL_COMMANDS):
            with self.subTest(cmd=cmd):
                with self.assertRaises(SystemExit) as ctx:
                    _parse([cmd, "--help"])
                assert ctx.exception.code == 0, cmd

    def test_global_help_lists_all_commands(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
            _parse(["--help"])
        # argparse --help 退出码 0
        assert ctx.exception.code == 0
        out = buf.getvalue()
        for cmd in ALL_COMMANDS:
            assert cmd in out


class TestGlobalParser(IsolatedAsyncioTestCase):
    def test_no_command_exits_minus_one(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            _parse([])
        assert ctx.exception.code == -1

    def test_help_pseudo_command_exits_minus_one(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            _parse(["help"])
        assert ctx.exception.code == -1

    def test_version_flag_routes_to_version_command(self) -> None:
        args = _parse(["--version"])
        assert args.command == "version"
        assert args.version is True

    def test_version_subcommand_short(self) -> None:
        args = _parse(["version", "--short"])
        assert args.command == "version"
        assert args.short is True
        assert args.format == "pretty"

    def test_podman_path_default_and_global_flags(self) -> None:
        args = _parse(["ps"])
        assert args.podman_path == "podman"
        assert args.dry_run is False
        assert args.verbose is False
        assert args.no_ansi is False
        assert args.env_file == []
        assert args.podman_args == []
        for podman_cmd in cli_parser.PODMAN_CMDS:
            assert getattr(args, f"podman_{podman_cmd}_args") == []

    def test_parallel_default_is_sys_maxsize(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMPOSE_PARALLEL_LIMIT", None)
            args = _parse(["ps"])
        assert args.parallel == sys.maxsize

    def test_parallel_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"COMPOSE_PARALLEL_LIMIT": "7"}):
            args = _parse(["ps"])
        assert args.parallel == 7

    def test_parse_result_is_written_to_engine_global_args(self) -> None:
        engine = ComposeEngine()
        args = parse_args(engine, ["-p", "proj", "ps", "-q"])
        assert engine.global_args is args
        assert engine.global_args.project_name == "proj"
        assert engine.global_args.quiet is True


class TestPullPolicyAction(IsolatedAsyncioTestCase):
    def test_pull_without_value_means_newer(self) -> None:
        args = _parse(["build", "--pull"])
        assert args.pull == "newer"

    def test_pull_with_explicit_value(self) -> None:
        args = _parse(["build", "--pull", "always"])
        assert args.pull == "always"

    def test_pull_always_without_value_sets_always(self) -> None:
        args = _parse(["build", "--pull-always"])
        assert args.pull == "always"

    def test_pull_always_true_sets_always(self) -> None:
        args = _parse(["build", "--pull-always", "true"])
        assert args.pull == "always"

    def test_pull_always_false_leaves_pull_at_default(self) -> None:
        # 上游行为：values == "false" 时直接 return，action 不赋值；
        # argparse 解析前已按 action.default 预置 pull=None，故保持 None
        args = _parse(["build", "--pull-always", "false"])
        assert args.pull is None

    def test_pull_choice_rejected(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            _parse(["build", "--pull", "bogus"])
        assert ctx.exception.code == 2


class TestSubcommandArgs(IsolatedAsyncioTestCase):
    def test_build_has_build_arg_no_cache_and_services(self) -> None:
        args = _parse(["build", "--build-arg", "A=1", "--no-cache", "svc1", "svc2"])
        assert args.build_arg == ["A=1"]
        assert args.no_cache is True
        assert args.services == ["svc1", "svc2"]

    def test_run_has_no_build_arg_attribute(self) -> None:
        # T7 交接注记：run parser 上游即无 --build-arg（handler 构造 build
        # 子 Namespace 时显式补 build_arg=[]，重复键会 TypeError）
        args = _parse(["run", "svc", "sh", "-c", "true"])
        assert not hasattr(args, "build_arg")
        assert args.service == "svc"
        assert args.cnt_command == ["sh", "-c", "true"]

    def test_exec_remainder_and_index(self) -> None:
        args = _parse(["exec", "--index", "2", "svc", "ls"])
        assert args.index == 2
        assert args.service == "svc"
        assert args.cnt_command == ["ls"]

    def test_cp_src_dst(self) -> None:
        args = _parse(["cp", "svc:/in", "/out"])
        assert args.src == "svc:/in"
        assert args.dst == "/out"
        assert args.archive is True

    def test_kill_all_and_signal(self) -> None:
        args = _parse(["kill", "-s", "HUP", "-a"])
        assert args.signal == "HUP"
        assert args.all is True

    def test_down_rmi_optional_value(self) -> None:
        # nargs="?" + const="all"：裸 --rmi → all；带值 → local
        assert _parse(["down", "--rmi"]).rmi == "all"
        assert _parse(["down", "--rmi", "local"]).rmi == "local"

    def test_logs_tail_and_flags(self) -> None:
        args = _parse(["logs", "-f", "-n", "-t", "--tail", "5", "a", "b"])
        assert args.follow is True
        assert args.names is True
        assert args.timestamps is True
        assert args.tail == "5"
        assert args.services == ["a", "b"]

    def test_port_positional(self) -> None:
        args = _parse(["port", "--protocol", "udp", "svc", "9000"])
        assert args.protocol == "udp"
        assert args.service == "svc"
        assert args.private_port == 9000
        assert args.index == 1

    def test_systemd_action_choices_and_default(self) -> None:
        assert _parse(["systemd"]).action == "register"
        assert _parse(["systemd", "-a", "ls"]).action == "ls"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse(["systemd", "-a", "nope"])

    def test_ps_quiet_and_format_precedence(self) -> None:
        # parser 层两者可共存；quiet 优先是 handler 行为（T7 已测）
        args = _parse(["ps", "-q"])
        assert args.quiet is True

    def test_stats_interval_and_format(self) -> None:
        args = _parse(["stats", "-i", "2", "-f", "json"])
        assert args.interval == 2
        assert args.format == "json"

    def test_ls_format_default_table(self) -> None:
        assert _parse(["ls"]).format == "table"
        assert _parse(["ls", "-f", "json"]).format == "json"

    def test_config_flags(self) -> None:
        args = _parse(["config", "--services", "-q"])
        assert args.services is True
        assert args.quiet is True
        assert args.no_normalize is False


def _fake_podman(version_bytes: bytes | None = b"podman version 5.2.3\n") -> mock.Mock:
    podman = mock.Mock()
    podman.run = mock.AsyncMock(return_value=0)
    if version_bytes is not None:
        podman.output = mock.AsyncMock(return_value=version_bytes)
    else:
        podman.output = mock.AsyncMock(side_effect=FileNotFoundError("podman"))
    return podman


class TestAsyncMainAssembly(IsolatedAsyncioTestCase):
    async def test_dry_run_version_short_runs_end_to_end_without_podman(self) -> None:
        # 零 mock 端到端：dry-run 跳过版本探测，--short 不发起任何子进程，
        # version 命令豁免 compose 文件加载
        buf = io.StringIO()
        with redirect_stdout(buf):
            await async_main(["--dry-run", "version", "--short"])
        from xuan_compose import __version__

        assert buf.getvalue().strip() == __version__

    async def test_version_and_systemd_create_unit_skip_compose_load(self) -> None:
        with mock.patch.object(ComposeEngine, "_parse_compose_file") as load:
            await async_main(["--dry-run", "version", "--short"])
            assert load.call_count == 0
            await async_main(["--dry-run", "systemd", "-a", "create-unit"])
            assert load.call_count == 0

    async def test_other_commands_load_compose_file(self) -> None:
        with (
            mock.patch.object(ComposeEngine, "_parse_compose_file") as load,
            mock.patch("xuan_compose.cli.main.Podman") as podman_cls,
        ):
            podman_cls.return_value = _fake_podman()
            # ps handler 只读 project_name 并发一次 podman run
            await async_main(["ps"])
        load.assert_called_once_with()

    async def test_podman_version_probed_when_not_dry_run(self) -> None:
        captured: dict = {}

        def capture(engine: object, *a: object, **kw: object) -> mock.Mock:
            captured["engine"] = engine
            return _fake_podman(b"podman version 5.2.3\n")

        with (
            mock.patch.object(ComposeEngine, "_parse_compose_file"),
            mock.patch("xuan_compose.cli.main.Podman", side_effect=capture),
        ):
            await async_main(["version"])
        assert captured["engine"].podman_version == "5.2.3"

    async def test_missing_podman_is_fatal_exit_1(self) -> None:
        with (
            mock.patch("xuan_compose.cli.main.Podman") as podman_cls,
        ):
            podman_cls.return_value = _fake_podman(version_bytes=None)
            with self.assertRaises(SystemExit) as ctx:
                await async_main(["version"])
        assert ctx.exception.code == 1

    async def test_integer_handler_return_code_becomes_exit(self) -> None:
        # pull handler 在空容器集时返回 0 → sys.exit(0)
        with (
            mock.patch.object(ComposeEngine, "_parse_compose_file"),
            mock.patch("xuan_compose.cli.main.Podman") as podman_cls,
        ):
            podman_cls.return_value = _fake_podman()
            with self.assertRaises(SystemExit) as ctx:
                await async_main(["pull"])
        assert ctx.exception.code == 0

    async def test_executable_injected_for_systemd_template(self) -> None:
        captured: dict = {}

        def capture(engine: object, *a: object, **kw: object) -> mock.Mock:
            captured["engine"] = engine
            pod = mock.Mock()
            pod.run = mock.AsyncMock(return_value=0)
            pod.output = mock.AsyncMock(return_value=b"podman version 5.2.3\n")
            return pod

        with (
            mock.patch.object(ComposeEngine, "_parse_compose_file"),
            mock.patch("xuan_compose.cli.main.Podman", side_effect=capture),
        ):
            await async_main(["ps"])
        assert captured["engine"].executable == os.path.realpath(sys.argv[0])

    async def test_custom_podman_path_missing_is_fatal(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            await async_main(["--podman-path", "/no/such/podman-bin", "version"])
        assert ctx.exception.code == 1

    async def test_missing_custom_podman_path_allowed_in_dry_run(self) -> None:
        # dry-run 下二进制不存在只警告不退出（上游 44-46 行）
        buf = io.StringIO()
        with redirect_stdout(buf):
            await async_main(
                ["--dry-run", "--podman-path", "/no/such/podman-bin", "version", "--short"]
            )
        from xuan_compose import __version__

        assert buf.getvalue().strip() == __version__

    async def test_existing_custom_podman_path_realpathed(self) -> None:
        # 用当前解释器作为"存在且可执行"的替身，两平台通用
        interpreter = sys.executable
        captured: dict = {}

        class _Cap:
            def __init__(self, engine: object, path: object, *a: object, **kw: object) -> None:
                captured["path"] = path

            run = mock.AsyncMock(return_value=0)
            output = mock.AsyncMock(return_value=b"podman version 5.2.3\n")

        with (
            mock.patch.object(ComposeEngine, "_parse_compose_file"),
            mock.patch("xuan_compose.cli.main.Podman", side_effect=_Cap),
        ):
            await async_main(["--podman-path", interpreter, "ps"])
        assert captured["path"] == os.path.realpath(interpreter)

    async def test_version_probe_called_process_error_is_fatal(self) -> None:
        # CalledProcessError 且带 output：错误信息拼接 output 后仍 fatal
        # （版本探测失败的终局与 FileNotFoundError 一致，都是 exit 1）
        pod = mock.Mock()
        pod.run = mock.AsyncMock(return_value=0)
        pod.output = mock.AsyncMock(
            side_effect=CalledProcessError(125, ["podman"], output=b"podman: boom\n")
        )
        with mock.patch("xuan_compose.cli.main.Podman", return_value=pod):
            with self.assertRaises(SystemExit) as ctx:
                await async_main(["version"])
        assert ctx.exception.code == 1

    async def test_handlers_installed_for_inter_command_lookup(self) -> None:
        # install_handlers 必须在分发前装配：up/run 经 compose.commands 互调
        engine_holder: dict = {}

        class _CapturingPodman:
            def __init__(self, engine: object, *a: object, **kw: object) -> None:
                engine_holder["engine"] = engine
                self.run = mock.AsyncMock(return_value=0)
                self.output = mock.AsyncMock(return_value=b"podman version 5.2.3\n")

        with (
            mock.patch.object(ComposeEngine, "_parse_compose_file"),
            mock.patch("xuan_compose.cli.main.Podman", side_effect=_CapturingPodman),
        ):
            await async_main(["ps"])
        assert set(engine_holder["engine"].commands) == ALL_COMMANDS
        assert callable(engine_holder["engine"].commands["up"])
