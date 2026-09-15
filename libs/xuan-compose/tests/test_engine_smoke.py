# SPDX-License-Identifier: GPL-2.0-only
"""引擎层冒烟测试（TR-6.1/TR-6.4）。

验证分层重构的核心承诺：
- ``ComposeEngine()`` 可独立实例化，无需 argv/全局单例；
- 实例化过程不创建 runner、不发起子进程；
- 两个引擎实例的可变状态互不污染；
- 构造期/导入期不存在命令装饰器副作用（commands 为空表，T7 才显式注册）。
"""

from unittest import mock

from xuan_compose.engine import ComposeEngine


def test_instantiation_creates_no_runner_or_subprocess() -> None:
    # 构造引擎若胆敢拉起子进程即视为分层失败
    with mock.patch("subprocess.Popen", side_effect=AssertionError("no subprocess allowed")):
        engine = ComposeEngine()
    assert engine.podman is None
    assert engine.podman_version is None
    # 命令表不由装饰器在构造期填充
    assert engine.commands == {}


def test_runner_can_be_explicitly_injected() -> None:
    sentinel = object()
    engine = ComposeEngine(podman=sentinel)  # type: ignore[arg-type]
    assert engine.podman is sentinel


def test_two_engines_have_independent_state() -> None:
    engine_a = ComposeEngine()
    engine_b = ComposeEngine()

    engine_a.containers.append({"name": "a"})
    engine_a.all_services.add("svc-a")
    engine_a.networks["net-a"] = None
    engine_a.x_podman["injected"] = True
    engine_a.global_args.custom = 1
    engine_a.project_name = "project-a"

    assert engine_b.containers == []
    assert engine_b.all_services == set()
    assert engine_b.networks == {}
    assert engine_b.x_podman == {}
    assert not hasattr(engine_b.global_args, "custom")
    assert engine_b.project_name is None

    # 反向同样隔离
    engine_b.project_name = "project-b"
    assert engine_a.project_name == "project-a"


def test_engine_module_does_not_read_sys_argv() -> None:
    # TR-8.3 预演：argv 读取只能存在于未来的 cli/ 层；引擎源码不得出现
    import inspect

    from xuan_compose import engine

    source = inspect.getsource(engine)
    assert "sys.argv" not in source
    # 引擎允许 sys.exit 终止致命错误，但不得自行触碰 subprocess
    # （子进程边界唯一入口是 runner 层，TR-4.2）
    assert "import subprocess" not in source
