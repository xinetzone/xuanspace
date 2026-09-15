# SPDX-License-Identifier: GPL-2.0-only
"""translate.run_args 分支补测（移植的 6+3+2 例覆盖主路径；这里补
get_excluded 的 dep_field/no_deps/未知服务分支、get_volume_names、
run 更新函数的全选项分支、exec 全选项——上游第 3795-3809、
4594-4705 行真实分支）。"""

import argparse
from unittest import mock

from xuan_compose.translate.run_args import (
    compose_exec_args,
    compose_run_update_container_from_args,
    deps_from_container,
    get_excluded,
    get_service_info,
    get_volume_names,
)
from xuan_compose.types import DependField


def _ns(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(**kw)


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


class TestGetExcluded:
    def test_empty_services_returns_empty(self) -> None:
        compose = mock.Mock(services={"a": {}, "b": {}})
        assert get_excluded(compose, _ns(services=[])) == set()

    def test_excludes_targets_but_keeps_their_dependencies(self) -> None:
        compose = mock.Mock(
            services={
                "a": {DependField.DEPENDENCIES: [_Named("b")]},
                "b": {},
                "c": {},
            }
        )
        # 目标 a 被排除；其依赖 b 必须保留；c 与目标无关被排除
        result = get_excluded(compose, _ns(services=["a"], no_deps=False))
        assert result == {"c"}

    def test_no_deps_excludes_entire_tree_except_targets(self) -> None:
        compose = mock.Mock(
            services={"a": {DependField.DEPENDENCIES: [_Named("b")]}, "b": {}, "c": {}}
        )
        result = get_excluded(compose, _ns(services=["a"], no_deps=True))
        assert result == {"b", "c"}

    def test_missing_no_deps_attr_defaults_to_false(self) -> None:
        # compose_down_parse 不配置 no_deps：getattr 默认 False
        compose = mock.Mock(services={"a": {DependField.DEPENDENCIES: [_Named("b")]}, "b": {}})
        result = get_excluded(compose, _ns(services=["a"]))
        assert result == set()

    def test_unknown_target_service_just_discarded(self) -> None:
        compose = mock.Mock(services={"a": {}})
        result = get_excluded(compose, _ns(services=["ghost"], no_deps=False))
        assert result == {"a"}


def test_deps_from_container() -> None:
    cnt = {"_deps": {"a", "b"}}
    assert deps_from_container(_ns(no_deps=True), cnt) == set()
    assert deps_from_container(_ns(no_deps=False), cnt) == {"a", "b"}


def test_get_service_info() -> None:
    compose = mock.Mock(containers=[{"_service": "a"}, {"_service": "b", "x": 1}])
    found = get_service_info(compose, "b")
    assert found is not None
    cnt, index = found
    assert index == 1 and cnt["x"] == 1
    assert get_service_info(compose, "ghost") is None


class TestGetVolumeNames:
    @staticmethod
    def _compose():
        c = mock.Mock()
        c.dirname = "/base"
        c.project_name = "proj"
        c.vols = {}
        c.format_name.side_effect = lambda *parts: "proj_" + "_".join(parts)[:12]
        return c

    def test_only_named_volumes_listed(self) -> None:
        compose = self._compose()
        cnt = {
            "_service": "web",
            "volumes": [
                "/host:/data",  # bind → 跳过
                "data:/var/lib",  # 命名卷 → proj_data
                {"type": "tmpfs", "target": "/t"},  # tmpfs → 跳过
                {"type": "volume", "source": None, "target": "/anon"},  # 匿名卷
            ],
        }
        names = get_volume_names(compose, cnt)
        assert names[0] == "proj_data"
        # 匿名卷生成名以 proj_web_ 开头
        assert names[1].startswith("proj_web_")
        assert len(names) == 2

    def test_no_volumes(self) -> None:
        assert get_volume_names(self._compose(), {"_service": "web"}) == []


class TestComposeRunUpdate:
    def test_all_optional_adjustments(self) -> None:
        compose = mock.Mock()
        compose.format_name.return_value = "proj_web_tmp123"
        cnt = {
            "environment": {"A": "1"},
            "expose": ["80"],
            "ports": ["80:80"],
            "restart": "always",
        }
        args = _ns(
            service="web",
            name=None,
            entrypoint=["/bin/sh"],
            user="1000",
            workdir="/app",
            env=["B=2", "C=3"],
            service_ports=False,
            publish=["9090:9090"],
            volume=["extra:/e"],
            T=False,
            cnt_command=["echo", "hi"],
            rm=True,
        )
        compose_run_update_container_from_args(compose, cnt, args)
        assert cnt["name"] == "proj_web_tmp123"
        assert cnt["entrypoint"] == ["/bin/sh"]
        assert cnt["user"] == "1000"
        assert cnt["working_dir"] == "/app"
        assert cnt["environment"] == {"A": "1", "B": "2", "C": "3"}
        assert "expose" not in cnt
        assert "publishall" not in cnt
        # 原 ports 已被 service_ports=False 分支删除，--publish 重建为仅新值
        assert cnt["ports"] == ["9090:9090"]
        assert cnt["volumes"] == ["extra:/e"]
        assert cnt["tty"] is True
        assert cnt["command"] == ["echo", "hi"]
        assert "restart" not in cnt  # --rm 与 restart 互斥

    def test_explicit_name_and_service_ports_kept(self) -> None:
        compose = mock.Mock()
        cnt = {"expose": ["80"], "ports": ["80:80"], "publishall": True}
        args = _ns(
            service="web",
            name="oneoff",
            entrypoint=None,
            user=None,
            workdir=None,
            env=None,
            service_ports=True,
            publish=None,
            volume=None,
            T=True,
            cnt_command=None,
            rm=False,
        )
        compose_run_update_container_from_args(compose, cnt, args)
        assert cnt["name"] == "oneoff"
        assert cnt["expose"] == ["80"]
        assert cnt["ports"] == ["80:80"]
        assert cnt["tty"] is False
        assert "command" not in cnt

    def test_container_name_fallback(self) -> None:
        compose = mock.Mock()
        cnt = {"container_name": "fixed-name"}
        args = _ns(
            service="web",
            name=None,
            entrypoint=None,
            user=None,
            workdir=None,
            env=None,
            service_ports=True,
            publish=None,
            volume=None,
            T=True,
            cnt_command=[],
            rm=False,
        )
        compose_run_update_container_from_args(compose, cnt, args)
        assert cnt["name"] == "fixed-name"


class TestComposeExecArgs:
    def test_full_option_set_with_bare_env_name(self) -> None:
        cnt = {"environment": {"A": "1"}}
        args = _ns(
            privileged=True,
            user="1000",
            workdir="/app",
            T=False,
            env=["B=2", "C"],  # C 无等号 → 裸名（值来自宿主环境语义）
            cnt_command=["sh"],
        )
        out = compose_exec_args(cnt, "web_1", args)
        assert out[:3] == ["--interactive", "--privileged", "--user"]
        assert "--workdir" in out and "--tty" in out
        # 环境项：A=1、B=2、C 裸名（无等号形式）
        env_pairs = {out[i + 1] for i, x in enumerate(out) if x == "--env"}
        assert env_pairs == {"A=1", "B=2", "C"}
        assert out[-2:] == ["web_1", "sh"]

    def test_minimal_non_tty(self) -> None:
        args = _ns(privileged=False, user=None, workdir=None, T=True, env=None, cnt_command=None)
        out = compose_exec_args({}, "web_1", args)
        assert out == ["--interactive", "web_1"]
