# SPDX-License-Identifier: GPL-2.0-only
"""dependencies 分支补测（test_depends_on 已覆盖 rec_deps 主路径与
condition 判定；这里补自依赖/环剪枝、links 别名、extends 短路以及
_validate_completed_successfully 的轮询/消失/非零退出分支——
上游第 724-801、4109-4158 行真实分支）。"""

import asyncio
from unittest import mock

import pytest

from xuan_compose.dependencies import (
    _validate_completed_successfully,
    flat_deps,
    rec_deps,
)
from xuan_compose.model import ServiceDependency
from xuan_compose.runner import CalledProcessError


def _sd(name: str, condition: str = "service_started") -> ServiceDependency:
    return ServiceDependency(name, condition)


class TestRecDepsCycles:
    def test_self_dependency_not_recursively_expanded(self) -> None:
        # 上游行为：自依赖边保留在 _deps 集合中，仅跳过递归展开（不新增）
        services = {"a": {"_deps": {_sd("a"), _sd("b")}, "b": {"_deps": set()}}}
        result = rec_deps(services, "a")
        assert result == {_sd("a"), _sd("b")}

    def test_direct_cycle_pruned(self) -> None:
        # a -> b -> a：展开 b 时发现 b 依赖起点 a，剪枝，无无限递归
        services = {
            "a": {"_deps": {_sd("b")}},
            "b": {"_deps": {_sd("a")}},
        }
        assert rec_deps(services, "a") == {_sd("b")}

    def test_unknown_dependency_edge_kept_but_not_expanded(self) -> None:
        services = {"a": {"_deps": {_sd("ghost")}}}
        assert rec_deps(services, "a") == {_sd("ghost")}


class TestFlatDepsExtras:
    def test_links_string_form_and_alias(self) -> None:
        services = {
            "web": {"links": "db:dbalias"},
            "db": {},
        }
        flat_deps(services)
        assert _sd("db") in services["web"]["_deps"]
        assert services["db"]["_aliases"] == {"dbalias"}

    def test_links_list_with_alias(self) -> None:
        services = {
            "web": {"links": ["db:dbalias", "cache"]},
            "db": {},
            "cache": {},
        }
        flat_deps(services)
        names = {d.name for d in services["web"]["_deps"]}
        assert names == {"db", "cache"}
        assert services["db"]["_aliases"] == {"dbalias"}
        assert "_aliases" not in services["cache"]

    def test_extends_short_circuits_depends_on(self) -> None:
        services = {
            "app": {"extends": {"service": "base"}, "depends_on": {"db": {"condition": "service_started"}}},
            "base": {},
            "db": {},
        }
        flat_deps(services, with_extends=True)
        # extends 命中后 continue：depends_on 不再处理，仅收 base
        assert services["app"]["_deps"] == {_sd("base")}

    def test_extends_self_name_skipped(self) -> None:
        services = {"app": {"extends": {"service": "app"}}}
        flat_deps(services, with_extends=True)
        assert services["app"]["_deps"] == set()


def _compose(outputs: list[object]):
    compose = mock.Mock()
    compose.podman.output = mock.AsyncMock(side_effect=outputs)
    return compose


async def _instant_sleep(*_a: object, **_kw: object) -> None:
    return None


class TestValidateCompletedSuccessfully:
    _validate = staticmethod(_validate_completed_successfully)

    def test_happy_path_running_then_exit_zero(self) -> None:
        compose = _compose(
            [
                b"running\n",  # inspect --format status
                b"",  # wait --condition=stopped
                b'[{"State": {"ExitCode": 0}}]',  # per-container inspect
            ]
        )
        asyncio.run(self._validate(compose, ["web_1"]))
        assert compose.podman.output.await_count == 3

    def test_polls_through_created_state(self) -> None:
        compose = _compose(
            [
                b"created\n",  # 第一次仍在 created → 继续轮询
                b"running\n",
                b"",
                b'[{"State": {"ExitCode": 0}}]',
            ]
        )
        with mock.patch("xuan_compose.dependencies.asyncio.sleep", _instant_sleep):
            asyncio.run(self._validate(compose, ["web_1"]))

    def test_inspect_failure_during_polling_is_retried(self) -> None:
        compose = _compose(
            [
                CalledProcessError(1, []),  # 轮询期 inspect 暂时失败 → debug 后重试
                b"running\n",
                b"",
                b'[{"State": {"ExitCode": 0}}]',
            ]
        )
        with mock.patch("xuan_compose.dependencies.asyncio.sleep", _instant_sleep):
            asyncio.run(self._validate(compose, ["web_1"]))

    def test_container_disappeared_after_wait_raises(self) -> None:
        compose = _compose(
            [
                b"running\n",
                b"",
                CalledProcessError(1, []),  # 终态 inspect 失败 → 消失
            ]
        )
        with pytest.raises(RuntimeError, match="disappeared after waiting"):
            asyncio.run(self._validate(compose, ["web_1"]))

    def test_nonzero_exit_code_raises(self) -> None:
        compose = _compose(
            [
                b"running\n",
                b"",
                b'[{"State": {"ExitCode": 2}}]',
            ]
        )
        with pytest.raises(RuntimeError, match="didn't complete successfully: exit code 2"):
            asyncio.run(self._validate(compose, ["web_1"]))
