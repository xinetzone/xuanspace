# SPDX-License-Identifier: GPL-2.0-only
"""engine 纯函数方法与解析辅助的分支补测（不触达 _parse_compose_file
全流程；该流程的多文件/include/profile 主路径已由
test_parse_compose_file_build/test_include/test_can_merge_build 覆盖）。

分支来源：上游 PodmanCompose 第 2419-2523、2598-2650、2915-2970、
3120-3147 行同名方法，行为断言逐条对照。
"""

import argparse

import pytest

from xuan_compose.engine import ComposeEngine
from xuan_compose.merge import OverrideTag, ResetTag
from xuan_compose.model import XPodmanSettingKey


def _engine(**global_kwargs: object) -> ComposeEngine:
    eng = ComposeEngine()
    defaults = {
        "podman_args": [],
        "podman_run_args": [],
        "podman_create_args": [],
        "podman_pull_args": [],
        "in_pod": None,
        "pod_args": None,
        "profile": [],
    }
    defaults.update(global_kwargs)
    eng.global_args = argparse.Namespace(**defaults)
    return eng


class TestAssertServices:
    def test_string_wrapped_to_set(self) -> None:
        eng = _engine()
        eng.all_services = {"db"}
        # 单个字符串服务名存在：不退出
        eng.assert_services("db")  # type: ignore[arg-type]

    def test_missing_service_exits_1(self) -> None:
        eng = _engine()
        eng.all_services = {"db"}
        with pytest.raises(SystemExit) as ctx:
            eng.assert_services(["db", "ghost"])
        assert ctx.value.code == 1

    def test_none_or_empty_is_noop(self) -> None:
        eng = _engine()
        eng.all_services = set()
        eng.assert_services(None)  # type: ignore[arg-type]
        eng.assert_services([])


class TestGetPodmanArgs:
    def test_global_and_command_specific_concatenated(self) -> None:
        eng = _engine(podman_args=["--root /x", "--log-level=debug"], podman_pull_args=["--tls-verify"])
        assert eng.get_podman_args("pull") == [
            "--root",
            "/x",
            "--log-level=debug",
            "pull",
            "--tls-verify",
        ]

    def test_create_normalized_to_run_args(self) -> None:
        eng = _engine(podman_run_args=["--rm"])
        assert eng.get_podman_args("create") == ["create", "--rm"]

    def test_empty(self) -> None:
        eng = _engine()
        assert eng.get_podman_args("ps") == ["ps"]


class TestConfigHashAndOriginalConfiguration:
    def test_cached_hash_returned_verbatim(self) -> None:
        eng = _engine()
        svc = {"_config_hash": "fixed"}
        assert eng.config_hash(svc) == "fixed"

    def test_hash_stable_and_cached(self) -> None:
        eng = _engine()
        svc = {"image": "img:1", "labels": {"a": "1"}}
        h1 = eng.config_hash(svc)
        h2 = eng.config_hash(svc)
        assert h1 == h2 == svc["_config_hash"]
        assert len(h1) == 64

    def test_original_configuration_filters_and_recurses(self) -> None:
        eng = _engine()
        tag_reset = ResetTag()
        tag_over = OverrideTag.__new__(OverrideTag)
        tag_over.value = {}
        configuration = {
            "image": "img",
            "_deps": ["x"],
            42: "non-string-key",
            "drop_reset": tag_reset,
            "drop_over": tag_over,
            "nested": {"keep": 1, "_inner": 2, "again": {"k": "v"}},
        }
        out = eng.original_configuration(configuration)
        assert set(out) == {"image", "nested"}
        assert out["nested"] == {"keep": 1, "again": {"k": "v"}}


class TestPodResolution:
    @pytest.mark.parametrize(
        ("in_pod", "expected"),
        [("true", "pod_proj"), ("false", None)],
    )
    def test_resolve_pod_name_cli_string_forms(self, in_pod, expected) -> None:
        # --in-pod 是字符串选项（非 store_true）："true"/"false" 经
        # try_parse_bool 解析；CLI 缺省（None）时回落 x_podman 设置
        eng = _engine(in_pod=in_pod)
        eng.project_name = "proj"
        assert eng.resolve_pod_name() == expected

    def test_resolve_pod_name_custom_string(self) -> None:
        eng = _engine(in_pod="custom-pod")
        assert eng.resolve_pod_name() == "custom-pod"

    def test_resolve_pod_name_from_x_podman_setting(self) -> None:
        eng = _engine()
        eng.project_name = "proj"
        eng.x_podman = {XPodmanSettingKey.IN_POD: True}
        assert eng.resolve_pod_name() == "pod_proj"

    def test_cli_pod_args_take_priority(self) -> None:
        eng = _engine(pod_args="--infra=true --share=net")
        assert eng.resolve_pod_args() == ["--infra=true", "--share=net"]

    def test_default_pod_args_from_x_podman(self) -> None:
        eng = _engine()
        eng.x_podman = {XPodmanSettingKey.POD_ARGS: ["--infra=true"]}
        assert eng.resolve_pod_args() == ["--infra=true"]

    def test_default_pod_args_builtin(self) -> None:
        eng = _engine()
        assert eng.resolve_pod_args() == ["--infra=false", "--share="]


class TestJoinNameParts:
    def test_underscore_default(self) -> None:
        eng = _engine()
        eng.project_name = "proj"
        assert eng.format_name("svc", "1") == "proj_svc_1"

    def test_dash_when_compat_enabled(self) -> None:
        eng = _engine()
        eng.project_name = "proj"
        eng.x_podman = {XPodmanSettingKey.NAME_SEPARATOR_COMPAT: True}
        assert eng.format_name("svc", "1") == "proj-svc-1"


class TestParseXPodmanSettings:
    def test_unknown_compose_key_warns(self, caplog) -> None:
        eng = _engine()
        with caplog.at_level("WARNING"):
            eng._parse_x_podman_settings({"x-podman": {"bogus_key": 1}}, {})
        assert "unknown x-podman key" in caplog.text
        assert eng.x_podman == {}

    def test_known_compose_key_stored(self) -> None:
        eng = _engine()
        eng._parse_x_podman_settings({"x-podman": {"in_pod": "mypod"}}, {})
        assert eng.x_podman[XPodmanSettingKey.IN_POD] == "mypod"

    def test_environment_known_key_overrides_and_unknown_warns(self, caplog) -> None:
        eng = _engine()
        env = {
            "PODMAN_COMPOSE_IN_POD": "envpod",
            "PODMAN_COMPOSE_BOGUS": "1",
            "PODMAN_COMPOSE_PROVIDER": "ignored",
        }
        with caplog.at_level("WARNING"):
            eng._parse_x_podman_settings({}, env)
        assert eng.x_podman[XPodmanSettingKey.IN_POD] == "envpod"
        assert "unknown PODMAN_COMPOSE_" in caplog.text
        assert not any("provider" in str(k) for k in eng.x_podman)

    def test_docker_compose_compat_fills_defaults(self) -> None:
        eng = _engine()
        eng._parse_x_podman_settings(
            {"x-podman": {"docker_compose_compat": True}}, {}
        )
        assert eng.x_podman[XPodmanSettingKey.DEFAULT_NET_BEHAVIOR_COMPAT] is True
        assert eng.x_podman[XPodmanSettingKey.NAME_SEPARATOR_COMPAT] is True
        assert eng.x_podman[XPodmanSettingKey.IN_POD] is False

    def test_docker_compose_compat_does_not_override_explicit(self) -> None:
        eng = _engine()
        eng._parse_x_podman_settings(
            {"x-podman": {"docker_compose_compat": True, "in_pod": True}}, {}
        )
        assert eng.x_podman[XPodmanSettingKey.IN_POD] is True


class TestResolveProfiles:
    def test_no_profiles_means_always_included(self) -> None:
        eng = _engine()
        out = eng._resolve_profiles({"a": {}, "b": {"profiles": ["debug"]}}, [], set())
        assert set(out) == {"a"}

    def test_requested_profile_matches(self) -> None:
        eng = _engine()
        out = eng._resolve_profiles(
            {"a": {}, "b": {"profiles": ["debug"]}}, [], {"debug"}
        )
        assert set(out) == {"a", "b"}

    def test_target_service_pulls_in_its_profiles(self) -> None:
        eng = _engine()
        out = eng._resolve_profiles(
            {"a": {"profiles": ["debug"]}}, ["a"], set()
        )
        assert set(out) == {"a"}

    def test_requested_profiles_defaults_to_empty_set(self) -> None:
        eng = _engine()
        out = eng._resolve_profiles({"a": {"profiles": ["x"]}}, [])
        assert out == {}


class TestResolveContextDependencies:
    def test_non_kv_and_non_service_entries_kept_as_is(self) -> None:
        eng = _engine()
        services = {
            "a": {
                "image": "img-a",
                "build": {"additional_contexts": ["plain", "other=value:not-service"]},
            }
        }
        eng._resolve_context_dependencies(services)
        assert services["a"]["build"]["additional_contexts"] == [
            "plain",
            "other=value:not-service",
        ]
        assert services["a"]["build"]["build_deps"] == []

    def test_service_reference_resolved_to_image_url(self) -> None:
        eng = _engine()
        eng.project_name = "proj"
        services = {
            "a": {
                "build": {"additional_contexts": ["shared=service:b"]},
            },
            "b": {"image": "registry.example.com/b:1"},
        }
        eng._resolve_context_dependencies(services)
        ctx = services["a"]["build"]["additional_contexts"]
        # urllib quote 会百分号编码 tag 冒号（b:1 → b%3A1，上游原样）
        assert ctx == ["shared=docker://registry.example.com/b%3A1"]
        assert services["a"]["build"]["build_deps"] == ["b"]

    def test_missing_image_uses_project_localhost_name(self) -> None:
        eng = _engine()
        eng.project_name = "proj"
        services = {
            "a": {"build": {"additional_contexts": ["s=service:b"]}},
            "b": {"build": {"context": "./b"}},
        }
        eng._resolve_context_dependencies(services)
        assert services["a"]["build"]["additional_contexts"] == [
            "s=docker://localhost/proj_b"
        ]

    def test_explicit_none_target_raises(self) -> None:
        eng = _engine()
        services = {
            "a": {"build": {"additional_contexts": ["s=service:b"]}},
            "b": None,
        }
        with pytest.raises(ValueError, match="references non-existent service"):
            eng._resolve_context_dependencies(services)

    def test_circular_dependency_detected(self) -> None:
        eng = _engine()
        services = {
            "a": {"image": "a", "build": {"additional_contexts": ["x=service:b"]}},
            "b": {"image": "b", "build": {"additional_contexts": ["y=service:a"]}},
        }
        with pytest.raises(ValueError, match="Circular dependency"):
            eng._resolve_context_dependencies(services)

    def test_services_without_build_skipped(self) -> None:
        eng = _engine()
        services = {"a": {"image": "img"}}
        eng._resolve_context_dependencies(services)  # 不应抛错
        assert services["a"] == {"image": "img"}
