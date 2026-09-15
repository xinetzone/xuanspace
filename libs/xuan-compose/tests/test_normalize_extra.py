# SPDX-License-Identifier: GPL-2.0-only
"""normalize 分支补测（上游 test_normalize_service 已覆盖服务归一步径；
这里按上游第 465-2218 行补齐 norm_as_dict/norm_ulimit 全形态、标签
短路、sub_dir 路径重写、depends_on 标签形态与 final 构建上下文分支）。"""

import os

import pytest
import yaml

from xuan_compose.errors import PodmanComposeError
from xuan_compose.merge import OverrideTag, ResetTag
from xuan_compose.normalize import (
    norm_as_dict,
    norm_ulimit,
    normalize,
    normalize_service,
    normalize_service_final,
)


class TestNormAsDict:
    def test_none(self) -> None:
        assert norm_as_dict(None) == {}

    def test_dict_copied(self) -> None:
        src = {"A": "1"}
        out = norm_as_dict(src)
        assert out == {"A": "1"}
        assert out is not src

    def test_list_with_and_without_equals_and_blank_filtered(self) -> None:
        assert norm_as_dict(["A=1", "B", "", "C=x=y"]) == {
            "A": "1",
            "B": None,
            "C": "x=y",
        }

    def test_string_forms(self) -> None:
        assert norm_as_dict("A=1") == {"A": "1"}
        assert norm_as_dict("A") == {"A": None}

    def test_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="dictionary or iterable"):
            norm_as_dict(10)  # type: ignore[arg-type]


class TestNormUlimit:
    def test_soft_only_fills_hard_from_soft(self) -> None:
        assert norm_ulimit({"soft": 1024}) == "1024:1024"

    def test_hard_only_fills_soft_from_hard(self) -> None:
        assert norm_ulimit({"hard": 2048}) == "2048:2048"

    def test_both(self) -> None:
        assert norm_ulimit({"soft": 1024, "hard": 2048}) == "1024:2048"

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="soft or hard"):
            norm_ulimit({"unlimited": True})

    def test_list_form(self) -> None:
        assert norm_ulimit(["soft=1024", "hard=2048"]) == "1024:2048"

    @pytest.mark.parametrize("value", [65536, "65536", -1])
    def test_scalar_passthrough(self, value) -> None:
        assert norm_ulimit(value) == value


class TestNormalizeServiceTags:
    def test_reset_tag_returned_as_is(self) -> None:
        tag = ResetTag()
        assert normalize_service(tag) is tag

    def test_override_tag_unwrapped_to_mapping_value(self) -> None:
        svc = yaml.safe_load(
            "x: !override\n  image: img:1\n  command: [\"a\"]\n"
        )["x"]
        result = normalize_service(svc)
        assert result == {"image": "img:1", "command": ["a"]}
        assert not isinstance(result, OverrideTag)


class TestNormalizeServiceBuild:
    def test_string_build_becomes_context_mapping(self) -> None:
        svc = normalize_service({"build": "./dir"})
        assert svc["build"] == {"context": "./dir"}

    def test_subdir_rewrites_relative_context(self) -> None:
        svc = normalize_service({"build": {"context": "./ctx"}}, sub_dir="sub")
        assert svc["build"]["context"] == os.path.join("sub", "ctx")

    def test_subdir_with_empty_context(self) -> None:
        # context="" + sub_dir：join 后 rstrip 仍非空，结果为 sub_dir 自身
        svc = normalize_service({"build": {"context": ""}}, sub_dir="sub")
        assert svc["build"]["context"] == "sub"

    def test_additional_contexts_dict_to_list(self) -> None:
        svc = normalize_service(
            {"build": {"context": ".", "additional_contexts": {"shared": "/srv/shared"}}}
        )
        assert sorted(svc["build"]["additional_contexts"]) == ["shared=/srv/shared"]

    def test_build_args_dict_to_list(self) -> None:
        svc = normalize_service({"build": {"context": ".", "args": {"A": "1", "B": None}}})
        # None 值走 norm_as_list 的裸键形式（"B"，不拼 "=None"）
        assert sorted(svc["build"]["args"]) == ["A=1", "B"]


class TestNormalizeServiceScalarLists:
    def test_string_fields_wrapped(self) -> None:
        svc = normalize_service(
            {"env_file": ".env", "security_opt": "seccomp:unconfined", "volumes": "/data:/data"}
        )
        assert svc["env_file"] == [".env"]
        assert svc["security_opt"] == ["seccomp=unconfined"]
        assert svc["volumes"] == ["/data:/data"]

    def test_security_opt_known_unconfined_replaced(self) -> None:
        svc = normalize_service(
            {"security_opt": ["seccomp:unconfined", "apparmor:unconfined", "label=user:foo"]}
        )
        assert svc["security_opt"] == [
            "seccomp=unconfined",
            "apparmor=unconfined",
            "label=user:foo",
        ]

    def test_labels_normalized(self) -> None:
        assert normalize_service({"labels": {"a": "1"}})["labels"] == {"a": "1"}
        assert normalize_service({"labels": ["a=1", "b"]})["labels"] == {"a": "1", "b": None}

    def test_extends_string_to_mapping(self) -> None:
        svc = normalize_service({"extends": "base"})
        assert svc["extends"] == {"service": "base"}


class TestNormalizeDependsOn:
    def test_reset_tag_short_circuits(self) -> None:
        svc = {"depends_on": ResetTag(), "image": "x"}
        assert normalize_service(svc) is svc
        assert isinstance(svc["depends_on"], ResetTag)

    def test_string_dependency(self) -> None:
        svc = normalize_service({"depends_on": "db"})
        assert svc["depends_on"] == {"db": {"condition": "service_started"}}

    def test_list_dependencies_get_default_condition(self) -> None:
        svc = normalize_service({"depends_on": ["db", "cache"]})
        assert svc["depends_on"] == {
            "db": {"condition": "service_started"},
            "cache": {"condition": "service_started"},
        }

    def test_explicit_condition_preserved(self) -> None:
        svc = normalize_service({"depends_on": {"db": {"condition": "service_healthy"}}})
        assert svc["depends_on"] == {"db": {"condition": "service_healthy"}}

    def test_override_tag_sequence_becomes_mapping(self) -> None:
        svc = yaml.safe_load("x:\n  depends_on: !override [\"db\", \"cache\"]\n")["x"]
        out = normalize_service(svc)
        deps = out["depends_on"]
        assert isinstance(deps, OverrideTag)
        assert deps.value == {
            "db": {"condition": "service_started"},
            "cache": {"condition": "service_started"},
        }


class TestNormalizeSubdirPaths:
    def test_relative_volume_string_prefixed(self) -> None:
        # 上游怪癖（原样保留）：对整个卷规格串（含 ":/" 目标段）做 join，
        # 不剥离 "./"——POSIX 下结果为 "sub/./data:/data"
        svc = normalize_service({"volumes": ["./data:/data", "/abs:/abs"]}, sub_dir="sub")
        assert svc["volumes"] == [os.path.join("sub", "./data:/data"), "/abs:/abs"]

    def test_relative_volume_dict_source_prefixed(self) -> None:
        svc = normalize_service(
            {"volumes": [{"type": "bind", "source": "./src", "target": "/srv"}]},
            sub_dir="sub",
        )
        # 与 volumes 短格式一致：不剥离 "./"（仅 build.context 分支剥离）
        assert svc["volumes"][0]["source"] == os.path.join("sub", "./src")

    def test_relative_env_file_string_and_dict_path(self) -> None:
        svc = normalize_service(
            {"env_file": ["./.env", {"path": "./dev.env", "required": False},
                          {"path": "/abs.env", "required": True}]},
            sub_dir="sub",
        )
        # 上游原样：短格式与 dict.path 均不剥离 "./"
        assert svc["env_file"][0] == os.path.join("sub", "./.env")
        assert svc["env_file"][1]["path"] == os.path.join("sub", "./dev.env")
        assert svc["env_file"][2]["path"] == "/abs.env"


class TestNormalizeSecrets:
    def test_top_level_string_wrapped_dict_is_error(self) -> None:
        assert normalize_service({"secrets": "token"})["secrets"] == ["token"]
        with pytest.raises(PodmanComposeError, match="secrets must be a list"):
            normalize_service({"secrets": {"token": {}}})

    def test_build_secrets_string_and_dict(self) -> None:
        svc = normalize_service({"build": {"context": ".", "secrets": "token"}})
        assert svc["build"]["secrets"] == ["token"]
        with pytest.raises(PodmanComposeError, match="build.secrets must be a list"):
            normalize_service({"build": {"context": ".", "secrets": {"token": {}}}})


def test_normalize_iterates_services() -> None:
    compose = {"services": {"a": {"build": "./a"}, "b": {"extends": "base"}}}
    normalize(compose, sub_dir="proj")
    assert compose["services"]["a"]["build"]["context"] == os.path.join("proj", "a")
    assert compose["services"]["b"]["extends"] == {"service": "base"}


class TestNormalizeServiceFinal:
    def test_local_context_joined_with_project_dir(self) -> None:
        svc = normalize_service_final({"build": {"context": "ctx"}}, "/proj")
        assert svc["build"]["context"] == os.path.normpath("/proj/ctx")

    def test_string_build_replaced_by_mapping(self) -> None:
        svc = normalize_service_final({"build": "ctx"}, "/proj")
        assert isinstance(svc["build"], dict)
        assert svc["build"]["context"] == os.path.normpath("/proj/ctx")

    def test_git_url_context_left_untouched(self) -> None:
        url = "https://example.com/repo.git#branch:dir"
        svc = normalize_service_final({"build": {"context": url}}, "/proj")
        assert svc["build"]["context"] == url
