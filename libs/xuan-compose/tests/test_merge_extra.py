# SPDX-License-Identifier: GPL-2.0-only
"""merge 模块分支补测（上游 test_can_merge_build/test_rec_merge_depends_on
经引擎路径覆盖了合并主路径；这里在单元层补齐标签 YAML 构造/dump、
reset/override 各分支、volumes 去重、类型冲突、load_yaml_or_die 与
resolve_extends 双形式——全部对应上游第 1764-2365 行真实分支）。"""

import os

import pytest
import yaml

from xuan_compose.merge import (
    OverrideTag,
    ResetTag,
    clone,
    load_yaml_or_die,
    rec_merge,
    rec_merge_one,
    resolve_extends,
)


class TestOverrideTagConstruction:
    def test_mapping_form_becomes_dict(self) -> None:
        loaded = yaml.safe_load("x: !override\n  a: 1\n  b: 2\n")
        tag = loaded["x"]
        assert isinstance(tag, OverrideTag)
        assert tag.value == {"a": "1", "b": "2"}

    def test_sequence_form_becomes_list(self) -> None:
        tag = yaml.safe_load("x: !override [\"a\", \"b\"]\n")["x"]
        assert isinstance(tag, OverrideTag)
        assert tag.value == ["a", "b"]

    def test_mapping_value_with_nested_sequence(self) -> None:
        # node.value[0] 为 (key_node, sequence_node)：item[1].value 是 node 列表；
        # 自定义 from_yaml 直接取 ScalarNode.value，无隐式类型解析，故为字符串
        loaded = yaml.safe_load("x: !override\n  ports:\n    - 80\n    - 81\n")
        assert loaded["x"].value == {"ports": ["80", "81"]}

    def test_empty_mapping(self) -> None:
        tag = yaml.safe_load("x: !override {}\n")["x"]
        assert tag.value == []

    def test_to_yaml_emits_tagged_scalar(self) -> None:
        # to_yaml 契约：represent_scalar(标签, 标量值)；dict/list 值的 dump
        # 上游本身不支持（resolver 会抛），此处白盒覆盖标量路径
        tag = OverrideTag.__new__(OverrideTag)
        tag.value = "scalar-value"
        node = OverrideTag.to_yaml(yaml.SafeDumper(None), tag)
        assert node.tag == "!override"
        assert node.value == "scalar-value"


class TestResetTag:
    def test_load_and_dump(self) -> None:
        tag = yaml.safe_load("x: !reset\n")["x"]
        assert isinstance(tag, ResetTag)
        assert ResetTag.to_json() == "!reset"
        assert "!reset" in yaml.safe_dump({"x": tag})


def test_clone_variants() -> None:
    assert clone([1, 2]) == [1, 2]
    assert clone({"a": 1}) == {"a": 1}
    assert clone("scalar") == "scalar"
    assert clone(5) == 5
    # 容器是浅拷贝
    src = {"a": [1]}
    cp = clone(src)
    assert cp == src
    assert cp is not src


class TestRecMergeTags:
    def test_unneeded_reset_removes_key(self) -> None:
        # target 持 !reset 而 source 无此键：登记信息后键被删除
        target = {"keep": 1, "gone": ResetTag()}
        result = rec_merge_one(target, {"keep": 2})
        assert "gone" not in result
        # 双方都有的标量：第二轮循环以 source 值覆盖
        assert result["keep"] == 2

    def test_unneeded_override_unwrapped(self) -> None:
        tag = yaml.safe_load("x: !override [1, 2]\n")["x"]
        target = {"a": tag}
        result = rec_merge_one(target, {"other": 9})
        assert result["a"] == ["1", "2"]
        assert not isinstance(result["a"], OverrideTag)

    def test_reset_on_both_sides_deletes(self) -> None:
        target = {"a": ResetTag(), "b": 1}
        source = {"a": ResetTag(), "b": 2}
        result = rec_merge_one(target, source)
        assert "a" not in result
        assert result["b"] == 2

    def test_reset_when_target_plain_source_tag(self) -> None:
        result = rec_merge_one({"a": [1]}, {"a": ResetTag()})
        assert "a" not in result

    def test_override_replaces_with_tag_value_target_side(self) -> None:
        tag = yaml.safe_load("x: !override\n  k: v\n")["x"]
        result = rec_merge_one({"a": tag}, {"a": {"k": "old"}})
        assert result["a"] == {"k": "v"}

    def test_override_replaces_with_tag_value_source_side(self) -> None:
        tag = yaml.safe_load("x: !override\n  k: v\n")["x"]
        result = rec_merge_one({"a": {"k": "old"}}, {"a": tag})
        assert result["a"] == {"k": "v"}

    def test_new_source_key_cloned(self) -> None:
        src_list = [1, 2]
        result = rec_merge_one({}, {"new": src_list})
        assert result["new"] == [1, 2]
        assert result["new"] is not src_list


class TestRecMergeListSemantics:
    def test_volumes_dedupes_by_mount_target(self) -> None:
        target = {"volumes": ["/host1:/data", "/old:/shared", "named:/named"]}
        source = {"volumes": ["/host2:/shared", "/host3:/data", "/new:/other"]}
        result = rec_merge_one(target, source)
        # /shared 与 /data 的旧条目按挂载目标去重，源端顺序保留，其余保留
        assert result["volumes"] == [
            "named:/named",
            "/host2:/shared",
            "/host3:/data",
            "/new:/other",
        ]

    def test_plain_list_extends(self) -> None:
        assert rec_merge_one({"x": [1]}, {"x": [2, 3]}) == {"x": [1, 2, 3]}

    def test_scalar_overwrite(self) -> None:
        assert rec_merge_one({"x": 1}, {"x": 2}) == {"x": 2}

    def test_command_and_entrypoint_replace_not_extend(self) -> None:
        assert rec_merge_one({"command": ["a"]}, {"command": ["b"]}) == {"command": ["b"]}
        assert rec_merge_one({"entrypoint": ["sh"]}, {"entrypoint": ["bash"]}) == {
            "entrypoint": ["bash"]
        }

    def test_none_target_merges_into_dict(self) -> None:
        result = rec_merge_one({"labels": None}, {"labels": {"a": "1"}})
        assert result == {"labels": {"a": "1"}}

    def test_type_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"can't merge value of \[x\]"):
            rec_merge_one({"x": [1]}, {"x": {"not": "a list"}})

    def test_rec_merge_multiple_sources(self) -> None:
        assert rec_merge({}, {"a": 1}, {"b": 2}, {"a": 9}) == {"a": 9, "b": 2}


def test_load_yaml_or_die_scanner_error_exits() -> None:
    # tab 缩进气走 ScannerError 分支（flow seq 未闭合属 ParserError，
    # 上游只捕获 ScannerError——区别在此显式固化）
    bad = "services:\n\ta: 1\n"
    with pytest.raises(SystemExit) as ctx:
        load_yaml_or_die("broken.yaml", bad)
    assert ctx.value.code == 1


class TestResolveExtends:
    def test_string_form_shortcut(self) -> None:
        services = {
            "base": {"_deps": [], "image": "base:1", "environment": {"A": "1"}},
            "app": {"extends": "base", "environment": {"B": "2"}},
        }
        resolve_extends(services, ["app"], {})
        merged = services["app"]
        assert merged["image"] == "base:1"
        # command/entrypoint 之外的 dict 递归合并：两个环境变量都在
        assert merged["environment"] == {"A": "1", "B": "2"}
        assert "_deps" not in merged
        # 上游原样行为：service 自身的 extends 声明保留在合并结果中
        assert merged["extends"] == "base"

    def test_without_service_key_is_noop(self) -> None:
        services = {"app": {"extends": {}, "image": "x"}}
        resolve_extends(services, ["app"], {})
        assert services["app"]["image"] == "x"

    def test_missing_deps_raises_descriptive_keyerror(self) -> None:
        services = {
            "base": {"image": "base:1"},  # 无 _deps
            "app": {"extends": {"service": "base"}},
        }
        with pytest.raises(KeyError, match="not found at services.app.extends"):
            resolve_extends(services, ["app"], {})

    def test_local_reference_with_base_own_extends(self) -> None:
        # base 自身带 extends 键：拷出后 del _deps 与 del extends 均存在，安全删除
        services = {
            "base": {"_deps": [], "extends": "root", "image": "base:1"},
            "app": {"extends": {"service": "base"}},
        }
        resolve_extends(services, ["app"], {})
        assert services["app"]["image"] == "base:1"
        # 注意（上游原样行为）：service 自身的 extends 声明经 rec_merge 第三参
        # 仍保留在合并结果中，解析仅消费 from_service 侧的链条
        assert services["app"]["extends"] == {"service": "base"}

    def test_file_reference(self, tmp_path) -> None:
        # file 形式按进程 CWD 解析（与上游 open(filename) 一致），切换 CWD 后还原
        base_file = tmp_path / "base.yaml"
        base_file.write_text(
            "services:\n"
            "  base:\n"
            "    image: base:${TAG}\n"
            "    command: [\"sleep\", \"1\"]\n",
            encoding="utf-8",
        )
        services = {
            "app": {
                "extends": {"file": f"./{base_file.name}", "service": "base"},
                "environment": {"X": "y"},
            }
        }
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            resolve_extends(services, ["app"], {"TAG": "9"})
        finally:
            os.chdir(old_cwd)
        assert services["app"]["image"] == "base:9"
        assert services["app"]["command"] == ["sleep", "1"]
        assert services["app"]["environment"] == {"X": "y"}
