# SPDX-License-Identifier: GPL-2.0-only
"""基础层分支补测（上游 tests/unit 无对应文件，按上游真实行为自建）。

覆盖 compat 标量辅助、translate.ports 端口归一、envfile 缺失文件
分支；平台专属 import 分支（nt/posix 二选一）天然只能在运行平台
覆盖，不做跨平台模拟。
"""

import pytest

from xuan_compose.compat import (
    filteri,
    is_list,
    is_relative_ref,
    secondarypathisabs,
    str_to_seconds,
    strverscmp_lt,
    try_float,
    try_int,
    try_parse_bool,
)
from xuan_compose.envfile import dotenv_to_dict
from xuan_compose.translate.ports import norm_ports, port_dict_to_str


class TestIsList:
    # bytes 也是可迭代且非 str/dict——上游 is_list 对 bytes 返回 True（原样保留）
    @pytest.mark.parametrize("value", [[], [1], (1, 2), {1, 2}, b"bytes"])
    def test_iterable_non_scalar(self, value) -> None:
        assert is_list(value) is True

    @pytest.mark.parametrize("value", ["str", {"a": 1}, 1, None, 1.5])
    def test_scalar_and_mapping(self, value) -> None:
        assert is_list(value) is False


def test_is_relative_ref_prefixes() -> None:
    assert is_relative_ref("./a.yaml")
    assert is_relative_ref(".:/x")
    assert is_relative_ref("../a.yaml")
    assert is_relative_ref("..:/x")
    assert not is_relative_ref("/abs/path")
    assert not is_relative_ref("image:tag")


def test_filteri_drops_falsy_keeps_strings() -> None:
    assert filteri(["a", None, "", "b"]) == ["a", "b"]


def test_secondary_path_abs_checker_loaded() -> None:
    # 双平台交叉判定函数必须可用（posix 上判 NT 盘符，nt 上判 POSIX 斜杠）
    assert callable(secondarypathisabs)


class TestTryNumbers:
    def test_try_int(self) -> None:
        assert try_int("12", 0) == 12
        assert try_int(5, 0) == 5
        assert try_int("x", 7) == 7
        assert try_int("x", None) is None
        assert try_int(None, 3) == 3  # type: ignore[arg-type]

    def test_try_float(self) -> None:
        assert try_float("1.5", 0.0) == 1.5
        assert try_float("x", None) is None
        assert try_float([], -1.0) == -1.0  # type: ignore[arg-type]


class TestStrToSeconds:
    def test_falsy_returns_none(self) -> None:
        assert str_to_seconds(None) is None
        assert str_to_seconds("") is None
        assert str_to_seconds(0) is None

    def test_number_passthrough(self) -> None:
        assert str_to_seconds(30) == 30
        assert str_to_seconds(2.5) == 2.5

    def test_seconds_only(self) -> None:
        assert str_to_seconds("30s") == 30
        assert str_to_seconds("30") == 30
        assert str_to_seconds("1.5s") == 1

    def test_minutes_and_seconds(self) -> None:
        # 上游正则 [m:] 接受 "1:30"（冒号分隔）与 "1m30s"（m 后直接秒）
        assert str_to_seconds("1:30") == 90
        assert str_to_seconds("1m30s") == 90
        assert str_to_seconds("2:00") == 120

    def test_unparsable_returns_none(self) -> None:
        assert str_to_seconds("abc") is None
        # "1m:30s" 同时含 m 与冒号，切不出合法两组——上游怪癖，返回 None
        assert str_to_seconds("1m:30s") is None


class TestVersionCompare:
    def test_basic_ordering(self) -> None:
        assert strverscmp_lt("1.0.0", "1.0.1")
        assert not strverscmp_lt("2.0", "1.9.9")
        assert not strverscmp_lt("1.0", "1.0")

    def test_empty_operand(self) -> None:
        # ``a or ""``：None 与 "" 都归一为空版本序列，小于任意非空版本
        assert strverscmp_lt("", "1")
        assert not strverscmp_lt("1", "")
        assert strverscmp_lt(None, "1")  # type: ignore[arg-type]


class TestTryParseBool:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("0", False),
            (1, True),
            (0, False),
            ("yes", None),
            (2, None),
            ([], None),
        ],
    )
    def test_matrix(self, value, expected) -> None:
        assert try_parse_bool(value) is expected


class TestPortDictToStr:
    def test_minimal_target_only(self) -> None:
        assert port_dict_to_str({"target": 8080}) == "8080"

    def test_published(self) -> None:
        # target/published 原样插值（int 经 f-string 转 "80"），不做语义翻转
        assert port_dict_to_str({"target": 80, "published": 8080}) == "8080:80"

    def test_host_ip_and_published(self) -> None:
        assert port_dict_to_str(
            {"target": 80, "published": 8081, "host_ip": "127.0.0.1"}
        ) == "127.0.0.1:8081:80"

    def test_non_tcp_protocol_suffix(self) -> None:
        assert port_dict_to_str({"target": 53, "protocol": "udp"}) == "53/udp"
        assert port_dict_to_str(
            {"target": 53, "published": 5353, "protocol": "udp"}
        ) == "5353:53/udp"

    def test_target_required(self) -> None:
        with pytest.raises(ValueError, match="target container port"):
            port_dict_to_str({"published": 80})
        with pytest.raises(ValueError, match="target container port"):
            port_dict_to_str({})


class TestNormPorts:
    def test_none_becomes_empty(self) -> None:
        assert norm_ports(None) == []

    def test_string_wrapped(self) -> None:
        assert norm_ports("8080:80") == ["8080:80"]

    def test_mixed_list(self) -> None:
        assert norm_ports(["80:80", 53, {"target": 443}]) == [
            "80:80",
            "53",
            "443",
        ]

    def test_top_level_scalar_int_rejected(self) -> None:
        # 上游怪癖（原样保留）：仅字符串顶层值被包装，顶层 int 过不了
        # ``assert isinstance(ports_in, list)``；int 只支持列表元素
        with pytest.raises(AssertionError):
            norm_ports(9000)  # type: ignore[arg-type]

    def test_invalid_element_type(self) -> None:
        with pytest.raises(TypeError, match="string or dict"):
            norm_ports([["nested"]])  # type: ignore[list-item]


def test_dotenv_missing_file_returns_empty(tmp_path) -> None:
    assert dotenv_to_dict(str(tmp_path / "does-not-exist.env")) == {}
