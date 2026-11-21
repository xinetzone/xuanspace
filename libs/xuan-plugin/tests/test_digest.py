"""摘要身份（P0-2）测试：epoch 由 (key, impl_digest, config_digest) 构成。

验收要点见设计文档 §7 P0-2：摘要函数须满足「同输入同输出」与「跨进程一致」，
且不得再依赖对象地址。
"""

import subprocess
import sys
from pathlib import Path

from xuan_plugin.context import Context
from xuan_plugin.plugin import Fiber, InjectSpec, Plugin, _config_digest, _digest, _stable_repr

TESTS_DIR = Path(__file__).resolve().parent


class HexTag:
    """自定义 repr，内含短十六进制字面量（非堆地址）。"""

    def __repr__(self) -> str:
        return "<HexTag 0xff00ff>"


def _no_source_type(name: str) -> type:
    """构造无源码类型：``type()`` 生成的类没有可读取的源文件。"""
    return type(name, (), {})


def sample_impl() -> str:
    """供跨进程一致性测试使用的固定实现。"""
    return "sample"


def other_impl() -> str:
    """与 sample_impl 源码不同的实现。"""
    return "other"


def test_digest_deterministic() -> None:
    """同一对象多次计算摘要相等。"""
    assert _digest(sample_impl) == _digest(sample_impl)


def test_digest_differs_for_different_sources() -> None:
    """源码不同则摘要不同——结构身份随实现变更而变化。"""
    assert _digest(sample_impl) != _digest(other_impl)


def test_digest_contains_no_memory_address() -> None:
    """摘要不得包含对象地址，旧实现的 id() 语义必须消失。"""
    text = _digest(sample_impl)
    assert "0x" not in text
    assert str(id(sample_impl)) not in text


def test_stable_repr_collapses_address() -> None:
    """默认 repr 中的堆地址被折叠为固定占位符。"""
    text = _stable_repr(object())
    assert "0xADDR" in text
    assert _stable_repr(object()) == text


def test_stable_repr_keeps_short_hex_literal() -> None:
    """短十六进制字面量（<8 位）是内容而非地址，必须原样保留。"""
    assert _stable_repr(HexTag()) == "<HexTag 0xff00ff>"


def test_digest_source_less_same_type_instances_share_identity() -> None:
    """无源码类型：同类型实例共享身份——身份属于实现而非某次分配。"""
    assert _digest(object()) == _digest(object())


def test_digest_source_less_types_are_distinguished() -> None:
    """无源码但类型不同，摘要必须不同。"""
    first = _no_source_type("DynamicA")
    second = _no_source_type("DynamicB")
    assert _digest(first()) != _digest(second())


def test_digest_source_less_reproducible_across_processes() -> None:
    """M2：无源码对象的摘要跨进程一致。"""
    code = "from xuan_plugin.plugin import _digest;print(_digest(object()))"
    outputs = [
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=TESTS_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    assert outputs[0] == _digest(object())


def test_digest_scalars() -> None:
    """标量按其字面量构成身份。"""
    assert _digest(1) == _digest(1)
    assert _digest(1) != _digest(2)
    assert _digest("a") != _digest("b")


def test_config_digest_order_insensitive() -> None:
    """配置摘要与键的构造顺序无关。"""
    assert _config_digest({"a": 1, "b": 2}) == _config_digest({"b": 2, "a": 1})


def test_config_digest_none_and_empty_equal() -> None:
    """无配置与空配置取同一哨兵。"""
    assert _config_digest(None) == "none"
    assert _config_digest({}) == "none"


def test_config_digest_detects_value_change() -> None:
    """配置值变化则摘要变化。"""
    assert _config_digest({"a": 1}) != _config_digest({"a": 2})


def test_config_digest_normalizes_addresses() -> None:
    """配置中的不可序列化对象不把堆地址带入摘要。"""
    assert _config_digest({"impl": object()}) == _config_digest({"impl": object()})


def _epoch_for(ctx: Context, config: dict | None) -> str:
    plugin = Plugin(name="p", apply=lambda c, cfg: None, inject=[InjectSpec("dep", config=config)])
    return Fiber(plugin, ctx)._compute_epoch()


def test_epoch_includes_config() -> None:
    """同一实现配不同注入配置必须产生不同 epoch。"""
    ctx = Context()
    ctx.provide("dep", sample_impl)
    assert _epoch_for(ctx, {"a": 1}) != _epoch_for(ctx, {"a": 2})
    assert _epoch_for(ctx, {"a": 1}) == _epoch_for(ctx, {"a": 1})


def test_epoch_matches_digest_composition() -> None:
    """epoch 由 key、实现摘要与配置摘要三段拼接而成。"""
    ctx = Context()
    ctx.provide("dep", sample_impl)
    expected = f"dep:{_digest(sample_impl)}:{_config_digest({'a': 1})}"
    assert _epoch_for(ctx, {"a": 1}) == expected


def test_epoch_reproducible_across_processes() -> None:
    """P0-2 验收：两次独立进程计算同一实现得到相等摘要。"""
    code = "from test_digest import sample_impl;from xuan_plugin.plugin import _digest;print(_digest(sample_impl))"
    outputs = [
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=TESTS_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    assert outputs[0] == _digest(sample_impl)
