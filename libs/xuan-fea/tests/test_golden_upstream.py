"""与上游 Maple Mono ``source/py/feature/ast.py`` 的逐字节对照。

做法：把同一棵节点树分别交给上游实现与本库，比较 ``create()`` 结果、
各辅助函数输出与缩进层级。上游源码不在工作区时自动跳过。
"""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

import xuan_fea

_UPSTREAM = (
    Path(__file__).resolve().parents[9]
    / "notebook"
    / "libs"
    / "maple-font"
    / "source"
    / "py"
    / "feature"
    / "ast.py"
)

pytestmark = pytest.mark.skipif(
    not _UPSTREAM.exists(), reason=f"上游源码不存在：{_UPSTREAM}"
)


def _load_upstream() -> ModuleType:
    name = "maple_feature_ast_reference"
    if name in sys.modules:
        return sys.modules[name]

    spec = spec_from_file_location(name, _UPSTREAM)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(m: ModuleType) -> dict:
    """用同一个构造脚本驱动任一实现，返回可比较的文本快照。"""
    digit = m.Clazz("Digit", ["zero", "one", "two"])
    pair = m.Clazz("Pair", ["parenleft", "parenright"])

    lookup = m.subst_liga(
        "!=",
        lookup_name="exclam_equal.liga",
        desc="!=",
        surround=[([], []), (["a"], ["b"])],
        ign_prefix="!",
        ign_suffix="?",
        extra_rules=[m.ign("~", "!", ["="])],
    )
    cv = m.CharacterVariant(
        1,
        "Slashed `zero` (alt)",
        [m.subst(None, "zero", None, "zero.slash"), digit],
        "7.9",
        "0",
    )
    ss = m.StylisticSet(
        3,
        "Round quotes",
        [m.subst_map(["Q", "all"], target_suffix=".ss03")],
        "7.9",
        '"',
    )
    feature = m.Feature(
        "calt", [lookup, m.Line("sub one two by one_two.liga;")], "7.9"
    )

    return {
        "gly": [
            m.gly("_"),
            m.gly("++"),
            m.gly("cl"),
            m.gly("--", ".suffix"),
            m.gly("--", ".suffix", True),
            m.gly(digit),
        ],
        "gly_seq": [m.gly_seq("{", "sta"), m.gly_seq("->", "end")],
        "cls": m.cls(["a", "@", "++"], digit, pair),
        "cls_states": [line.text for line in m.cls_states(digit, pair)],
        "subst": [
            m.subst(None, "a", None, "b").text,
            m.subst(["-"], "b", pair, "d").text,
            m.subst(["x"], "a", None, "b").text,
        ],
        "subst_map": [line.text for line in m.subst_map(["Q", "all", m.gly("{{")], target_suffix=".cv01")],
        "ign": [
            m.ign("{", "b", ["c", "d"]).text,
            m.ign(None, "b", None).text,
        ],
        "lookup": [line.text for line in lookup.state()],
        "lookup_levels": [line.level for line in lookup.state()],
        "cv": [line.text for line in cv.get_name_lines()],
        "ss": [line.text for line in ss.get_name_lines()],
        "feature": [line.text for line in feature.state()],
        "feature_levels": [line.level for line in feature.state()],
        "desc_item": cv.desc_item(),
        "create": m.create([digit, pair, cv, ss, feature, m.EMPTY_FEAT_CONTENT]),
        "create_indent4": m.create([digit, feature], indent=4),
        "empty_content": m.create(m.EMPTY_FEAT_CONTENT),
        "clone_empty": m.create([m.clone_empty(cv, "OFF: ")]),
        "filter_empty": [
            item.tag for item in m.filter_empty([cv, m.clone_empty(ss)], False)
        ],
        "flatten": [line.text for line in m.flatten_to_lines([digit, m.Line("x")])],
        "lang": [m.langsys("DFLT", "dflt").text, m.lang("ZHS ").text, m.script("hani").text],
    }


def test_snapshot_matches_upstream():
    assert _snapshot(xuan_fea) == _snapshot(_load_upstream())


def test_create_output_is_byte_stable():
    text = _snapshot(xuan_fea)["create"]
    assert "\t" not in text
    assert text == text.rstrip("\n")
