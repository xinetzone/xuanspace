"""xuan_fea.dsl 的单元测试：节点渲染、辅助函数与边界条件。"""

import pytest

from xuan_fea import (
    EMPTY_FEAT_CONTENT,
    EMPTY_FEAT_SYMBOL,
    CharacterVariant,
    Clazz,
    Feature,
    FeatureWithDocs,
    Line,
    Lookup,
    StylisticSet,
    clone_empty,
    cls,
    cls_states,
    create,
    filter_empty,
    flatten_to_lines,
    gly,
    gly_seq,
    ign,
    lang,
    langsys,
    recursive_iterate,
    script,
    subst,
    subst_liga,
    subst_map,
)


def _texts(lines: list[Line]) -> list[str]:
    return [line.text for line in lines]


class TestNodes:
    def test_line_indent_skips_blank(self):
        assert Line("a").indent().level == 1
        assert Line("").indent().level == 0

    def test_clazz_state_and_use(self):
        clazz = Clazz("Digit", ["zero", "one"])
        assert clazz.use() == "@Digit"
        assert clazz.state().text == "@Digit = [zero one];"
        assert Clazz("Empty").glyphs == ()

    def test_clazz_converts_sequence_to_tuple(self):
        glyphs = ["a", "b"]
        assert Clazz("X", glyphs).glyphs == ("a", "b")
        assert Clazz("Y", None).glyphs == ()

    def test_lookup_with_desc(self):
        lookup = Lookup("feature_x", "demo", [Line("sub a by b;")])
        assert _texts(lookup.state()) == [
            "# demo",
            "lookup feature_x {",
            "sub a by b;",
            "} feature_x;",
        ]
        assert lookup.state()[2].level == 1

    def test_lookup_without_desc(self):
        lookup = Lookup("feature_x", None, [Line("sub a by b;")])
        assert _texts(lookup.state()) == [
            "lookup feature_x {",
            "sub a by b;",
            "} feature_x;",
        ]

    def test_lookup_use(self):
        assert Lookup("liga", None, []).use().text == "lookup liga;"

    def test_feature_state(self):
        feature = Feature("calt", [Line("sub a by b;")], "7.9")
        assert _texts(feature.state()) == [
            "feature calt {",
            "",
            "sub a by b;",
            "",
            "} calt;",
        ]
        assert feature.has_lookup is False
        assert feature.use().text == "feature calt;"

    def test_feature_detects_nested_lookup(self):
        lookup = Lookup("liga", None, [Line("sub a by b;")])
        feature = Feature("calt", [Line("# x"), lookup], "7.9")
        assert feature.has_lookup is True
        assert feature.content[0].text == "# x"
        assert feature.content[1] is lookup

    def test_feature_accepts_bare_lookup(self):
        lookup = Lookup("liga", None, [])
        feature = Feature("calt", lookup, "7.9")
        assert feature.has_lookup is True
        assert feature.content == [lookup]

    def test_feature_with_docs(self):
        feature = FeatureWithDocs(1, "cv01", "desc", [], "7.9", "e")
        assert feature.desc_item() == "- [v7.9] cv01: desc"
        assert feature.get_name_lines() == []


class TestCharacterVariant:
    def test_tag_and_name_lines(self):
        cv = CharacterVariant(1, "Slashed `zero` (alt)", [Line("sub a by b;")], "7.9", "0")
        assert cv.tag == "cv01"
        assert _texts(cv.get_name_lines()) == [
            "cvParameters {",
            "FeatUILabelNameID {",
            'name "CV01: Slashed zero";',
            "};",
            "};",
            "",
        ]
        assert cv.get_name_lines()[1].level == 1
        assert cv.get_name_lines()[2].level == 2

    def test_id_bounds(self):
        with pytest.raises(TypeError, match="Character Variants"):
            CharacterVariant(0, "d", [], "1", "e")
        with pytest.raises(TypeError, match="Character Variants"):
            CharacterVariant(100, "d", [], "1", "e")

    def test_empty_symbol_is_stripped_from_name(self):
        cv = CharacterVariant(7, f"{EMPTY_FEAT_SYMBOL}Off", [], "1", "e")
        assert 'name "CV07: Off";' in _texts(cv.get_name_lines())


class TestStylisticSet:
    def test_tag_and_name_lines(self):
        ss = StylisticSet(3, "Round quotes", [Line("sub a by b;")], "7.9", '"')
        assert ss.tag == "ss03"
        assert _texts(ss.get_name_lines()) == [
            "featureNames {",
            'name "SS03: Round quotes";',
            "};",
            "",
        ]
        assert ss.get_name_lines()[1].level == 1

    def test_id_bounds(self):
        with pytest.raises(TypeError, match="Stylistic Sets"):
            StylisticSet(0, "d", [], "1", "e")
        with pytest.raises(TypeError, match="Stylistic Sets"):
            StylisticSet(21, "d", [], "1", "e")


class TestGlyphHelpers:
    def test_gly(self):
        assert gly("_") == "underscore"
        assert gly("++") == "plus_plus.liga"
        assert gly("cl") == "c_l.liga"
        assert gly("--", ".suffix") == "hyphen_hyphen.liga.suffix"
        assert gly("--", ".suffix", True) == "hyphen_hyphen.suffix"
        assert gly("a") == "a"

    def test_gly_accepts_clazz(self):
        clazz = Clazz("Digit", ["zero"])
        assert gly(clazz) == "@Digit"

    def test_gly_seq(self):
        assert gly_seq("{", "sta") == "braceleft.sta.seq"
        assert gly_seq("->", "end") == "hyphen_greater.end.seq"

    def test_cls(self):
        clazz = Clazz("Digit", ["zero", "one"])
        assert cls(["a", "@", "++"], clazz) == "[a at plus_plus.liga @Digit]"
        assert cls("a") == "[a]"

    def test_cls_states(self):
        lines = cls_states(Clazz("A", ["a"]), Clazz("B", ["b"]))
        assert _texts(lines) == [
            "",
            "@A = [a];",
            "@B = [b];",
        ]


class TestRender:
    def test_create_basic(self):
        text = create(
            [
                Line("languagesystem DFLT dflt;"),
                Clazz("A", ["a"]),
                Lookup("liga", None, [Line("sub a by b;")]),
                Feature("calt", [Line("sub c by d;")], "1"),
            ]
        )
        assert text == (
            "languagesystem DFLT dflt;\n"
            "@A = [a];\n"
            "\n"
            "lookup liga {\n"
            "  sub a by b;\n"
            "} liga;\n"
            "\n"
            "feature calt {\n"
            "\n"
            "  sub c by d;\n"
            "\n"
            "} calt;"
        )

    def test_create_custom_indent(self):
        assert create([Line("a", 1)], indent=4) == "    a"

    def test_create_keeps_comment_gap(self):
        assert create([Line("a;"), Line("# note"), Line("b;")]) == "a;\n\n# note\nb;"

    def test_create_empty_content(self):
        assert create([]) == ""

    def test_flatten_to_lines(self):
        clazz = Clazz("A", ["a"])
        assert _texts(flatten_to_lines([clazz, [Line("x")]])) == ["@A = [a];", "x"]

    def test_flatten_skips_falsy(self):
        assert flatten_to_lines([None, "", []]) == []

    def test_flatten_rejects_invalid(self):
        with pytest.raises(TypeError, match="Invalid item"):
            flatten_to_lines([123])

    def test_recursive_iterate(self):
        assert list(recursive_iterate([["a", ("b",)], "c"])) == ["a", "b", "c"]
        assert list(recursive_iterate("abc")) == ["abc"]


class TestRules:
    def test_langsys_lang_script(self):
        assert langsys("DFLT", "dflt").text == "languagesystem DFLT dflt;"
        assert lang("ZHS ").text == "language ZHS ;"
        assert script("hani").text == "script hani;"

    def test_subst_without_context_has_no_marker(self):
        assert subst(None, "a", None, "b").text == "sub a by b;"

    def test_subst_with_prefix(self):
        assert subst(["x"], "a", None, "b").text == "sub x a' by b;"

    def test_subst_with_suffix_clazz(self):
        clazz = Clazz("Digit", ["zero"])
        assert subst(None, "a", clazz, "b").text == "sub a' @Digit by b;"

    def test_subst_accepts_tuple_context(self):
        assert subst(("(", ")"), "b", ("-", "_"), "d").text == "sub parenleft parenright b' hyphen underscore by d;"

    def test_subst_map(self):
        assert _texts(subst_map(["Q", "all", gly("{{")], target_suffix=".cv01")) == [
            "sub Q by Q.cv01;",
            "sub all by all.cv01;",
            "sub braceleft_braceleft.liga by braceleft_braceleft.liga.cv01;",
        ]

    def test_subst_map_accepts_single_string(self):
        assert _texts(subst_map("a", source_suffix=".src", target_suffix=".dst")) == ["sub a.src by a.dst;"]

    def test_ign(self):
        assert ign("{", "b", ["c", "d"]).text == "ignore sub braceleft b' c d;"
        assert ign(None, "b", None).text == "ignore sub b';"

    def test_subst_liga_default(self):
        lookup = subst_liga("!=")
        assert lookup.name == "exclam_equal.liga"
        assert lookup.desc == "!="
        assert _texts(lookup.state()) == [
            "# !=",
            "lookup exclam_equal.liga {",
            "sub SPC equal' by exclam_equal.liga;",
            "sub exclam' equal by SPC;",
            "} exclam_equal.liga;",
        ]

    def test_subst_liga_with_ignores(self):
        lookup = subst_liga("!=", ign_prefix="!", ign_suffix="?", surround=[([], [])])
        assert _texts(lookup.state()) == [
            "# !=",
            "lookup exclam_equal.liga {",
            "ignore sub exclam exclam' equal;",
            "ignore sub exclam' equal question;",
            "sub SPC equal' by exclam_equal.liga;",
            "sub exclam' equal by SPC;",
            "} exclam_equal.liga;",
        ]

    def test_subst_liga_custom_names_and_extra_rules(self):
        lookup = subst_liga(
            ["!", "="],
            target="notequal",
            lookup_name="neq",
            desc="neq",
            extra_rules=[ign("~", "!", ["="])],
        )
        assert _texts(lookup.state()) == [
            "# neq",
            "lookup neq {",
            "ignore sub asciitilde exclam' equal;",
            "sub SPC equal' by notequal;",
            "sub exclam' equal by SPC;",
            "} neq;",
        ]

    def test_subst_liga_list_source_defaults_desc_to_lookup_name(self):
        lookup = subst_liga(["!", "="])
        assert lookup.desc == "exclam_equal.liga"


class TestEmptyFeatureHelpers:
    def test_empty_content(self):
        assert create(EMPTY_FEAT_CONTENT) == "# Placeholder\nsub EMquad by space;"

    def test_clone_empty_character_variant(self):
        cv = CharacterVariant(2, "Origin", [Line("sub a by b;")], "7.9", "x")
        cloned = clone_empty(cv, "OFF: ")
        assert isinstance(cloned, CharacterVariant)
        assert cloned.id == 2
        assert cloned.desc.startswith("OFF: " + EMPTY_FEAT_SYMBOL)
        assert _texts(cloned.content) == _texts(EMPTY_FEAT_CONTENT)
        assert cloned.version == "7.9"

    def test_clone_empty_stylistic_set(self):
        ss = StylisticSet(4, "Origin", [], "7.9", "x")
        assert isinstance(clone_empty(ss), StylisticSet)

    def test_clone_empty_unknown_feature(self):
        with pytest.raises(Exception, match="Unknown feature"):
            clone_empty(FeatureWithDocs(1, "calt", "d", [], "1", "e"))

    def test_filter_empty(self):
        cv = CharacterVariant(1, "Keep", [], "1", "e")
        off = clone_empty(cv)
        assert filter_empty([cv, off], False) == [cv]
        assert filter_empty([cv, off], True) == [cv, off]
