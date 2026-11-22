"""OpenType Feature（``.fea``）AST 构建原语。

本模块把 ``.fea`` 特征文件视为一棵可组合的语法树：``Line`` 是叶子，
``Clazz`` / ``Lookup`` / ``Feature`` 是中间节点，``create()`` 把整棵树渲染成
缩进稳定、空行规范的字符串。

与直接拼接字符串相比，这种写法换来三点确定性：

1. 嵌套结构由对象层级表达，缩进在渲染期统一计算，规则不会因为拼接顺序漂移；
2. 类、查找块、特征块之间的空行由 ``create()`` 集中处理，调用方无需关心格式；
3. 每个节点自带名称与描述，渲染结果同时就是可读的注释与文档。

来源：Maple Mono ``scripts/feature/ast.py``（OFL-1.1）的等价萃取，
渲染输出与原实现逐字节一致。实现上以 ``dataclasses``（``frozen`` + ``slots``）
承载节点，自动获得初始化、值相等与内存布局；另修正了原实现中一处不影响输出的
异常消息拼写。因 ``frozen`` 生效，节点属性不再允许赋值。
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "EMPTY_FEAT_CONTENT",
    "EMPTY_FEAT_SYMBOL",
    "SPC",
    "CharacterVariant",
    "Clazz",
    "Feature",
    "FeatureContent",
    "FeatureWithDocs",
    "GlyphContext",
    "Line",
    "Lookup",
    "StylisticSet",
    "cls",
    "cls_states",
    "clone_empty",
    "create",
    "filter_empty",
    "flatten_to_lines",
    "gly",
    "gly_seq",
    "ign",
    "lang",
    "langsys",
    "recursive_iterate",
    "script",
    "subst",
    "subst_liga",
    "subst_map",
]


@dataclass(frozen=True, slots=True)
class Line:
    """渲染叶子节点：一行待输出文本及其缩进层级。"""

    text: str
    level: int = 0

    def indent(self) -> Line:
        """返回缩进一级的副本；空行不缩进。"""
        return Line(self.text, self.level + 1 if self.text else 0)

    def __repr__(self) -> str:
        return f"Line({self.text!r}, {self.level})"


@dataclass(frozen=True, slots=True)
class Clazz:
    """字形类节点，渲染为 ``@Name = [g1 g2 ...];``。"""

    name: str
    # 引号不可去：类体内 Clazz 尚未绑定，去引号后注解会在类创建期求值并 NameError
    glyphs: "Sequence[str | Clazz] | None" = None  # noqa: UP037

    def __post_init__(self) -> None:
        glyphs = () if self.glyphs is None else self.glyphs
        object.__setattr__(self, "glyphs", tuple(glyphs))

    def use(self) -> str:
        """在规则中引用本类的写法：``@Name``。"""
        return f"@{self.name}"

    def state(self) -> Line:
        return Line(f"{self.use()} = {cls(self.glyphs)};")


# 可出现在规则上下文位置的字形描述：单个字形名、字形类、序列或空
type GlyphContext = str | Clazz | Sequence[str | Clazz] | None


@dataclass(frozen=True, slots=True)
class Lookup:
    """查找块节点，渲染为带注释的 ``lookup Name { ... } Name;``。"""

    name: str
    desc: str | None
    content: list

    def use(self) -> Line:
        return Line(f"lookup {self.name};")

    def state(self) -> list[Line]:
        arr = []

        if self.desc:
            arr.append(Line(f"# {self.desc}"))

        arr.append(Line(f"lookup {self.name} {{"))

        for c in flatten_to_lines(self.content):
            arr.append(c.indent())

        arr.append(Line(f"}} {self.name};"))
        return arr


# 特征块可承载的内容，允许任意深度的序列嵌套
type FeatureContent = Clazz | Lookup | Line | Sequence[FeatureContent] | None


@dataclass(frozen=True, slots=True)
class Feature:
    """特征块节点，渲染为 ``feature tag { ... } tag;``。"""

    tag: str
    content: Clazz | Lookup | Line | list
    version: str
    # 由 content 推导，不进入构造签名；置于末位以保持 (tag, content, version) 的位置参数
    has_lookup: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """把 content 归一化为 list，并标记是否含有查找块。"""
        raw = self.content
        if isinstance(raw, Lookup):
            normalized, has_lookup = [raw], True
        elif isinstance(raw, list):
            normalized, has_lookup = [], False
            for item in recursive_iterate(raw):
                if isinstance(item, Lookup):
                    has_lookup = True
                normalized.append(item)
        else:
            normalized, has_lookup = [raw], False
        object.__setattr__(self, "content", normalized)
        object.__setattr__(self, "has_lookup", has_lookup)

    def use(self) -> Line:
        return Line(f"feature {self.tag};")

    def get_name_lines(self) -> list[Line]:
        return []

    def state(self) -> list[Line]:
        target = []
        for c in self.get_name_lines() + flatten_to_lines(self.content):
            target.append(c.indent())

        return [
            Line(f"feature {self.tag} {{"),
            Line(""),
            *target,
            Line(""),
            Line(f"}} {self.tag};"),
        ]


# 描述文本中被括号包裹的部分不进入特征标签名
REGEXP = r"\(.*\)"


@dataclass(frozen=True, slots=True, init=False)
class FeatureWithDocs(Feature):
    """带版本号、描述与示例的特征，供文档表与版本清单复用。"""

    id: int
    desc: str
    example: str

    # init=False：dataclass 会先排基类字段再排子类字段，自动签名将变成
    # (tag, content, version, id, desc, example)，破坏与原实现一致的位置参数，
    # 因此保留手写签名并在调用基类后显式赋值。
    def __init__(
        self,
        id: int,
        tag: str,
        desc: str,
        content: Clazz | Lookup | Line | list,
        version: str,
        example: str,
    ) -> None:
        Feature.__init__(self, tag, content, version)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "desc", desc)
        object.__setattr__(self, "example", example)

    def desc_item(self):
        return f"- [v{self.version}] {self.tag}: {self.desc}"


@dataclass(frozen=True, slots=True, init=False)
class CharacterVariant(FeatureWithDocs):
    """字符变体（``cv01``–``cv99``），自动补 ``cvParameters`` 标签名。"""

    def __init__(
        self,
        id: int,
        desc: str,
        content: Clazz | Lookup | Line | list,
        version: str,
        example: str,
    ) -> None:
        if id < 1 or id > 99:
            raise TypeError(f"id should > 0 and < 100 in Character Variants, current is {id}")
        FeatureWithDocs.__init__(
            self,
            id=id,
            tag=f"cv{id:02d}",
            desc=desc,
            content=content,
            version=version,
            example=example,
        )

    def get_name_lines(self) -> list[Line]:
        _name = re.sub(REGEXP, "", self.desc.replace("`", "").replace(EMPTY_FEAT_SYMBOL, " ")).strip()
        return [
            Line("cvParameters {"),
            Line("FeatUILabelNameID {", 1),
            Line(f'name "{self.tag.upper()}: {_name}";', 2),
            Line("};", 1),
            Line("};"),
            Line(""),
        ]


@dataclass(frozen=True, slots=True, init=False)
class StylisticSet(FeatureWithDocs):
    """样式集（``ss01``–``ss20``），自动补 ``featureNames`` 标签名。"""

    def __init__(
        self,
        id: int,
        desc: str,
        content: Clazz | Lookup | Line | list,
        version: str,
        example: str,
    ) -> None:
        if id < 1 or id > 20:
            raise TypeError(f"id should > 0 and < 20 in Stylistic Sets, current is {id}")

        FeatureWithDocs.__init__(
            self,
            id=id,
            tag=f"ss{id:02d}",
            desc=desc,
            content=content,
            version=version,
            example=example,
        )

    def get_name_lines(self) -> list[Line]:
        _name = re.sub(REGEXP, "", self.desc.replace("`", " ")).strip()
        return [
            Line("featureNames {"),
            Line(f'name "{self.tag.upper()}: {_name}";', 1),
            Line("};"),
            Line(""),
        ]


# 描述中表示「该特征已被清空」的占位符号
EMPTY_FEAT_SYMBOL = "$$$"

_PUNCTUATION_MAP = {
    "{": "braceleft",
    "}": "braceright",
    "[": "bracketleft",
    "]": "bracketright",
    "(": "parenleft",
    ")": "parenright",
    "<": "less",
    ">": "greater",
    ".": "period",
    ",": "comma",
    "-": "hyphen",
    "_": "underscore",
    "=": "equal",
    "+": "plus",
    ":": "colon",
    ";": "semicolon",
    "?": "question",
    "!": "exclam",
    "@": "at",
    "#": "numbersign",
    "$": "dollar",
    "%": "percent",
    "^": "asciicircum",
    "&": "ampersand",
    "*": "asterisk",
    "'": "quotesingle",
    '"': "quotedbl",
    "/": "slash",
    "\\": "backslash",
    "|": "bar",
    "`": "grave",
    "~": "asciitilde",
}

_PUNCTUATION_CN_MAP = {
    "“": "quotedblleft",
    "”": "quotedblright",
    "‘": "quoteleft",
    "’": "quoteright",
    "…": "ellipsis",
    "—": "emdash",
}


def _gly_name(g: GlyphContext) -> str:
    """把单个字形（或字形序列）转换为 ``.fea`` 中的字形名。"""
    if not g:
        return ""
    if isinstance(g, Clazz):
        return g.use()
    if isinstance(g, str):
        if g in _PUNCTUATION_MAP:
            return _PUNCTUATION_MAP[g]
        if g in _PUNCTUATION_CN_MAP:
            return _PUNCTUATION_CN_MAP[g]
        return g
    if isinstance(g, Sequence):
        return " ".join(_gly_name(item) for item in g)
    raise TypeError(f"{g}({type(g)}) is invalid for glyph name")


def _prefix(data: GlyphContext) -> str:
    if data:
        return _gly_name(data) + " "
    return ""


def _suffix(data: GlyphContext) -> str:
    if data:
        return " " + _gly_name(data)
    return ""


def _subst(source: str, target: str) -> Line:
    return Line(f"sub {source} by {target};")


def _parse_glyph(g: str | Clazz) -> str:
    if isinstance(g, str) and len(g) > 1 and g[0] in _PUNCTUATION_MAP:
        return "_".join(map(_gly_name, list(g))) + ".liga"
    return _gly_name(g)


SPC = "SPC"


def gly(g: str | Clazz | Sequence[str | Clazz], suffix: str = "", overwrite: bool = False):
    """规范化字形名。

    无后缀且 ``len(g) > 1`` 时自动补 ``.liga``，否则保持原名：

    >>> gly("_")
    'underscore'
    >>> gly("++")
    'plus_plus.liga'
    >>> gly("cl")
    'c_l.liga'
    >>> gly("--", ".suffix")
    'hyphen_hyphen.liga.suffix'
    >>> gly("--", ".suffix", True)
    'hyphen_hyphen.suffix'
    """
    if not isinstance(g, Clazz) and len(g) > 1:
        suf = suffix if overwrite else (".liga" + suffix)
        return "_".join(map(_gly_name, list(g))) + suf
    return _gly_name(g) + suffix


def gly_seq(g: str | list[str], variant: Literal["sta", "mid", "end"]):
    """生成序列位置变体字形名。

    >>> gly_seq("{", "sta")
    'braceleft.sta.seq'
    """
    return gly(g, f".{variant}.seq", True)


def cls(glyphs: str | Clazz | Sequence[str | Clazz], *rest: str | Clazz) -> str:
    """生成内联字形类 ``[...]``。

    >>> clazz = Clazz("Digit", ["zero", "one"])
    >>> cls(["a", "@", "++"], clazz)
    '[a at plus_plus.liga @Digit]'
    """
    glyphs_list = list(recursive_iterate(glyphs)) + list(rest)
    return "[" + " ".join([_parse_glyph(g) for g in glyphs_list]) + "]"


def cls_states(*classes: Clazz) -> list[Line]:
    """声明多个类，并在最前面留一个空行作为段落分隔。"""
    return [Line("")] + flatten_to_lines(classes)


def create(content: list, indent: int = 2) -> str:
    """把节点树渲染为完整的 ``.fea`` 文本。

    渲染期做两件事：折叠连续空行；在注释、查找块、特征块之前补一个空行。
    """
    lines = []
    for line in flatten_to_lines(content):
        # 跳过重复空行
        if not line.text and lines and not lines[-1].text:
            continue

        # 在特征块与查找块之前补空行
        if lines and lines[-1].text and not lines[-1].text.startswith("#"):
            if (
                line.text.startswith("#")
                or (line.text.startswith("lookup") and not line.text.endswith(";"))
                or (line.text.startswith("feature ") and line.text.endswith("{"))
            ):
                lines.append(Line(""))

        lines.append(line)

    return "\n".join(f"{' ' * (indent * line.level)}{line.text}" for line in lines)


def langsys(script_name: str, lang_name: str) -> Line:
    return Line(f"languagesystem {script_name} {lang_name};")


def lang(lang_name: str) -> Line:
    return Line(f"language {lang_name};")


def script(script_name: str) -> Line:
    return Line(f"script {script_name};")


def subst(
    prefix: GlyphContext,
    glyph: str | Clazz,
    suffix: GlyphContext,
    replace: str | Clazz,
) -> Line:
    """生成单条替换规则。

    仅当存在上文或下文时，才在被替换字形后加 ``'`` 标记。

    >>> clazz = Clazz("Digit", ["zero", "one"])
    >>> subst(["-"], "b", clazz, "d")
    Line("sub hyphen b' @Digit by d;", 0)
    """
    marker = "'"
    if not prefix and not suffix:
        marker = ""
    return _subst(
        f"{_prefix(prefix)}{_gly_name(glyph)}{marker}{_suffix(suffix)}",
        f"{_gly_name(replace)}",
    )


def subst_map(
    glyphs: str | list[str],
    source_suffix: str = "",
    target_suffix: str = "",
) -> list[Line]:
    """为一组字形批量生成「加后缀」替换规则。

    >>> subst_map(["Q", "all", gly("{{")], target_suffix=".cv01")
    [Line("sub Q by Q.cv01;", 0), Line("sub all by all.cv01;", 0), Line("sub braceleft_braceleft.liga by braceleft_braceleft.liga.cv01;", 0)]
    """
    result = []

    if isinstance(glyphs, str):
        glyphs = [glyphs]

    for g in glyphs:
        _g = _parse_glyph(g)
        result.append(_subst(f"{_g}{source_suffix}", f"{_g}{target_suffix}"))

    return result


def subst_liga(
    source: str | list[str],
    target: str | None = None,
    lookup_name: str | None = None,
    desc: str | None = None,
    surround: list[tuple[GlyphContext, GlyphContext]] | None = None,
    ign_prefix: str | Clazz | None = None,
    ign_suffix: str | Clazz | None = None,
    extra_rules: list[Line] | None = None,
) -> Lookup:
    """生成连字替换查找块。

    默认 ``target`` 与 ``lookup_name`` 都取 ``gly(source)``；``surround`` 给出
    上文/下文匹配序列，``ign_prefix`` / ``ign_suffix`` 生成排除规则。

    Args:
        source: 组成连字的字形，如 ``"!="`` 或 ``["!", "="]``。
        target: 连字字形名，默认 ``gly(source)``。
        lookup_name: 查找块名称，默认同 ``target``。
        desc: 查找块前注释，默认取 ``source``；``source`` 为列表时取 ``lookup_name``。
        surround: ``(上文, 下文)`` 序列列表；为空时生成无上下文规则。
        ign_prefix: 出现在连字前面即阻止替换的字形。
        ign_suffix: 出现在连字后面即阻止替换的字形。
        extra_rules: 插在主规则与自动生成的排除规则之间的额外规则。

    Returns:
        可直接嵌入 ``Feature`` 的 ``Lookup`` 节点。

    Examples:
        >>> lookup = subst_liga("!=")
        >>> create([lookup])
        '# !=\\nlookup exclam_equal.liga {\\n  sub SPC equal\\' by exclam_equal.liga;\\n  sub exclam\\' equal by SPC;\\n} exclam_equal.liga;'
    """
    source_arr = list(source)
    if not target:
        target = gly(source)
    if not lookup_name:
        lookup_name = target
    if not desc:
        desc = source if isinstance(source, str) else lookup_name
    if extra_rules is None:
        extra_rules = []

    def to_list(item):
        if item is None:
            return []
        if isinstance(item, (str, Clazz)):
            return [item]
        return list(item)

    generated_ignores = []
    if ign_prefix:
        generated_ignores.append(ign(ign_prefix, source_arr[0], source_arr[1:]))
    if ign_suffix:
        generated_ignores.append(ign(None, source_arr[0], source_arr[1:] + [ign_suffix]))

    subst_rules = []
    if not surround:
        surround = [([], [])]

    for prfx, sfx in surround:
        prfx_list = to_list(prfx)
        sfx_list = to_list(sfx)
        n = len(source_arr)

        for i in range(n - 1):
            subst_prefix = prfx_list + [SPC] * i
            subst_suffix = source_arr[i + 1 :] + sfx_list
            subst_rules.insert(0, subst(subst_prefix, source_arr[i], subst_suffix, SPC))

        subst_prefix = prfx_list + [SPC] * (n - 1)
        subst_rules.insert(0, subst(subst_prefix, source_arr[-1], sfx_list, target))

    return Lookup(
        lookup_name,
        desc,
        generated_ignores + extra_rules + subst_rules,
    )


def ign(
    prefix: GlyphContext,
    glyph: str | Clazz,
    suffix: GlyphContext,
) -> Line:
    """生成排除规则 ``ignore sub ...``。

    >>> ign("{", "b", ["c", "d"])
    Line("ignore sub braceleft b' c d;", 0)
    """
    return Line(f"ignore sub {_prefix(prefix)}{_gly_name(glyph)}'{_suffix(suffix)};")


def recursive_iterate(data: Sequence | str | object) -> Iterator:
    """深度优先展开任意嵌套序列；字符串视为叶子。"""
    if isinstance(data, Sequence) and not isinstance(data, str):
        for item in data:
            yield from recursive_iterate(item)
    else:
        yield data


def flatten_to_lines(
    data: Line | Clazz | Lookup | Feature | list | tuple,
) -> list[Line]:
    """把任意节点树展开为扁平的 ``Line`` 序列。"""
    result = []

    for item in recursive_iterate(data):
        if not item:
            continue
        elif isinstance(item, Clazz):
            result.append(item.state())
        elif isinstance(item, Line):
            result.append(item)
        elif isinstance(item, (Lookup, Feature)):
            result.extend(item.state())
        else:
            raise TypeError(f"Invalid item to flatten: {item}")

    return result


# 被清空的特征仍需要一个占位规则，避免 fontTools 告警
EMPTY_FEAT_CONTENT = [Line("# Placeholder"), subst(None, "EMquad", None, "space")]


def clone_empty(feature: FeatureWithDocs, desc_prefix: str = ""):
    """克隆一个仅含占位内容的同名特征，用于「关闭某特征」的构建分支。"""
    if isinstance(feature, CharacterVariant):
        return CharacterVariant(
            id=feature.id,
            desc=desc_prefix + EMPTY_FEAT_SYMBOL + feature.desc,
            content=EMPTY_FEAT_CONTENT,
            version=feature.version,
            example=feature.example,
        )
    if isinstance(feature, StylisticSet):
        return StylisticSet(
            id=feature.id,
            desc=desc_prefix + EMPTY_FEAT_SYMBOL + feature.desc,
            content=EMPTY_FEAT_CONTENT,
            version=feature.version,
            example=feature.example,
        )
    raise Exception(f"Unknown feature: {feature.tag}")


def filter_empty(features: list[FeatureWithDocs], full: bool):
    """按是否保留占位特征过滤列表；``full=True`` 时原样返回。"""
    if full:
        return features

    return list(filter(lambda x: EMPTY_FEAT_SYMBOL not in x.desc, features))
