"""OKF frontmatter 解析器：最小 YAML 子集实现（零第三方依赖）。

仅使用 Python 标准库 ``re`` 与 ``pathlib``，不依赖 PyYAML。
支持标量（字符串 / 整数 / 布尔值 / null）、流程列表 ``[a, b]``、
流程映射 ``{k: v}``、块级序列与嵌套映射，以及多行字符串续行。
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Concept

__all__ = [
    "FrontmatterError",
    "parse_frontmatter",
    "parse_concept",
    "validate_type",
]

# §4.1 已知 frontmatter 字段；其余键均归入 Concept.extra
_KNOWN_FIELDS = frozenset(
    {
        "type",
        "title",
        "description",
        "resource",
        "tags",
        "sources",
        "usage_window",
        "generated",
        "verified",
        "status",
        "stale_after",
        "runtime",
        "parameters",
        "computation",
        "executor",
        "attester",
    }
)

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(?P<fm>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

_INT_RE = re.compile(r"[-+]?\d+")


class FrontmatterError(Exception):
    """frontmatter 校验失败（例如缺少必填的 ``type`` 字段）。"""


# ─── 顶层 API ────────────────────────────────────────────────────────────


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """提取并解析文本开头的 YAML frontmatter 块。

    返回 ``(frontmatter, body)``。若文本不以 ``---`` 开头或无法识别
    frontmatter 分隔符，返回 ``({}, text)``。
    """
    if not text.startswith("---"):
        return {}, text
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        return {}, text
    data = _YamlParser(m.group("fm")).parse()
    _normalize_verified(data)
    return data, text[m.end():]


def parse_concept(filepath: Path) -> Concept:
    """读取 markdown 文件并解析为 :class:`Concept` 实例。"""
    text = filepath.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    concept_type = validate_type(frontmatter)
    extra = {k: v for k, v in frontmatter.items() if k not in _KNOWN_FIELDS}
    return Concept(
        path=filepath,
        type=concept_type,
        title=_as_str(frontmatter.get("title")),
        description=_as_str(frontmatter.get("description")),
        resource=_as_str(frontmatter.get("resource")),
        tags=_coerce_tags(frontmatter.get("tags")),
        frontmatter=frontmatter,
        body=body,
        extra=extra,
    )


def validate_type(frontmatter: dict) -> str:
    """校验并返回 frontmatter 的必填 ``type`` 字段。"""
    value = frontmatter.get("type")
    if value is None:
        raise FrontmatterError("missing required frontmatter field: 'type'")
    value = value if isinstance(value, str) else str(value)
    value = value.strip()
    if value == "":
        raise FrontmatterError("frontmatter field 'type' must not be empty")
    return value


# ─── 内部辅助函数 ─────────────────────────────────────────────────────────


def _normalize_verified(data: dict) -> None:
    """§5.2：裸 ``verified`` 映射等价于单元素列表。"""
    if "verified" in data and not isinstance(data["verified"], list):
        data["verified"] = [data["verified"]]


def _as_str(value, default: str = "") -> str:
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)


def _coerce_tags(value) -> list[str]:
    """将 ``tags`` 字段统一为字符串列表。

    支持 YAML 列表 ``[a, b]`` 与逗号分隔字符串 ``"a, b"``。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _strip_comment(s: str) -> str:
    """去除行级 ``#`` 注释（不含引号内与前无空白的 ``#``）。"""
    in_quote: str | None = None
    i = 0
    while i < len(s):
        ch = s[i]
        if in_quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in "\"'":
            in_quote = ch
        elif ch == "#" and (i == 0 or s[i - 1] in " \t"):
            return s[:i].rstrip()
        i += 1
    return s.rstrip()


def _unquote(s: str) -> str:
    """去除单/双引号并解析转义。"""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        inner = s[1:-1]
        if s[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        else:
            inner = inner.replace("''", "'")
        return inner
    return s


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """按顶层分隔符拆分，忽略嵌套容器与引号内的分隔符。"""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
        i += 1
    parts.append(s[start:])
    return parts


def _parse_scalar(s: str):
    """解析单个标量值。"""
    s = _strip_comment(s).strip()
    if s == "":
        return ""
    if s[0] in "\"'":
        return _unquote(s)
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~"):
        return None
    if _INT_RE.fullmatch(s):
        return int(s)
    return s


def _fold_lines(lines: list[str]) -> str:
    """将多行文本折叠为单行：空行表示换行符，否则空格连接。"""
    out: list[str] = []
    for line in lines:
        if line == "":
            out.append("\n")
        else:
            if out and out[-1] != "\n":
                out.append(" ")
            out.append(line)
    return "".join(out).strip()


# ─── 行式递归下降 YAML 子集解析器 ────────────────────────────────────────


class _YamlParser:
    """行式递归下降的最小 YAML 子集解析器。

    支持：
    - 块级映射（``key: value``）
    - 块级序列（``- item``）
    - 流程列表（``[a, b, c]``）
    - 流程映射（``{k: v, k2: v2}``）
    - 多行字符串续行（缩进续行）
    - 嵌套结构
    """

    def __init__(self, text: str):
        self._lines = text.splitlines()
        self._pos = 0

    def parse(self) -> dict:
        return self._parse_mapping(0)

    # ── 行迭代 ──────────────────────────────────────────────────────────

    def _eof(self) -> bool:
        return self._pos >= len(self._lines)

    @staticmethod
    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _skip_blank_and_comment(self) -> None:
        while not self._eof():
            stripped = self._lines[self._pos].strip()
            if stripped == "" or stripped.startswith("#"):
                self._pos += 1
            else:
                return

    def _current_line(self) -> str:
        return self._lines[self._pos]

    def _current_indent(self) -> int:
        return self._indent(self._current_line())

    def _current_content(self) -> str:
        return self._current_line()[self._current_indent():]

    # ── 块级映射 ────────────────────────────────────────────────────────

    def _parse_mapping(self, indent: int) -> dict:
        result: dict = {}
        while True:
            self._skip_blank_and_comment()
            if self._eof():
                break
            if self._current_indent() != indent:
                break
            content = self._current_content()
            if content.startswith("- ") or content == "-":
                break
            key, rest = self._parse_mapping_key(content)
            if key is None:
                break
            self._pos += 1
            result[key] = self._parse_value(rest, indent)
        return result

    def _parse_mapping_key(self, content: str) -> tuple[str | None, str]:
        """从 ``key: value`` 行提取键与剩余部分。"""
        # 处理加引号的键
        if content.startswith('"') or content.startswith("'"):
            q = content[0]
            end = content.find(q + ":", 1)
            if end == -1:
                # 键跨行或格式错误，回退
                return None, ""
            key = _unquote(content[: end + 1])
            rest = content[end + 2:].strip()
            return key, rest
        # 普通键
        colon = content.find(":")
        if colon == -1:
            return None, ""
        key = content[:colon].strip()
        rest = content[colon + 1:].strip()
        return key, rest

    # ── 值解析 ──────────────────────────────────────────────────────────

    def _parse_value(self, rest: str, parent_indent: int):
        """解析 ``key:`` 后的值部分。"""
        rest = rest.strip()
        if rest == "":
            return self._parse_indented_value(parent_indent)
        value = self._parse_flow_or_scalar(rest)
        if isinstance(value, str):
            value = self._fold_continuation(value, parent_indent)
        return value

    def _fold_continuation(self, value: str, parent_indent: int) -> str:
        """将后续缩进续行折叠到标量值中（多行字符串续行）。"""
        lines = [value]
        while True:
            self._skip_blank_and_comment()
            if self._eof():
                break
            if self._current_indent() <= parent_indent:
                break
            content = self._current_content()
            if content.startswith("- ") or content == "-":
                break
            if ":" in content:
                break
            lines.append(self._lines[self._pos].strip())
            self._pos += 1
        if len(lines) == 1:
            return value
        return _fold_lines(lines)

    def _parse_flow_or_scalar(self, s: str):
        """解析可能包含流程集合或标量的单行片段。"""
        s = s.strip()
        if s == "":
            return ""
        if s.startswith("["):
            return self._parse_flow_list(s)
        if s.startswith("{"):
            return self._parse_flow_map(s)
        return _parse_scalar(s)

    def _parse_indented_value(self, parent_indent: int):
        """解析缩进块值：嵌套映射、序列或多行字符串。"""
        self._skip_blank_and_comment()
        if self._eof():
            return ""
        cur = self._current_indent()
        if cur <= parent_indent:
            return ""
        content = self._current_content()
        if content.startswith("- ") or content == "-":
            return self._parse_list(cur)
        if ":" in content:
            stripped = content.strip()
            if not stripped.startswith("-"):
                return self._parse_mapping(cur)
        # 多行字符串续行：收集缩进大于 parent_indent 的行
        return self._parse_multiline_string(parent_indent)

    def _parse_multiline_string(self, parent_indent: int) -> str:
        """收集续行作为多行字符串。"""
        lines: list[str] = []
        while not self._eof():
            cur = self._current_indent()
            if cur <= parent_indent:
                break
            stripped = self._lines[self._pos].strip()
            if stripped == "":
                self._pos += 1
                lines.append("")
            elif stripped.startswith("#"):
                self._pos += 1
            else:
                self._pos += 1
                lines.append(stripped)
        return _fold_lines(lines)

    # ── 块级序列 ────────────────────────────────────────────────────────

    def _parse_list(self, indent: int) -> list:
        result: list = []
        while True:
            self._skip_blank_and_comment()
            if self._eof():
                break
            if self._current_indent() != indent:
                break
            content = self._current_content()
            if not (content.startswith("- ") or content == "-"):
                break
            rest = content[content.index("-") + 1:].strip()
            self._pos += 1
            if rest == "" or rest == "|" or rest == ">":
                result.append(self._parse_indented_value(indent))
            elif rest.startswith("[") or rest.startswith("{"):
                result.append(self._parse_flow_or_scalar(rest))
            elif ":" in rest:
                # 内联映射起始：``- key: value``，后续缩进行为同层映射键值对
                content_indent = indent + 2
                k, vr = self._parse_mapping_key(rest)
                item: dict = {}
                if k is not None:
                    item[k] = self._parse_flow_or_scalar(vr)
                self._continue_mapping(item, content_indent)
                result.append(item)
            else:
                result.append(self._parse_flow_or_scalar(rest))
        return result

    def _continue_mapping(self, result: dict, indent: int) -> None:
        """在给定缩进层级继续解析键值对，写入 ``result``。"""
        while True:
            self._skip_blank_and_comment()
            if self._eof():
                break
            if self._current_indent() != indent:
                break
            content = self._current_content()
            if content.startswith("- ") or content == "-":
                break
            key, rest = self._parse_mapping_key(content)
            if key is None:
                break
            self._pos += 1
            result[key] = self._parse_value(rest, indent)

    # ── 流程集合 ────────────────────────────────────────────────────────

    def _parse_flow_list(self, s: str) -> list:
        """解析 ``[a, b, c]`` 流程列表。"""
        s = s.strip()
        if s.startswith("["):
            s = s[1:]
        if s.endswith("]"):
            s = s[:-1]
        items = _split_top_level(s, ",")
        return [_parse_scalar(item) for item in items]

    def _parse_flow_map(self, s: str) -> dict:
        """解析 ``{k: v, k2: v2}`` 流程映射。"""
        s = s.strip()
        if s.startswith("{"):
            s = s[1:]
        if s.endswith("}"):
            s = s[:-1]
        result: dict = {}
        for pair in _split_top_level(s, ","):
            pair = pair.strip()
            if pair == "":
                continue
            (key, _, val) = pair.partition(":")
            key = key.strip()
            val = val.strip()
            if key.startswith('"') or key.startswith("'"):
                key = _unquote(key)
            result[key] = _parse_scalar(val)
        return result
