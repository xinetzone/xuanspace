# SPDX-License-Identifier: GPL-2.0-only
"""argparse parser 树序列化器（TR-9.1 对等快照的共享提取层）。

本模块 vendor 无关：既被快照生成脚本
（``tests/generate_parity_snapshots.py``，从 vendor 上游提取）使用，
也被新库对等测试（``tests/test_cli_parity.py``）使用，保证两侧提取
逻辑严格同一——对比可信度不依赖"两边各写一遍提取器"。

序列化字段刻意只包含 AC-4 列举的参数表维度（option strings / action /
default / nargs / required）外加 const/choices/metavar/type/dest/help
（help 文本逐行翻译，纳入对比可捕获文案漂移）；**prog 明确不采集**
（AC-4 允许差异：``podman-compose`` vs ``python -m xuan_compose``）。
"""

import argparse
from typing import Any

#: default 遇到 argparse.SUPPRESS 时的 JSON 安全标记
SUPPRESS_MARKER = "__ARGPARSE_SUPPRESS__"


def normalize_value(value: Any) -> Any:
    """把 argparse 属性值转为 JSON 可序列化的原生结构。"""
    if value is argparse.SUPPRESS:
        return SUPPRESS_MARKER
    if isinstance(value, tuple):
        return [normalize_value(v) for v in value]
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    return value


def serialize_action(action: argparse.Action) -> dict[str, Any]:
    """序列化单个 action（位置参数与可选参数统一形态）。"""
    action_type = getattr(action.type, "__name__", None) if action.type is not None else None
    choices = list(action.choices) if action.choices is not None else None
    return {
        "option_strings": list(action.option_strings),
        "dest": action.dest,
        "action": type(action).__name__,
        "nargs": action.nargs,
        "const": normalize_value(action.const),
        "default": normalize_value(action.default),
        "choices": choices,
        "required": bool(action.required),
        "metavar": action.metavar,
        "type": action_type,
        "help": action.help,
    }


def _choice_help_map(sub_action: argparse._SubParsersAction[Any]) -> dict[str, str]:
    """add_parser(help=...) 的 help 挂在父 SubParsersAction 的选择项上。"""
    return {item.dest: item.help for item in sub_action._choices_actions}


def serialize_parser_tree(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """序列化完整 parser 树：全局参数 + 子命令序 + 每子命令参数表。

    输出结构（JSON 安全）::

        {
          "global": {"options": [action, ...]},
          "command_order": ["help", "ls", ...],
          "commands": {name: {"help", "description", "options"}}
        }
    """
    sub_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(sub_actions) == 1, "对等快照假定恰好一个 subparsers 组"
    sub_action = sub_actions[0]

    global_options = [
        serialize_action(a)
        for a in parser._actions
        if not isinstance(a, argparse._SubParsersAction)
    ]
    help_map = _choice_help_map(sub_action)

    commands: dict[str, Any] = {}
    for name in sub_action.choices:  # type: ignore[attr-defined]
        subparser = sub_action.choices[name]
        commands[name] = {
            "help": help_map.get(name),
            "description": subparser.description,
            "options": [serialize_action(a) for a in subparser._actions],
        }

    return {
        "global": {"options": global_options},
        "command_order": list(sub_action.choices),
        "commands": commands,
    }
