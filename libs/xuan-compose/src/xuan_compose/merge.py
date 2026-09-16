# SPDX-License-Identifier: GPL-2.0-only
"""compose 字典递归合并与 YAML 加载。

迁移自上游 1.6.0：

* ``OverrideTag`` / ``ResetTag``（第 1764-1808 行，YAML 标签 ``!override``/``!reset``）
* ``clone`` / ``rec_merge_one`` / ``rec_merge``（第 2221-2317 行）
* ``load_yaml_or_die``（第 2320-2326 行）

``resolve_extends``（上游第 2329-2365 行）在分层归属上属于规范化流程，
T11 下沉到 :mod:`xuan_compose.normalize`——它内部要调用 ``rec_subs``/
``normalize_service``（同模块）与本模块的 ``rec_merge``/``load_yaml_or_die``，
下沉后规范层内部依赖变为单向（normalize → merge），消除原先的惰性导入环。

差异登记（T10 汇总）：``OverrideTag``/``ResetTag`` 仍继承 ``yaml.YAMLObject``，
因此 import 本模块时会向上游一样把两个标签注册进全局 ``yaml.SafeLoader``/
``SafeDumper``——``load_yaml_or_die`` 使用 ``yaml.safe_load``，compose 文档的
``!override``/``!reset`` 语义依赖该全局注册；``import xuan_compose`` 包根本身
不导入本模块，故包根导入仍无副作用。
"""

import sys
from typing import Any

import yaml

from .compat import is_list
from .logging_utils import log

__all__ = [
    "OverrideTag",
    "ResetTag",
    "clone",
    "rec_merge_one",
    "rec_merge",
    "load_yaml_or_die",
]


class OverrideTag(yaml.YAMLObject):
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!override"

    def __init__(self, value: Any) -> None:
        self.value: dict[Any, Any] | list[Any]
        if len(value) > 0 and isinstance(value[0], tuple):
            self.value = {}
            # item is a tuple representing service's lower level key and value
            for item in value:
                # value can actually be a list, then all the elements from the list have to be
                # collected
                if isinstance(item[1].value, list):
                    self.value[item[0].value] = [i.value for i in item[1].value]
                else:
                    self.value[item[0].value] = item[1].value
        else:
            self.value = [item.value for item in value]

    @classmethod
    def from_yaml(cls, loader: Any, node: Any) -> OverrideTag:
        return OverrideTag(node.value)

    @classmethod
    def to_yaml(cls, dumper: Any, data: OverrideTag) -> str:
        return dumper.represent_scalar(cls.yaml_tag, data.value)


class ResetTag(yaml.YAMLObject):
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!reset"

    @classmethod
    def to_json(cls) -> str:
        return cls.yaml_tag

    @classmethod
    def from_yaml(cls, loader: Any, node: Any) -> ResetTag:
        return ResetTag()

    @classmethod
    def to_yaml(cls, dumper: Any, data: ResetTag) -> str:
        return dumper.represent_scalar(cls.yaml_tag, "")


def clone(value: Any) -> Any:
    return value.copy() if is_list(value) or isinstance(value, dict) else value


def rec_merge_one(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """
    update target from source recursively
    """
    done = set()
    remove = set()

    for key, value in source.items():
        if key in target:
            continue
        target[key] = clone(value)
        done.add(key)
    for key, value in target.items():
        if key in done:
            continue
        if key not in source:
            if isinstance(value, ResetTag):
                log.info("Unneeded !reset found for [%s]", key)
                remove.add(key)

            if isinstance(value, OverrideTag):
                log.info("Unneeded !override found for [%s] with value '%s'", key, value)
                target[key] = clone(value.value)

            continue

        value2 = source[key]

        if isinstance(value, ResetTag) or isinstance(value2, ResetTag):
            remove.add(key)
            continue

        if isinstance(value, OverrideTag) or isinstance(value2, OverrideTag):
            target[key] = (
                clone(value.value) if isinstance(value, OverrideTag) else clone(value2.value)
            )
            continue

        if key in ("command", "entrypoint"):
            target[key] = clone(value2)
            continue

        # We can merge dicts into an empty tag. E.g.:
        # vol_1:
        # and
        # vol_1:
        #   driver: "abcdef"
        if value is None and isinstance(value2, dict):
            target[key] = value = {}

        # normalizing inputs to dicts
        if key == "depends_on":
            if is_list(value) and isinstance(value2, dict):
                value = {x: {} for x in value}
                target[key] = value
            elif isinstance(value, dict) and is_list(value2):
                value2 = {x: {} for x in value2}

        if not isinstance(value2, type(value)):
            value_type = type(value)
            value2_type = type(value2)
            raise ValueError(f"can't merge value of [{key}] of type {value_type} and {value2_type}")

        if is_list(value2):
            if key == "volumes":
                # clean duplicate mount targets
                pts = {v.split(":", 2)[1] for v in value2 if ":" in v}
                del_ls = [
                    ix for (ix, v) in enumerate(value) if ":" in v and v.split(":", 2)[1] in pts
                ]
                for ix in reversed(del_ls):
                    del value[ix]
                value.extend(value2)
            else:
                value.extend(value2)
        elif isinstance(value2, dict):
            rec_merge_one(value, value2)
        else:
            target[key] = value2

    for key in remove:
        del target[key]

    return target


def rec_merge(target: dict[str, Any], *sources: dict[str, Any]) -> dict[str, Any]:
    """
    update target recursively from sources
    """
    for source in sources:
        rec_merge_one(target, source)
    return target


def load_yaml_or_die(file_path: str, stream: Any) -> dict[str, Any]:
    try:
        return yaml.safe_load(stream)
    except yaml.scanner.ScannerError as e:
        log.fatal("Compose file contains an error:\n%s", e)
        log.info("Compose file %s contains an error:", file_path, exc_info=e)
        sys.exit(1)
