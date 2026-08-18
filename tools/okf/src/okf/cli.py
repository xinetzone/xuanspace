"""okf CLI 命令。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .conformance import format_report
from .harness import Harness


def _harness_for(path: Path) -> Harness:
    """为给定 Bundle 路径创建 Harness 并加载默认插件。"""
    harness = Harness(bundle_path=path)
    harness._load_default_plugins()
    return harness


def _get_service(harness: Harness, name: str):
    """获取服务实现；服务未注册时输出友好错误并返回 None。"""
    try:
        return harness.ctx.get(name)
    except KeyError:
        print(f"错误: 服务 '{name}' 未注册（插件加载失败或 Bundle 无效）", file=sys.stderr)
        return None


def _cmd_validate(args) -> int:
    harness = _harness_for(Path(args.path))
    report = _get_service(harness, "conformance_report")
    if report is None:
        return 1
    print(format_report(report))
    return 0 if not report.errors else 1


def _cmd_init(args) -> int:
    root = Path(args.path)
    subdirs = ["concepts", "playbooks", "references"]

    for name in subdirs:
        d = root / name
        if d.exists():
            print(f"跳过（已存在）: {d}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {d}")

    index_path = root / "index.md"
    if index_path.exists():
        print(f"跳过（已存在）: {index_path}")
    else:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(f"# {root.name}\n", encoding="utf-8")
        print(f"创建文件: {index_path}")

    log_path = root / "log.md"
    if log_path.exists():
        print(f"跳过（已存在）: {log_path}")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("# Change Log\n", encoding="utf-8")
        print(f"创建文件: {log_path}")

    return 0


def _cmd_index(args) -> int:
    path = Path(args.path)
    harness = _harness_for(path)
    generator = _get_service(harness, "index_generator")
    if generator is None:
        return 1
    content = generator()
    target = path / "index.md"
    target.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0


def _cmd_inspect(args) -> int:
    harness = _harness_for(Path(args.path))
    bundle = _get_service(harness, "bundle_accessor")
    if bundle is None:
        return 1

    if args.concept_id:
        concept = bundle.concepts.get(args.concept_id)
        if concept is None:
            print(f"概念不存在: {args.concept_id}", file=sys.stderr)
            return 1
        print(f"ID: {args.concept_id}")
        print(f"Type: {concept.type}")
        print(f"Title: {concept.title}")
        print(f"Description: {concept.description}")
        tags = ", ".join(concept.tags) if concept.tags else "(无)"
        print(f"Tags: {tags}")
        print("Frontmatter:")
        for key, value in concept.frontmatter.items():
            print(f"  {key}: {value}")
        print("Body（前 5 行）:")
        for line in concept.body.splitlines()[:5]:
            print(f"  {line}")
        return 0

    print(f"Bundle: {bundle.root}")
    print(f"概念数量: {len(bundle.concepts)}")
    for cid, concept in bundle.concepts.items():
        print(f"- {cid}  [{concept.type}]  {concept.title}")
    return 0


def _cmd_trust(args) -> int:
    harness = _harness_for(Path(args.path))
    analyzer = _get_service(harness, "trust_analyzer")
    if analyzer is None:
        return 1

    result = analyzer(args.concept_id)
    if result is None:
        print(f"概念不存在: {args.concept_id}", file=sys.stderr)
        return 1

    def _fmt(info: dict) -> str:
        tier = info.get("trust_tier")
        stale = "已过期" if info.get("is_stale") else "保鲜中"
        return f"信任等级={tier}, 保鲜状态={stale}"

    if args.concept_id:
        print(f"{args.concept_id}: {_fmt(result)}")
    else:
        for cid, info in result.items():
            print(f"{cid}: {_fmt(info)}")
    return 0


def _cmd_list(args) -> int:
    harness = _harness_for(Path(args.path))
    bundle = _get_service(harness, "bundle_accessor")
    if bundle is None:
        return 1

    for cid, concept in bundle.concepts.items():
        if args.type_filter is not None and concept.type != args.type_filter:
            continue
        if args.tag_filter is not None and args.tag_filter not in concept.tags:
            continue
        print(f"{cid}\t{concept.type}\t{concept.title}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="okf",
        description="OKF v0.2 工具链",
    )
    parser.add_argument("--version", "-V", action="version", version=f"okf {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # validate
    p_validate = subparsers.add_parser("validate", help="一致性校验")
    p_validate.add_argument("path", help="Bundle 路径")
    p_validate.add_argument("--strict", action="store_true", help="严格模式")
    p_validate.set_defaults(func=_cmd_validate)

    # init
    p_init = subparsers.add_parser("init", help="创建 Bundle 骨架")
    p_init.add_argument("path", help="Bundle 路径")
    p_init.set_defaults(func=_cmd_init)

    # index
    p_index = subparsers.add_parser("index", help="生成 index.md")
    p_index.add_argument("path", help="Bundle 路径")
    p_index.set_defaults(func=_cmd_index)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="查看概念详情")
    p_inspect.add_argument("path", help="Bundle 路径")
    p_inspect.add_argument("concept_id", nargs="?", default=None, help="概念 ID（可选）")
    p_inspect.set_defaults(func=_cmd_inspect)

    # trust
    p_trust = subparsers.add_parser("trust", help="信任等级与保鲜状态")
    p_trust.add_argument("path", help="Bundle 路径")
    p_trust.add_argument("concept_id", nargs="?", default=None, help="概念 ID（可选）")
    p_trust.set_defaults(func=_cmd_trust)

    # list
    p_list = subparsers.add_parser("list", help="列出概念")
    p_list.add_argument("path", help="Bundle 路径")
    p_list.add_argument("--type", dest="type_filter", default=None, help="按 type 过滤")
    p_list.add_argument("--tag", dest="tag_filter", default=None, help="按 tag 过滤")
    p_list.set_defaults(func=_cmd_list)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    sys.exit(args.func(args))
