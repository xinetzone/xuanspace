#!/usr/bin/env python3
"""Xuanspace「声明—实现一致性」门禁校验脚本（《优化建议书》P0-4 落地）。

用途：把仓库中**可机械提取的文档声明**与**磁盘/代码事实**逐项比对，
      不一致时以非零退出码使 CI 失败，防止文档声明与实现静默漂移。

覆盖三类声明：
  A. CLI 命令清单：README.md「N 个子命令」与 doc/intro.md 命令表
                   ↔ tools/xs/src/xs/cli.py 的注册表
  B. 子项目清单  ：libs/README.md「现有子项目」表 ↔ libs/ 磁盘目录
  C. 目录名口径  ：README.md「目录结构」块声明的目录 ↔ 磁盘存在性，
                   且顶层不得出现 docs/（文档目录约定为 doc/）

使用方法：
  python scripts/check_declaration_consistency.py            # 校验模式（默认）
  python scripts/check_declaration_consistency.py --generate # 生成模式：刷新清单快照
  python scripts/check_declaration_consistency.py --json     # 以 JSON 输出结果

退出码：0 全部一致；1 存在不一致；2 脚本自身错误（声明源缺失或结构不可解析）。

幂等性：--generate 仅在内容变化时写盘，重复执行无副作用。

共享库查阅结论：SpecWeave 主仓 `.agents/scripts/lib/` 已提供 Markdown 表格解析
（如 `knowledge_graph_core.parse_markdown_table`、`pattern_maturity.readme_ops.
parse_readme_index_table`）与检查器框架，但其按「主仓 scripts 相对导入」设计
（`from .xxx` / `from constants import ...`），且**不在 xuanspace 仓库内**
（单仓 CI 检出时不可导入）。故本脚本按 scripts/README.md「不导入项目内部包」
规范自包含实现，仅镜像共享库的表格解析思路。
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CLI_MODULE = "tools/xs/src/xs/cli.py"
README_FILE = "README.md"
INTRO_FILE = "doc/intro.md"
LIBS_README = "libs/README.md"
LIBS_DIR = "libs"
SNAPSHOT_FILE = "scripts/declaration-snapshot.json"

# 顶层目录命名约定：文档目录为 doc/，不得为 docs/
CANONICAL_DOC_DIR = "doc"
FORBIDDEN_TOP_DIRS = ("docs",)

# 目录扫描时忽略的系统目录
IGNORED_DIR_NAMES = {"__pycache__", ".git", "node_modules"}

# --- 声明提取正则 ---
# cli.py：命令组 `app.add_typer(x, name="...")` 与直接命令 `app.command("...")`
_CLI_GROUP_RE = re.compile(r'app\.add_typer\(\s*[\w.]+,\s*name="([^"]+)"')
_CLI_COMMAND_RE = re.compile(r'app\.command\("([^"]+)"\)')
# README.md 项目索引中的「（N 个子命令）」
_README_CMD_COUNT_RE = re.compile(r"（\s*(\d+)\s*个子命令\s*）")
# doc/intro.md 命令表中的 `xs <cmd>` 行
_INTRO_CMD_ROW_RE = re.compile(r"^\|\s*`xs ([a-z0-9\-]+)`\s*\|", re.MULTILINE)
# libs/README.md「现有子项目」表中的 `dir/` 行
_LIB_ROW_RE = re.compile(r"^\|\s*`([a-z0-9\-]+)/`", re.MULTILINE)
# README.md「目录结构」代码块中的 `├── name/` 行
_STRUCT_DIR_RE = re.compile(r"[├└]──\s*([A-Za-z0-9_.\-]+)/")


def _read_text(root: Path, rel: str) -> str:
    """读取仓库内声明源文件，缺失时抛 FileNotFoundError。"""
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"缺少声明源文件：{rel}")
    return path.read_text(encoding="utf-8")


def _section(text: str, header: str) -> str:
    """截取 header 起始到下一个二级标题之间的正文。"""
    idx = text.find(header)
    if idx == -1:
        raise ValueError(f"未找到章节：{header}")
    rest = text[idx + len(header) :]
    nxt = re.search(r"^##\s", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _first_fenced_block(text: str, header: str) -> str:
    """截取 header 之后第一个围栏代码块的正文。"""
    idx = text.find(header)
    if idx == -1:
        raise ValueError(f"未找到章节：{header}")
    start = text.find("```", idx)
    if start == -1:
        raise ValueError(f"{header} 之后未找到代码块")
    end = text.find("```", start + 3)
    if end == -1:
        raise ValueError(f"{header} 的代码块未闭合")
    return text[start + 3 : end]


def extract_cli_commands(root: Path) -> list[str]:
    """从 cli.py 注册表提取一级命令名（命令组 + 直接命令，去重排序）。"""
    text = _read_text(root, CLI_MODULE)
    names = set(_CLI_GROUP_RE.findall(text)) | set(_CLI_COMMAND_RE.findall(text))
    if not names:
        raise ValueError(f"{CLI_MODULE} 未解析到任何命令注册，正则或文件结构已变化")
    return sorted(names)


def extract_lib_projects(root: Path) -> list[str]:
    """从 libs/ 磁盘目录提取子项目名（忽略隐藏项与系统目录，去重排序）。"""
    lib = root / LIBS_DIR
    if not lib.is_dir():
        raise FileNotFoundError(f"缺少目录：{LIBS_DIR}")
    names = [
        p.name for p in lib.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in IGNORED_DIR_NAMES
    ]
    return sorted(names)


def declared_cli_count(root: Path) -> int | None:
    """README.md 声明的「（N 个子命令）」命令数；未声明返回 None。"""
    match = _README_CMD_COUNT_RE.search(_read_text(root, README_FILE))
    return int(match.group(1)) if match else None


def declared_intro_commands(root: Path) -> list[str]:
    """doc/intro.md 命令表中 `xs <cmd>` 行的命令名集合。"""
    return sorted(set(_INTRO_CMD_ROW_RE.findall(_read_text(root, INTRO_FILE))))


def declared_lib_projects(root: Path) -> list[str]:
    """libs/README.md「现有子项目」表中声明的目录名集合。"""
    body = _section(_read_text(root, LIBS_README), "## 现有子项目")
    return sorted(set(_LIB_ROW_RE.findall(body)))


def declared_structure_dirs(root: Path) -> list[str]:
    """README.md「目录结构」代码块中声明的顶层目录名集合。"""
    block = _first_fenced_block(_read_text(root, README_FILE), "## 目录结构")
    return sorted(set(_STRUCT_DIR_RE.findall(block)))


def collect_manifest(root: Path) -> dict[str, list[str]]:
    """提取可机械校验的事实清单（生成模式的产出）。"""
    return {
        "cli_commands": extract_cli_commands(root),
        "lib_projects": extract_lib_projects(root),
    }


def _diff_failure(label: str, declared: set[str], facts: set[str]) -> str | None:
    """构造集合差异描述；无差异返回 None。"""
    if declared == facts:
        return None
    missing = sorted(facts - declared)
    extra = sorted(declared - facts)
    return f"{label} 与事实不一致（声明缺：{missing}；声明多：{extra}）"


def run_checks(root: Path) -> tuple[list[str], list[str]]:
    """执行三类声明校验与快照比对，返回 (failures, notes)。"""
    failures: list[str] = []
    notes: list[str] = []

    manifest = collect_manifest(root)
    cli_commands = set(manifest["cli_commands"])
    lib_projects = set(manifest["lib_projects"])

    # A. CLI 命令清单
    count = declared_cli_count(root)
    if count is None:
        failures.append(f"{README_FILE} 未找到「N 个子命令」声明")
    elif count != len(cli_commands):
        failures.append(f"{README_FILE} 声明 {count} 个子命令，{CLI_MODULE} 实际注册 {len(cli_commands)} 个")

    intro_declared = set(declared_intro_commands(root))
    diff = _diff_failure(f"{INTRO_FILE} 命令表", intro_declared, cli_commands)
    if diff:
        failures.append(diff)

    # B. 子项目清单
    diff = _diff_failure(f"{LIBS_README}「现有子项目」", set(declared_lib_projects(root)), lib_projects)
    if diff:
        failures.append(diff)

    # C. 目录名口径
    absent = [d for d in declared_structure_dirs(root) if not (root / d).is_dir()]
    if absent:
        failures.append(f"{README_FILE}「目录结构」声明的目录在磁盘不存在：{absent}")
    for bad in FORBIDDEN_TOP_DIRS:
        if (root / bad).is_dir():
            failures.append(f"顶层存在 {bad}/，文档目录命名约定为 {CANONICAL_DOC_DIR}/")
    if not (root / CANONICAL_DOC_DIR).is_dir():
        failures.append(f"顶层缺少文档目录 {CANONICAL_DOC_DIR}/")

    # 快照比对：磁盘事实 ↔ 已提交的清单快照
    snap_path = root / SNAPSHOT_FILE
    if not snap_path.exists():
        notes.append(f"未找到清单快照 {SNAPSHOT_FILE}，请先运行 --generate 生成并提交")
    else:
        snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
        for key, facts in (("cli_commands", manifest["cli_commands"]), ("lib_projects", manifest["lib_projects"])):
            if snapshot.get(key) != facts:
                failures.append(f"清单快照 {SNAPSHOT_FILE} 的 {key} 已过期（磁盘事实已变），请运行 --generate 刷新")

    return failures, notes


def generate(root: Path) -> int:
    """生成模式：把事实清单写入快照文件（幂等）。"""
    manifest = collect_manifest(root)
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = root / SNAPSHOT_FILE
    if path.exists() and path.read_text(encoding="utf-8") == content:
        print(f"清单快照已是目标状态，跳过：{SNAPSHOT_FILE}")
        return 0
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"已刷新清单快照：{SNAPSHOT_FILE}")
    print(content, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Xuanspace「声明—实现一致性」门禁校验（P0-4）",
    )
    parser.add_argument("--generate", action="store_true", help="生成模式：刷新可提取清单快照")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出校验结果")
    args = parser.parse_args()

    try:
        if args.generate:
            return generate(REPO_ROOT)

        failures, notes = run_checks(REPO_ROOT)
        if args.json:
            print(
                json.dumps(
                    {"ok": not failures, "failures": failures, "notes": notes},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1 if failures else 0

        for note in notes:
            print(f"[NOTE] {note}")
        if failures:
            print(f"声明一致性校验未通过（{len(failures)} 项）：")
            for item in failures:
                print(f"  - {item}")
            return 1
        print("声明与实现一致：CLI 命令清单 / 子项目清单 / 目录名口径 全部通过。")
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
