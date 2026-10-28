#!/usr/bin/env python3
"""Xuanspace「声明—实现一致性」门禁校验脚本（《优化建议书》P0-4 落地）。

用途：把仓库中**可机械提取的文档声明**与**磁盘/代码事实**逐项比对，
      不一致时以非零退出码使 CI 失败，防止文档声明与实现静默漂移。

覆盖六类声明：
  A. CLI 命令清单：README.md「N 个子命令」与 doc/intro.md 命令表
                   ↔ tools/xs/src/xs/cli.py 的注册表
  B. 子项目清单  ：libs/README.md「现有子项目」生成块 ↔ libs/ 磁盘目录（P1-4①：
                   块由本脚本从磁盘派生并注入，门禁比对「块 == 脚本渲染结果」）
  C. 目录名口径  ：README.md「目录结构」块声明的目录 ↔ 磁盘存在性，
                   且顶层不得出现 docs/（文档目录约定为 doc/）
  D. vendor 清单 ：vendor/README.md「现有子项目」表 ↔ .gitmodules 中 vendor/ 子模块
  E. 子模块声明  ：CONTRIBUTING.md「N 个上游子模块」计数与子模块表 ↔ .gitmodules
  F. 项目索引    ：README.md「项目索引」表 ↔ tools/ + libs/ 磁盘目录 + vendor/ 子模块

使用方法：
  python scripts/check_declaration_consistency.py            # 校验模式（默认）
  python scripts/check_declaration_consistency.py --generate # 生成模式：刷新清单快照与清单生成块
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
CONTRIBUTING_FILE = "CONTRIBUTING.md"
VENDOR_README = "vendor/README.md"
VENDOR_DIR = "vendor"
TOOLS_DIR = "tools"
GITMODULES_FILE = ".gitmodules"
SNAPSHOT_FILE = "scripts/declaration-snapshot.json"

# 顶层目录命名约定：文档目录为 doc/，不得为 docs/
CANONICAL_DOC_DIR = "doc"
FORBIDDEN_TOP_DIRS = ("docs",)

# 目录扫描时忽略的系统目录
IGNORED_DIR_NAMES = {"__pycache__", ".git", "node_modules"}

# --- 生成块标记与子项目元数据（P1-4①） ---
# libs/README.md「现有子项目」表由本脚本从 libs/ 磁盘目录派生，块内内容勿手改。
LIBS_GENERATED_BEGIN = "<!-- BEGIN GENERATED: libs-projects -->"
LIBS_GENERATED_END = "<!-- END GENERATED: libs-projects -->"
# 子项目元数据：目录名 -> (类型, 说明)；新增 libs/ 子目录须先在此登记，否则门禁报错。
LIB_PROJECT_META: dict[str, tuple[str, str]] = {
    "caffe-ffi": ("原生扩展", "Caffe 深度学习框架 FFI 绑定"),
    "demo-ffi": ("原生扩展", "FFI 扩展开发模板（Stub/Real 双运行时）"),
    "mystx": ("Python（子模块）", "MyST 文档工具链/主题"),
    "npu-ffi": ("原生扩展", "VTA NPU 加速器 FFI 绑定"),
    "tvm-book": ("Python（子模块）", "TVM 中文手册 + flexloopy"),
    "xuan-compose": ("纯 Python", "Compose 规范声明式容器编排引擎（podman-compose 重构）"),
    "xuan-core": ("纯 Python", "核心工具库，提供基础类型、通用函数、错误处理"),
    "xuan-ext-demo": ("原生扩展", "C/C++ 扩展示例库"),
}

# --- 声明提取正则 ---
# cli.py：命令组 `app.add_typer(x, name="...")` 与直接命令 `app.command("...")`
_CLI_GROUP_RE = re.compile(r'app\.add_typer\(\s*[\w.]+,\s*name="([^"]+)"')
_CLI_COMMAND_RE = re.compile(r'app\.command\("([^"]+)"\)')
# README.md 项目索引中的「（N 个子命令）」
_README_CMD_COUNT_RE = re.compile(r"（\s*(\d+)\s*个子命令\s*）")
# doc/intro.md 命令表中的 `xs <cmd>` 行
_INTRO_CMD_ROW_RE = re.compile(r"^\|\s*`xs ([a-z0-9\-]+)`\s*\|", re.MULTILINE)
# README.md「目录结构」代码块中的 `├── name/` 行
_STRUCT_DIR_RE = re.compile(r"[├└]──\s*([A-Za-z0-9_.\-]+)/")
# .gitmodules 中的 `path = <子模块路径>` 行
_GITMODULES_PATH_RE = re.compile(r"^\s*path\s*=\s*(.+?)\s*$", re.MULTILINE)
# CONTRIBUTING.md 的「包含 N 个上游子模块」
_SUBMODULE_COUNT_RE = re.compile(r"包含\s*(\d+)\s*个上游子模块")
# 章节标题（用于把表格解析限定在单一章节内）
_SECTION_BREAK_RE = re.compile(r"^##\s", re.MULTILINE)


def _read_text(root: Path, rel: str) -> str:
    """读取仓库内声明源文件，缺失时抛 FileNotFoundError。"""
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"缺少声明源文件：{rel}")
    return path.read_text(encoding="utf-8")


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


def _generated_block(text: str, begin: str, end: str) -> str:
    """截取生成标记包裹的整段（含标记行）；缺失或顺序错误时抛 ValueError。"""
    start = text.find(begin)
    stop = text.find(end)
    if start == -1 or stop == -1 or stop < start:
        raise ValueError(f"未找到成对的生成标记 {begin} / {end}")
    return text[start : stop + len(end)]


def _section(text: str, header: str) -> str:
    """截取 header 起始到下一个二级标题之前的正文；未找到 header 时抛 ValueError。"""
    idx = text.find(header)
    if idx == -1:
        raise ValueError(f"未找到章节：{header}")
    rest = text[idx + len(header) :]
    match = _SECTION_BREAK_RE.search(rest)
    return rest[: match.start()] if match else rest


def _table_first_column(body: str) -> list[str]:
    """解析 body 中第一张 Markdown 表格的首列数据（去表头与分隔行，去重排序）。

    仅解析从首个 `|` 行开始、到首个非表格行为止的**连续**表格，
    避免误吞同章节内的后续表格。
    """
    lines: list[str] = []
    started = False
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            started = True
            lines.append(line)
        elif started:
            break
    if len(lines) < 2:
        return []
    cells: list[str] = []
    for line in lines[1:]:  # 跳过表头行
        parts = [c.strip() for c in line.strip("|").split("|")]
        first = parts[0] if parts else ""
        if not first or set(first) <= set("-: "):
            continue  # 分隔行
        cells.append(first.strip("`").strip().rstrip("/"))
    return sorted(set(cells))


def extract_cli_commands(root: Path) -> list[str]:
    """从 cli.py 注册表提取一级命令名（命令组 + 直接命令，去重排序）。"""
    text = _read_text(root, CLI_MODULE)
    names = set(_CLI_GROUP_RE.findall(text)) | set(_CLI_COMMAND_RE.findall(text))
    if not names:
        raise ValueError(f"{CLI_MODULE} 未解析到任何命令注册，正则或文件结构已变化")
    return sorted(names)


def extract_dir_names(root: Path, rel: str) -> list[str]:
    """提取 rel 目录下的直接子目录名（忽略隐藏项与系统目录，去重排序）。"""
    base = root / rel
    if not base.is_dir():
        raise FileNotFoundError(f"缺少目录：{rel}")
    names = [
        p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in IGNORED_DIR_NAMES
    ]
    return sorted(names)


def extract_lib_projects(root: Path) -> list[str]:
    """从 libs/ 磁盘目录提取子项目名（忽略隐藏项与系统目录，去重排序）。"""
    return extract_dir_names(root, LIBS_DIR)


def extract_submodules(root: Path) -> list[str]:
    """从 .gitmodules 提取子模块路径（排序）。"""
    paths = _GITMODULES_PATH_RE.findall(_read_text(root, GITMODULES_FILE))
    if not paths:
        raise ValueError(f"{GITMODULES_FILE} 未解析到任何子模块 path，文件结构已变化")
    return sorted(paths)


def extract_vendor_submodules(root: Path) -> list[str]:
    """从 .gitmodules 提取 vendor/ 下的子模块名（如 `vendor/caffe` -> `caffe`，排序）。"""
    prefix = f"{VENDOR_DIR}/"
    return sorted(p[len(prefix) :] for p in extract_submodules(root) if p.startswith(prefix))


def declared_vendor_projects(root: Path) -> list[str]:
    """vendor/README.md「现有子项目」表首列的目录名。"""
    return _table_first_column(_section(_read_text(root, VENDOR_README), "## 现有子项目"))


def declared_contributing_submodules(root: Path) -> list[str]:
    """CONTRIBUTING.md 子模块表首列的路径集合。"""
    return _table_first_column(_section(_read_text(root, CONTRIBUTING_FILE), "### 3.5 顶层子模块延迟初始化策略"))


def declared_submodule_count(root: Path) -> int | None:
    """CONTRIBUTING.md 声明的「N 个上游子模块」计数；未声明返回 None。"""
    match = _SUBMODULE_COUNT_RE.search(_read_text(root, CONTRIBUTING_FILE))
    return int(match.group(1)) if match else None


def declared_index_projects(root: Path) -> list[str]:
    """README.md「项目索引」表首列的名称集合。"""
    return _table_first_column(_section(_read_text(root, README_FILE), "## 项目索引"))


def expected_index_projects(root: Path) -> list[str]:
    """「项目索引」期望名称：tools/ + libs/ 直接子目录 ∪ vendor/ 子模块（去重排序）。"""
    names = extract_dir_names(root, TOOLS_DIR) + extract_lib_projects(root) + extract_vendor_submodules(root)
    return sorted(set(names))


def render_lib_projects_table(root: Path) -> tuple[str, list[str]]:
    """渲染 libs/README.md「现有子项目」生成块（含标记行）。

    返回 (markdown, 未在 LIB_PROJECT_META 中登记的目录名)。
    """
    names = extract_lib_projects(root)
    missing = [n for n in names if n not in LIB_PROJECT_META]
    lines = [
        LIBS_GENERATED_BEGIN,
        "",
        "| 目录名 | 类型 | 说明 |",
        "|--------|------|------|",
    ]
    for name in names:
        kind, desc = LIB_PROJECT_META.get(name, ("（待补）", "（待补说明）"))
        lines.append(f"| `{name}/` | {kind} | {desc} |")
    lines += ["", LIBS_GENERATED_END]
    return "\n".join(lines), missing


def declared_cli_count(root: Path) -> int | None:
    """README.md 声明的「（N 个子命令）」命令数；未声明返回 None。"""
    match = _README_CMD_COUNT_RE.search(_read_text(root, README_FILE))
    return int(match.group(1)) if match else None


def declared_intro_commands(root: Path) -> list[str]:
    """doc/intro.md 命令表中 `xs <cmd>` 行的命令名集合。"""
    return sorted(set(_INTRO_CMD_ROW_RE.findall(_read_text(root, INTRO_FILE))))


def declared_structure_dirs(root: Path) -> list[str]:
    """README.md「目录结构」代码块中声明的顶层目录名集合。"""
    block = _first_fenced_block(_read_text(root, README_FILE), "## 目录结构")
    return sorted(set(_STRUCT_DIR_RE.findall(block)))


def collect_manifest(root: Path) -> dict[str, list[str]]:
    """提取可机械校验的事实清单（生成模式的产出）。"""
    return {
        "cli_commands": extract_cli_commands(root),
        "lib_projects": extract_lib_projects(root),
        "submodules": extract_submodules(root),
    }


def _diff_failure(label: str, declared: set[str], facts: set[str]) -> str | None:
    """构造集合差异描述；无差异返回 None。"""
    if declared == facts:
        return None
    missing = sorted(facts - declared)
    extra = sorted(declared - facts)
    return f"{label} 与事实不一致（声明缺：{missing}；声明多：{extra}）"


def run_checks(root: Path) -> tuple[list[str], list[str]]:
    """执行六类声明校验与快照比对，返回 (failures, notes)。"""
    failures: list[str] = []
    notes: list[str] = []

    manifest = collect_manifest(root)
    cli_commands = set(manifest["cli_commands"])

    # A. CLI 命令清单
    cmd_count = declared_cli_count(root)
    if cmd_count is None:
        failures.append(f"{README_FILE} 未找到「N 个子命令」声明")
    elif cmd_count != len(cli_commands):
        failures.append(f"{README_FILE} 声明 {cmd_count} 个子命令，{CLI_MODULE} 实际注册 {len(cli_commands)} 个")

    intro_declared = set(declared_intro_commands(root))
    diff = _diff_failure(f"{INTRO_FILE} 命令表", intro_declared, cli_commands)
    if diff:
        failures.append(diff)

    # B. 子项目清单（生成块 ↔ 磁盘事实，P1-4①）
    try:
        block = _generated_block(_read_text(root, LIBS_README), LIBS_GENERATED_BEGIN, LIBS_GENERATED_END)
    except ValueError as exc:
        failures.append(str(exc))
    else:
        rendered, missing = render_lib_projects_table(root)
        if missing:
            failures.append(f"{LIBS_README} 生成块缺元数据登记：{missing}（请在 LIB_PROJECT_META 补充类型/说明）")
        if block != rendered:
            failures.append(f"{LIBS_README}「现有子项目」生成块与磁盘事实不一致（已过期），请运行 --generate 刷新")

    # C. 目录名口径
    absent = [d for d in declared_structure_dirs(root) if not (root / d).is_dir()]
    if absent:
        failures.append(f"{README_FILE}「目录结构」声明的目录在磁盘不存在：{absent}")
    for bad in FORBIDDEN_TOP_DIRS:
        if (root / bad).is_dir():
            failures.append(f"顶层存在 {bad}/，文档目录命名约定为 {CANONICAL_DOC_DIR}/")
    if not (root / CANONICAL_DOC_DIR).is_dir():
        failures.append(f"顶层缺少文档目录 {CANONICAL_DOC_DIR}/")

    # D. vendor/README.md「现有子项目」表 ↔ .gitmodules 中 vendor/ 子模块
    vendor_declared = set(declared_vendor_projects(root))
    vendor_facts = set(extract_vendor_submodules(root))
    if not vendor_declared:
        failures.append(f"{VENDOR_README}「现有子项目」表未解析到任何数据行")
    diff = _diff_failure(f"{VENDOR_README}「现有子项目」表", vendor_declared, vendor_facts)
    if diff:
        failures.append(diff)

    # E. CONTRIBUTING.md 子模块计数与表 ↔ .gitmodules
    submodules = set(manifest["submodules"])
    sub_count = declared_submodule_count(root)
    if sub_count is None:
        failures.append(f"{CONTRIBUTING_FILE} 未找到「N 个上游子模块」声明")
    elif sub_count != len(submodules):
        failures.append(
            f"{CONTRIBUTING_FILE} 声明 {sub_count} 个上游子模块，{GITMODULES_FILE} 实际登记 {len(submodules)} 个"
        )
    diff = _diff_failure(f"{CONTRIBUTING_FILE} 子模块表", set(declared_contributing_submodules(root)), submodules)
    if diff:
        failures.append(diff)

    # F. README.md「项目索引」表 ↔ tools/ + libs/ 磁盘目录 + vendor/ 子模块
    diff = _diff_failure(
        f"{README_FILE}「项目索引」表", set(declared_index_projects(root)), set(expected_index_projects(root))
    )
    if diff:
        failures.append(diff)

    # 快照比对：磁盘事实 ↔ 已提交的清单快照
    snap_path = root / SNAPSHOT_FILE
    if not snap_path.exists():
        notes.append(f"未找到清单快照 {SNAPSHOT_FILE}，请先运行 --generate 生成并提交")
    else:
        snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
        for key, facts in manifest.items():
            if snapshot.get(key) != facts:
                failures.append(f"清单快照 {SNAPSHOT_FILE} 的 {key} 已过期（磁盘事实已变），请运行 --generate 刷新")

    return failures, notes


def inject_lib_projects_block(root: Path) -> bool:
    """把 libs/README.md 的生成块重写为磁盘派生的表（幂等）；返回是否发生写入。"""
    path = root / LIBS_README
    text = path.read_text(encoding="utf-8")
    start = text.find(LIBS_GENERATED_BEGIN)
    stop = text.find(LIBS_GENERATED_END)
    if start == -1 or stop == -1 or stop < start:
        raise ValueError(f"{LIBS_README} 未找到成对的生成标记 {LIBS_GENERATED_BEGIN} / {LIBS_GENERATED_END}")
    rendered, _ = render_lib_projects_table(root)
    updated = text[:start] + rendered + text[stop + len(LIBS_GENERATED_END) :]
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def generate(root: Path) -> int:
    """生成模式：刷新清单快照与 libs/README.md 生成块（幂等）。"""
    manifest = collect_manifest(root)
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = root / SNAPSHOT_FILE
    if path.exists() and path.read_text(encoding="utf-8") == content:
        print(f"清单快照已是目标状态，跳过：{SNAPSHOT_FILE}")
    else:
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"已刷新清单快照：{SNAPSHOT_FILE}")

    _, missing = render_lib_projects_table(root)
    if missing:
        print(f"[ERROR] 以下目录未在 LIB_PROJECT_META 中登记：{missing}", file=sys.stderr)
        return 2
    if inject_lib_projects_block(root):
        print(f"已刷新生成块：{LIBS_README}")
    else:
        print(f"生成块已是目标状态，跳过：{LIBS_README}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Xuanspace「声明—实现一致性」门禁校验（P0-4 / P1-4① / 六类声明）",
    )
    parser.add_argument("--generate", action="store_true", help="生成模式：刷新可提取清单快照与 libs 清单生成块")
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
        print("声明与实现一致：CLI 命令清单 / 子项目清单 / 目录名口径 / vendor 清单 / 子模块声明 / 项目索引 全部通过。")
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
