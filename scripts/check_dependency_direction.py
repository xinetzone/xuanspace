#!/usr/bin/env python3
"""Xuanspace 分区依赖方向门禁（《项目蓝图》目标态特征 3「方向可断言」落地）。

用途：把 libs/README.md「依赖规则」一节声明的分区依赖规则从「只写在文档里」
      变为 CI 可断言的检查——扫描 libs/ 下所有 Python 源码的 import 语句，
      断言其不依赖被禁止的分区（apps/、scripts/、attic/、vendor/）。

规则来源（单一事实来源）：本文件 FORBIDDEN_IMPORTS 常量，与 libs/README.md
      「依赖规则」一节同源。四层方向（道 → 法 → 术 → 器）要求「上层可被下层
      引用、下层不得反向依赖上层」，分区依赖规则是该方向的显式化：
        - libs/（器）可以依赖其它 libs/ 与 PyPI 第三方包；
        - libs/（器）不应依赖 apps/（同层，防止循环依赖）；
        - libs/（器）谨慎依赖 tools/（术，主要面向开发环境）——本门禁不拦截；
        - libs/（器）禁止依赖 scripts/、attic/、vendor/。

检查方法：用 ast 解析每个 .py 文件，取绝对导入的顶层模块名，命中禁止集合即报
      「文件:行号:顶层模块」。相对导入（from . import x）视为 lib 内部引用，
      不检查。.gitmodules 声明的子模块（libs/mystx、libs/tvm-book）与构建产物
      /缓存目录不在扫描范围内。

用法：python scripts/check_dependency_direction.py
退出码：0 全部通过；1 存在违规依赖（或源码不可解析）；2 脚本自身错误。

幂等性：只读检查，无副作用，重复执行结果一致。
"""

import ast
import re
import sys
from pathlib import Path

# --- 规则定义（单一事实来源，与 libs/README.md「依赖规则」同源）---
# 分区 -> 该分区源码禁止导入的顶层模块名（仓库一级目录名）。
# 仅登记具备「禁止」语义的分区；「谨慎依赖」的 tools/ 不纳入拦截。
FORBIDDEN_IMPORTS: dict[str, tuple[str, ...]] = {
    "libs": ("apps", "scripts", "attic", "vendor"),
}

# 扫描时跳过的目录名（构建产物、缓存、虚拟环境与仓库内部管理目录）
IGNORED_DIR_NAMES = {
    ".agents",
    ".git",
    ".github",
    ".meta",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".temp",
    ".trae",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def _submodule_paths(root: Path) -> list[str]:
    """读取 .gitmodules 声明的子模块路径；无该文件返回空表。"""
    gitmodules = root / ".gitmodules"
    if not gitmodules.exists():
        return []
    paths: list[str] = []
    for line in gitmodules.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*path\s*=\s*(.+?)\s*$", line)
        if match:
            paths.append(match.group(1))
    return paths


def _is_nested_submodule(root: Path, target: Path) -> bool:
    """子模块判定：目标路径落在 .gitmodules 声明的路径之内。

    submodule 是独立仓库（libs/mystx、libs/tvm-book），有各自的构建标准与 CI，
    不受本仓分区依赖规则约束。
    """
    try:
        rel = target.relative_to(root)
    except ValueError:
        return False
    if rel == Path("."):
        return False
    for sub in _submodule_paths(root):
        parts = Path(sub).parts
        if rel.parts[: len(parts)] == parts:
            return True
    return False


def iter_py_files(root: Path, partition: str):
    """产出指定分区内应扫描的 Python 文件路径（排序稳定）。"""
    base = root / partition
    if not base.exists():
        return
    for path in sorted(base.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORED_DIR_NAMES for part in rel_parts):
            continue
        if _is_nested_submodule(root, path.parent):
            continue
        yield path


def check_file(path: Path, root: Path, forbidden: frozenset[str]) -> list[str]:
    """校验单个 Python 文件，返回违规依赖信息列表（空则通过）。"""
    rel = path.relative_to(root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{rel}: 无法解析（{exc.__class__.__name__}: {exc}）"]

    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # level > 0 为相对导入（lib 内部引用），不检查
            imported = [node.module] if node.level == 0 and node.module else []
        else:
            continue
        for name in imported:
            top = name.split(".", 1)[0]
            if top in forbidden:
                issues.append(f"{rel}:{node.lineno}: 禁止导入 {top!r}（来自 {name!r}）")
    return issues


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    partitions = sorted(FORBIDDEN_IMPORTS)

    files = [(partition, path) for partition in partitions for path in iter_py_files(root, partition)]
    if not files:
        print("未扫描到任何 Python 源码，请检查仓库结构。")
        return 2

    issues: list[str] = []
    for partition, path in files:
        issues.extend(check_file(path, root, frozenset(FORBIDDEN_IMPORTS[partition])))

    print(f"扫描到 {len(files)} 个 Python 文件（分区：{', '.join(partitions)}）")
    if issues:
        print(f"\n发现 {len(issues)} 处违规依赖：")
        for issue in sorted(issues):
            print(f"  - {issue}")
        return 1

    rules = "；".join(f"{p} 未依赖 {', '.join(FORBIDDEN_IMPORTS[p])}" for p in partitions)
    print(f"全部通过依赖方向校验：{rules}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
