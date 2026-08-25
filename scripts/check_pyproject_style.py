#!/usr/bin/env python3
"""玄境 canonical pyproject.toml 风格一致性校验脚本。

遍历根目录、libs/*、tools/* 及 tools/templates/** 下的 pyproject.toml，
检查：
1. TOML 可解析（tomllib）
2. build-backend == "scikit_build_core.build"（统一后端）
3. [build-system].requires 含 scikit-build-core 与 ninja
4. 无 setuptools / tool.setuptools 残留（作为唯一构建后端）

用法：python scripts/check_pyproject_style.py
退出码：0 全部通过；1 存在不合规项；2 脚本自身错误。

Python 版本要求：3.11+（使用标准库 tomllib）。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

# 需要扫描 pyproject.toml 的目录（相对仓库根）
SCAN_DIRS = [".", "libs", "tools"]

# 额外递归扫描的目录（含下级嵌套，如 tools/templates/**）
SCAN_DIRS_RECURSIVE = ["tools"]

# 明确排除的路径（嵌套仓库/模板占位除外，模板也需校验但允许占位）
EXCLUDE_DIRS = {
    ".git",
    "vendor",
    "build",
    "dist",
    "__pycache__",
    "conda.recipe",
    "attic",
    ".agents",
}

REQUIRED_REQUIRES = {"scikit-build-core", "ninja"}
DENIED_MARKERS = {
    "tool.setuptools",
    "tool.pdm.build",
}
DENIED_BACKENDS = {"setuptools.build_meta", "setuptools.build_meta:__legacy__"}


def iter_pyproject(root: Path):
    """产出仓库内所有应校验的 pyproject.toml 路径。"""
    seen: set[Path] = set()
    for scan_dir in SCAN_DIRS:
        base = root / scan_dir
        if not base.exists():
            continue
        if scan_dir == ".":
            seen.add(base / "pyproject.toml")
        else:
            # libs/*、tools/* 下的第一层子目录内的 pyproject.toml
            for child in sorted(base.iterdir()):
                if child.is_dir() and child.name not in EXCLUDE_DIRS:
                    seen.add(child / "pyproject.toml")

    # 递归扫描嵌套目录（如 tools/templates/**，模板也需校验）
    for scan_dir in SCAN_DIRS_RECURSIVE:
        base = root / scan_dir
        if not base.exists():
            continue
        for child in sorted(base.rglob("pyproject.toml")):
            if not any(part in EXCLUDE_DIRS for part in child.relative_to(root).parts):
                seen.add(child)

    # 仅保留真实存在的文件，排序保证输出稳定
    for path in sorted(seen):
        if path.exists():
            yield path


def check_one(path: Path) -> list[str]:
    """校验单个 pyproject.toml，返回违规信息列表（空则通过）。"""
    issues: list[str] = []
    try:
        with path.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        return [f"TOML 解析失败: {exc}"]

    build_system = data.get("build-system", {})
    backend = build_system.get("build-backend", "")
    requires = build_system.get("requires", [])

    # 1. 构建后端统一
    if backend != "scikit_build_core.build":
        issues.append(f"build-backend 应为 scikit_build_core.build，实际为 {backend!r}")

    # 2. requires 含 scikit-build-core 与 ninja
    if isinstance(requires, list):
        req_names = {r.split(">=")[0].split("==")[0].split("<")[0] for r in requires}
        for req in REQUIRED_REQUIRES:
            if req not in req_names:
                issues.append(f"[build-system].requires 缺少 {req}")
    else:
        issues.append("缺少 [build-system].requires")

    # 3. 无 setuptools 残留
    if backend in DENIED_BACKENDS:
        issues.append("build-backend 为 setuptools，应迁移到 scikit-build-core")
    if "setuptools" in requires:
        issues.append("[build-system].requires 含 setuptools，应移除")

    # 4. 无 tool.setuptools / tool.pdm.build 残留
    for marker in DENIED_MARKERS:
        if marker in data.get("tool", {}):
            issues.append(f"存在 [{marker}] 配置，应迁移到 [tool.scikit-build]")

    return issues


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    all_issues: dict[str, list[str]] = {}
    files = list(iter_pyproject(root))

    if not files:
        print("未扫描到任何 pyproject.toml，请检查仓库结构。")
        return 2

    for path in files:
        rel = path.relative_to(root).as_posix()
        issues = check_one(path)
        if issues:
            all_issues[rel] = issues

    # 输出
    print(f"扫描到 {len(files)} 个 pyproject.toml")
    if all_issues:
        print(f"\n发现 {len(all_issues)} 个文件不合规：")
        for rel, issues in sorted(all_issues.items()):
            print(f"\n[{rel}]")
            for issue in issues:
                print(f"  - {issue}")
        return 1

    print("全部 pyproject.toml 通过 canonical 风格校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())