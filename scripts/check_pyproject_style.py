#!/usr/bin/env python3
"""玄境 canonical pyproject.toml 风格一致性校验脚本。

遍历根目录、libs/*、tools/* 及 tools/templates/** 下的 pyproject.toml，
检查：
1. TOML 可解析（tomllib）
2. build-backend == "scikit_build_core.build"（统一后端）
3. [build-system].requires 含 scikit-build-core
4. 原生项目（未声明 wheel.cmake=false）必须含 ninja；
   纯 Python 项目（[tool.scikit-build.wheel] cmake=false）禁止声明
   cmake/ninja 构建需求，且同目录不得携带 CMakeLists.txt
5. 无 setuptools / tool.setuptools 残留（作为唯一构建后端）

用法：python scripts/check_pyproject_style.py
退出码：0 全部通过；1 存在不合规项；2 脚本自身错误。

Python 版本要求：3.11+（使用标准库 tomllib）。
"""

from __future__ import annotations

import re
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

REQUIRED_REQUIRES = {"scikit-build-core"}
NATIVE_REQUIRED_REQUIRES = {"ninja"}
PURE_PYTHON_DENIED_REQUIRES = {"cmake", "ninja"}
DENIED_MARKERS = {
    "tool.setuptools",
    "tool.pdm.build",
}
DENIED_BACKENDS = {"setuptools.build_meta", "setuptools.build_meta:__legacy__"}


def _req_name(req: str) -> str:
    """从 PEP 508 requirement 串取裸包名（剥离版本标记与环境标记）。"""
    return re.split(r"[<>=!~;\[\s]", req, maxsplit=1)[0].strip()


def is_pure_python(data: dict[str, Any]) -> bool:
    """纯 Python 包判定：[tool.scikit-build.wheel] cmake 显式为 false。"""
    wheel = data.get("tool", {}).get("scikit-build", {}).get("wheel", {})
    return isinstance(wheel, dict) and wheel.get("cmake") is False


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
        if path.exists() and not _is_nested_submodule(root, path.parent):
            yield path


def _is_nested_submodule(root: Path, project_dir: Path) -> bool:
    """嵌套 git submodule（gitlink）判定：非根目录且内含 .git 文件或目录。

    submodule 是独立仓库（如 vendor/*、libs/tvm-book、libs/mystx），
    有各自的构建标准与 CI，不受本仓 canonical 风格约束；
    根目录自身的 .git 是本仓标志，不算嵌套 submodule。
    """
    try:
        project_dir.relative_to(root)
    except ValueError:
        return False
    return project_dir != root and (project_dir / ".git").exists()


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

    # 2. requires 规则：scikit-build-core 始终必需；
    #    ninja 仅原生项目必需；纯 Python 项目禁止 cmake/ninja
    if isinstance(requires, list):
        req_names = {_req_name(r) for r in requires}
        for req in REQUIRED_REQUIRES:
            if req not in req_names:
                issues.append(f"[build-system].requires 缺少 {req}")
        if is_pure_python(data):
            for req in PURE_PYTHON_DENIED_REQUIRES:
                if req in req_names:
                    issues.append(
                        f"纯 Python 包（wheel.cmake=false）不需要 {req}，"
                        f"应从 [build-system].requires 移除"
                    )
            if (path.parent / "CMakeLists.txt").exists():
                issues.append(
                    "纯 Python 包（wheel.cmake=false）不得携带 CMakeLists.txt，应删除"
                )
        else:
            for req in NATIVE_REQUIRED_REQUIRES:
                if req not in req_names:
                    issues.append(f"[build-system].requires 缺少 {req}（原生项目必需）")
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