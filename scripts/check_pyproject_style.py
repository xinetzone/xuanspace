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
6. scikit-build-core 版本下限 >= 0.11（PEP 639 的 license 字符串 +
   project.license-files 自 0.11.0 起才被接受），且
   [tool.scikit-build].minimum-version 必须与之下限一致或写
   "build-system.requires" 交由构建系统派生

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

# PEP 639 元数据（license 字符串 / project.license-files）在
# scikit-build-core 0.11.0 才被接受，0.9/0.10 会直接拒绝 SPDX 字符串
SCIKIT_MIN_VERSION = (0, 11)
SCIKIT_MIN_VERSION_TEXT = "0.11"
# minimum-version 允许用该哨兵把下限交给 build-system.requires 派生
MINIMUM_VERSION_FROM_REQUIRES = "build-system.requires"


def _version_tuple(text: str) -> tuple[int, ...] | None:
    """把 '0.11.0' / 'v0.11' 解析成可比较的数字元组；无法解析返回 None。"""
    match = re.fullmatch(r"[vV]?([0-9]+(?:\.[0-9]+)*)", text.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _normalized(version: tuple[int, ...], width: int) -> tuple[int, ...]:
    """补齐位数后比较，避免 (0, 11) 与 (0, 11, 0) 被判不等。"""
    return version + (0,) * max(0, width - len(version))


def _lower_bound(req: str) -> tuple[int, ...] | None:
    """取 requirement 的版本下限；无下限或无法解析返回 None。"""
    match = re.search(r"(?:>=|==|~=)\s*([0-9][0-9.]*)", req)
    if not match:
        return None
    return _version_tuple(match.group(1))


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


def _is_nested_submodule(root: Path, project_dir: Path) -> bool:
    """子模块判定：项目目录落在 .gitmodules 声明的路径之内。

    submodule 是独立仓库（vendor/*、libs/tvm-book、libs/mystx），
    有各自的构建标准与 CI，不受本仓 canonical 风格约束。
    不能只用「目录内有 .git」判定：libs/caffe-ffi、libs/demo-ffi 是
    主仓直接追踪的普通目录，但残留了各自的 .git，会被误判为子模块。
    """
    try:
        rel = project_dir.relative_to(root)
    except ValueError:
        return False
    if rel == Path("."):
        return False
    for sub in _submodule_paths(root):
        parts = Path(sub).parts
        if rel.parts[: len(parts)] == parts:
            return True
    return False


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
                    issues.append(f"纯 Python 包（wheel.cmake=false）不需要 {req}，应从 [build-system].requires 移除")
            if (path.parent / "CMakeLists.txt").exists():
                issues.append("纯 Python 包（wheel.cmake=false）不得携带 CMakeLists.txt，应删除")
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

    # 5. scikit-build-core 版本下限契约（PEP 639 元数据需要 >= 0.11.0）
    if isinstance(requires, list):
        for req in requires:
            if _req_name(req) != "scikit-build-core":
                continue
            bound = _lower_bound(req)
            if bound is None:
                issues.append(
                    f"[build-system].requires 的 {req!r} 未声明版本下限，"
                    f"应写为 scikit-build-core>={SCIKIT_MIN_VERSION_TEXT}"
                )
            elif _normalized(bound, 2) < _normalized(SCIKIT_MIN_VERSION, 2):
                issues.append(
                    f"[build-system].requires 的 {req!r} 下限低于 {SCIKIT_MIN_VERSION_TEXT}（PEP 639 元数据所需）"
                )
            break

    minimum_version = data.get("tool", {}).get("scikit-build", {}).get("minimum-version")
    if minimum_version is None:
        issues.append(
            f"[tool.scikit-build] 缺少 minimum-version，"
            f'应为 "{SCIKIT_MIN_VERSION_TEXT}" 或 "{MINIMUM_VERSION_FROM_REQUIRES}"'
        )
    elif minimum_version != MINIMUM_VERSION_FROM_REQUIRES:
        parsed = _version_tuple(str(minimum_version))
        if parsed is None or _normalized(parsed, 2) < _normalized(SCIKIT_MIN_VERSION, 2):
            issues.append(
                f"[tool.scikit-build].minimum-version 为 {minimum_version!r}，"
                f'应 >= "{SCIKIT_MIN_VERSION_TEXT}" 或 "{MINIMUM_VERSION_FROM_REQUIRES}"'
            )

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
