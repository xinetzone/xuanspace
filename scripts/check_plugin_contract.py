#!/usr/bin/env python3
"""Xuanspace 插件契约 H 类断言门禁（《设计·05 方案》§5.7 落地）。

用途：把设计文档 §5.7 的 H 类不变量从「只写在文档里」变为 CI 可断言的检查。
      全部为只读、可复现断言，数据源仅两处：各 ``pyproject.toml`` 的
      ``[tool.xuan.plugin]`` 清单，与仓库根的 ``xuan.plugins.lock``（若存在）。

断言：
      H1 可解性：所有必需 inject 在候选集合内有提供者。判定复用内核
                 ``build_graph``（与求解器同口径），缺失依赖列出「引用者 + 键」。
      H2 隔离性：同键多提供者按 ADR-05 裁决——全 ``single`` 多提供者，或含
                 ``multi`` 但 ``priority`` 并列，均判冲突。裁决复用内核
                 ``find_conflicts``，消息列出冲突键与双方 id、mode、priority。
      H3 可撤销性：插件入口模块不得持有模块级可变全局（可变字面量，或
                 dict/list/set/defaultdict 等可变容器构造），函数内不得用
                 ``global`` 裸写全局表；违规给出「文件:行号」。
      H4 视图一致性：``xuan.plugins.lock`` 存在时，须等于由当前清单重算的规范
                 JSON（``schema``=1，``plugins`` 按 id 排序，``digest`` 为清单契约
                 的 sha256）；不一致时逐 id 给出期望/实际差异。lock 不存在时
                 跳过并在输出中提示（lock 为派生产物，可用 ``--write-lock`` 生成）。
      H5 结构兼容：``inject.constraint`` 的 ``methods:<m1,m2>`` 中每个方法须存在
                 于提供者入口模块（函数/类/模块级名字）；列出消费者的缺失方法。
                 其他约束语法无法静态校验，判失败并提示支持的语法。

依赖说明：本脚本导入 ``libs/xuan-plugin`` 的清单 API 作为判定单一事实来源
      （CI quality job 在此步骤前已安装该包）；入口模块解析与 AST 扫描为本地实现。

用法：python scripts/check_plugin_contract.py [--root PATH]
      python scripts/check_plugin_contract.py --write-lock   # 生成模式：写入 lock
退出码：0 全部通过；1 存在契约违规；2 仓库结构异常（未扫描到任何插件清单）。

幂等性：检查模式只读；``--write-lock`` 仅在内容变化时写盘，重复执行无副作用。
"""

import argparse
import ast
import json
import re
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from xuan_plugin.lock import build_lock, lock_diff, render_lock
from xuan_plugin.manifest import (
    ManifestError,
    PluginManifest,
    build_graph,
    find_conflicts,
    manifest_from_pyproject,
    merge_manifests,
)

# 扫描时跳过的目录名（版本库管理目录、构建产物、缓存与虚拟环境）
IGNORED_DIR_NAMES = {
    ".agents",
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".temp",
    ".trae",
    ".venv",
    "__pycache__",
    "attic",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}

LOCK_FILENAME = "xuan.plugins.lock"
ENTRY_POINT_GROUP = "xuan.plugins"
CONSTRAINT_PREFIX = "methods:"

# 模块级赋值若来自这些可变容器构造，视为「模块级可变单例」
MUTABLE_CONSTRUCTORS = frozenset(
    {"ChainMap", "Counter", "OrderedDict", "WeakValueDictionary", "defaultdict", "dict", "list", "set"}
)
MUTABLE_LITERAL_NODES = (ast.Dict, ast.DictComp, ast.List, ast.ListComp, ast.Set, ast.SetComp)


@dataclass(frozen=True)
class DeclaredPlugin:
    """一份清单及其来源：pyproject 路径与声明的入口模块。"""

    manifest: PluginManifest
    pyproject: Path
    entry_modules: tuple[str, ...]


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
    """子模块是独立仓库（各自有 CI 与构建标准），不受本仓契约约束。"""
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


def iter_pyprojects(root: Path) -> list[Path]:
    """列出仓库内应参与契约断言的 pyproject.toml（排序稳定）。"""
    found: list[Path] = []
    for path in sorted(root.rglob("pyproject.toml")):
        rel_parts = path.relative_to(root).parts[:-1]
        if any(part in IGNORED_DIR_NAMES for part in rel_parts):
            continue
        if _is_nested_submodule(root, path.parent):
            continue
        found.append(path)
    return found


def _entry_modules(data: dict) -> tuple[str, ...]:
    """取 ``[project.entry-points."xuan.plugins"]`` 的模块名（去掉 ``:attr`` 后缀）。"""
    project = data.get("project")
    if not isinstance(project, dict):
        return ()
    groups = project.get("entry-points")
    if not isinstance(groups, dict):
        return ()
    entries = groups.get(ENTRY_POINT_GROUP)
    if not isinstance(entries, dict):
        return ()
    modules: set[str] = set()
    for value in entries.values():
        if isinstance(value, str):
            module = value.split(":", 1)[0].strip()
            if module:
                modules.add(module)
    return tuple(sorted(modules))


def load_plugins(root: Path) -> tuple[list[DeclaredPlugin], list[str]]:
    """加载全部插件声明，返回（插件列表, 解析问题列表）。"""
    plugins: list[DeclaredPlugin] = []
    issues: list[str] = []
    for path in iter_pyprojects(root):
        rel = path.relative_to(root).as_posix()
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            manifest = manifest_from_pyproject(data, source=rel)
        except (UnicodeDecodeError, tomllib.TOMLDecodeError, ManifestError) as exc:
            issues.append(f"清单解析失败：{rel}（{exc}）")
            continue
        if manifest is None:
            continue
        plugins.append(DeclaredPlugin(manifest=manifest, pyproject=path, entry_modules=_entry_modules(data)))
    return plugins, issues


def _by_id(plugins: Iterable[DeclaredPlugin]) -> dict[str, DeclaredPlugin]:
    """id → 首份声明（同 id 不同契约由 merge_manifests 另行报错）。"""
    result: dict[str, DeclaredPlugin] = {}
    for plugin in plugins:
        result.setdefault(plugin.manifest.id, plugin)
    return result


def provider_index(manifests: Iterable[PluginManifest]) -> dict[str, list[tuple[str, str, int]]]:
    """key → [(id, mode, priority)]，与内核 ``build_graph`` 同口径：

    跳过 disabled，按 ``(-priority, id)`` 稳定定序。
    """
    index: dict[str, list[tuple[str, str, int]]] = {}
    for manifest in manifests:
        if manifest.disabled:
            continue
        for decl in manifest.provide:
            index.setdefault(decl.key, []).append((manifest.id, decl.mode, manifest.priority))
    for providers in index.values():
        providers.sort(key=lambda item: (-item[2], item[0]))
    return index


def check_h1(manifests: Sequence[PluginManifest]) -> list[str]:
    """H1：必需依赖必须可满足。"""
    graph = build_graph(manifests)
    return [
        f"H1 可解性：{consumer} 的必需依赖 {key} 没有任何提供者（候选插件 {len(graph.nodes)} 个）"
        for consumer, key in graph.unresolved
    ]


def check_h2(manifests: Sequence[PluginManifest]) -> list[str]:
    """H2：同键多提供者必须能裁决出唯一胜出者。"""
    index = provider_index(manifests)
    issues: list[str] = []
    for key, ids in find_conflicts(index):
        providers = {plugin_id: (mode, priority) for plugin_id, mode, priority in index[key]}
        detail = "、".join(
            f"{plugin_id}(mode={providers[plugin_id][0]}, priority={providers[plugin_id][1]})" for plugin_id in ids
        )
        if all(providers[plugin_id][0] == "single" for plugin_id in ids):
            issues.append(f"H2 隔离性：能力键 {key} 有 {len(ids)} 个 single 提供者，违反两两不交：{detail}")
        else:
            issues.append(f"H2 隔离性：能力键 {key} 的 multi 提供者 priority 并列，无显式 provider 可裁决：{detail}")
    return issues


def _resolve_module(package: Path, module: str) -> Path | None:
    """在包目录下解析模块文件（兼容 ``src/`` 布局与扁平布局）。"""
    rel = Path(*module.split("."))
    for base in (package / "src", package):
        for candidate in (base / f"{rel}.py", base / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def _mutable_kind(value: ast.expr) -> str | None:
    """判定模块级赋值是否产生可变对象；返回可读原因，不可变返回 None。"""
    if isinstance(value, MUTABLE_LITERAL_NODES):
        return "可变字面量"
    if isinstance(value, ast.Call):
        func = value.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            return None
        if name in MUTABLE_CONSTRUCTORS:
            return f"{name}() 可变容器"
    return None


def _mutable_globals(tree: ast.Module) -> list[tuple[int, str, str]]:
    """模块顶层的可变全局赋值：[(行号, 变量名, 原因)]。"""
    found: list[tuple[int, str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value, targets = node.value, [node.target]
        else:
            continue
        kind = _mutable_kind(value)
        if kind is None:
            continue
        for target in targets:
            # 双下划线名字（如 __all__）是模块导出清单，不是插件运行期状态
            if isinstance(target, ast.Name) and not (target.id.startswith("__") and target.id.endswith("__")):
                found.append((node.lineno, target.id, kind))
    return found


def check_h3(root: Path, plugins: Sequence[DeclaredPlugin]) -> list[str]:
    """H3：入口模块不得持有模块级可变状态，登记须由注册表句柄承载。"""
    issues: list[str] = []
    scanned: set[Path] = set()
    for plugin in plugins:
        package = plugin.pyproject.parent
        for module in plugin.entry_modules:
            path = _resolve_module(package, module)
            if path is None or path in scanned:
                continue
            scanned.add(path)
            rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError) as exc:
                issues.append(f"H3 可撤销性：{rel} 无法解析（{exc.__class__.__name__}），无法断言无裸写全局表")
                continue
            for lineno, name, kind in _mutable_globals(tree):
                issues.append(
                    f"H3 可撤销性：{rel}:{lineno} 模块级可变全局 {name} = {kind}，插件状态须由登记返回的句柄承载"
                )
            for node in ast.walk(tree):
                if isinstance(node, ast.Global):
                    issues.append(
                        f"H3 可撤销性：{rel}:{node.lineno} 函数内 global {', '.join(node.names)} 裸写全局表，无法按句柄撤销"
                    )
    return issues


def check_h4(root: Path, manifests: Sequence[PluginManifest]) -> tuple[list[str], str]:
    """H4：lock 必须可由当前清单重算得到。返回（问题列表, 状态描述）。"""
    lock_path = root / LOCK_FILENAME
    if not lock_path.is_file():
        return [], f"H4 视图一致性：未发现 {LOCK_FILENAME}，跳过（可用 --write-lock 生成）"
    try:
        actual = json.loads(lock_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"H4 视图一致性：{LOCK_FILENAME} 不可解析（{exc}）"], ""
    expected = build_lock(manifests)
    if actual == expected:
        return [], f"H4 视图一致性：{LOCK_FILENAME} 与清单重算一致（{len(expected['plugins'])} 个插件）"
    return [f"H4 视图一致性：{LOCK_FILENAME} 与清单重算不一致：{lock_diff(expected, actual)}"], ""


def write_lock(root: Path, manifests: Sequence[PluginManifest]) -> int:
    """生成模式：把当前清单的规范 lock 写入仓库根（幂等）。"""
    path = root / LOCK_FILENAME
    content = render_lock(manifests)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        print(f"{LOCK_FILENAME} 已是目标状态，跳过（{len(manifests)} 个插件）")
        return 0
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"已写入 {LOCK_FILENAME}：{len(manifests)} 个插件")
    return 0


def _defined_names(path: Path) -> set[str]:
    """模块内可用于满足 ``methods:`` 约束的名字集合（函数、类、模块级变量）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def check_h5(plugins: Sequence[DeclaredPlugin]) -> list[str]:
    """H5：``methods:<m1,m2>`` 约束须被提供者入口模块满足。"""
    manifests = [plugin.manifest for plugin in plugins]
    index = provider_index(manifests)
    conflicted = {key for key, _ in find_conflicts(index)}
    by_id = _by_id(plugins)

    issues: list[str] = []
    for plugin in plugins:
        consumer = plugin.manifest.id
        for decl in plugin.manifest.inject:
            constraint = decl.constraint.strip()
            if not constraint:
                continue
            if not constraint.startswith(CONSTRAINT_PREFIX):
                issues.append(
                    f"H5 结构兼容：{consumer} 对 {decl.key} 的约束 {constraint!r} 无法静态校验（仅支持 methods:<m1,m2>）"
                )
                continue
            providers = index.get(decl.key, [])
            if not providers or decl.key in conflicted:
                continue  # 缺失/冲突已由 H1、H2 报告
            provider_id = providers[0][0]
            provider = by_id.get(provider_id)
            if provider is None or not provider.entry_modules:
                issues.append(
                    f"H5 结构兼容：{consumer} 对 {decl.key} 的约束无法核对——提供者 {provider_id} 未声明 {ENTRY_POINT_GROUP} 入口模块"
                )
                continue
            path = _resolve_module(provider.pyproject.parent, provider.entry_modules[0])
            if path is None:
                issues.append(
                    f"H5 结构兼容：提供者 {provider_id} 的入口模块 {provider.entry_modules[0]!r} 在磁盘上不存在"
                )
                continue
            try:
                defined = _defined_names(path)
            except (SyntaxError, UnicodeDecodeError) as exc:
                issues.append(f"H5 结构兼容：提供者 {provider_id} 的入口模块无法解析（{exc.__class__.__name__}）")
                continue
            expected = [name.strip() for name in constraint[len(CONSTRAINT_PREFIX) :].split(",") if name.strip()]
            missing = [name for name in expected if name not in defined]
            if missing:
                issues.append(
                    f"H5 结构兼容：{consumer} 对 {decl.key} 的约束要求方法 {', '.join(missing)}，"
                    f"提供者 {provider_id} 未定义（缺失方法：{', '.join(missing)}）"
                )
    return issues


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="插件契约 H 类断言（H1–H5）与 lock 生成")
    parser.add_argument("--root", type=Path, default=None, help="仓库根目录（默认按脚本位置推断）")
    parser.add_argument("--write-lock", action="store_true", help="生成模式：把当前清单的规范 lock 写入仓库根")
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parent.parent).resolve()

    plugins, issues = load_plugins(root)
    if not plugins:
        print(f"未扫描到任何 [tool.xuan.plugin] 清单（{root}），请检查仓库结构。")
        for issue in issues:
            print(f"  - {issue}")
        return 2

    manifests = [plugin.manifest for plugin in plugins]
    try:
        merged = merge_manifests(manifests)
    except ManifestError as exc:
        issues.append(f"清单合并失败：{exc}")
        merged = tuple(sorted({manifest.id: manifest for manifest in manifests}.values(), key=lambda m: m.id))

    if args.write_lock:
        return write_lock(root, merged)

    print(f"扫描到 {len(merged)} 份插件清单：{', '.join(manifest.id for manifest in merged)}")

    h4_issues, h4_status = check_h4(root, merged)
    if h4_status:
        print(h4_status)

    issues.extend(check_h1(merged))
    issues.extend(check_h2(merged))
    issues.extend(check_h3(root, plugins))
    issues.extend(h4_issues)
    issues.extend(check_h5(plugins))

    if issues:
        print(f"\n发现 {len(issues)} 处插件契约违规：")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("全部通过 H1–H5 插件契约断言。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
