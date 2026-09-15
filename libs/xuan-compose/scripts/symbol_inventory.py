# SPDX-License-Identifier: GPL-2.0-only
"""T10 TR-10.1 证据脚本：统计上游公开符号数并对拍新库同名落点。

用法（子项目根 libs/xuan-compose/）：

    python scripts/symbol_inventory.py

输出：
1. 上游 ``vendor/podman-compose/podman_compose.py`` 顶层公开符号
   （公开函数/类 + 公开常量，distinct 名称）的总数与逐条行号；
2. 每个符号在新库 src/xuan_compose/ 下的同名落点模块；
3. 无同名落点的符号清单（应为重命名/删除项，见 docs/parity.md）。

只读：脚本不修改 vendor 与本库任何文件。vendor 路径可用环境变量
``PODMAN_COMPOSE_SRC`` 覆盖。
"""

import ast
import os
import pathlib

THIS = pathlib.Path(__file__).resolve()
LIB_ROOT = THIS.parents[1]
# scripts/ → xuan-compose → libs → xuanspace → projects → SpecWeave
DEFAULT_VENDOR = THIS.parents[5] / "vendor" / "podman-compose"


def _upstream() -> pathlib.Path:
    path = pathlib.Path(
        os.environ.get("PODMAN_COMPOSE_SRC", DEFAULT_VENDOR)
    ) / "podman_compose.py"
    if not path.is_file():
        raise SystemExit(f"找不到上游源码: {path}（可用 PODMAN_COMPOSE_SRC 覆盖）")
    return path


def top_level_public(tree: ast.AST) -> tuple[dict[str, tuple[str, int]], dict[str, int]]:
    defs: dict[str, tuple[str, int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            kind = (
                "async"
                if isinstance(node, ast.AsyncFunctionDef)
                else "class"
                if isinstance(node, ast.ClassDef)
                else "def"
            )
            defs.setdefault(node.name, (kind, node.lineno))
    consts: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and (
                    target.id.isupper() or target.id.startswith("__")
                ):
                    consts.setdefault(target.id, node.lineno)
    return defs, consts


def new_definitions(root: pathlib.Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.setdefault(node.name, str(py.relative_to(root)))
    return found


def main() -> None:
    upstream = _upstream()
    defs, consts = top_level_public(ast.parse(upstream.read_text(encoding="utf-8")))
    targets = new_definitions(LIB_ROOT / "src" / "xuan_compose")
    total = len(defs) + len(consts)
    print(f"上游公开符号总数（distinct）: {total} = 函数/类 {len(defs)} + 常量 {len(consts)}")
    missing = []
    for name, (_, lineno) in sorted(defs.items(), key=lambda kv: kv[1][1]):
        loc = targets.get(name)
        if not loc:
            missing.append(f"{name} (L{lineno})")
    for name, lineno in sorted(consts.items()):
        if name not in targets and name not in {"PODMAN_CMDS", "COMPOSE_DEFAULT_LS",
                                               "STOP_GRACE_PERIOD", "__version__"}:
            missing.append(f"{name} (L{lineno})")
    print(f"无同名落点（须在 docs/parity.md 登记重命名/删除）: {len(missing)}")
    for item in missing:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
