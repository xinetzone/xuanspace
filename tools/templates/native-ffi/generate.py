"""从 native-ffi 模板生成新项目脚手架。

用法: python tools/templates/native-ffi/generate.py <package_name> <module_name> <target_dir>

占位符替换:
  {{package_name}}       -> Python包名 (snake_case, 如 demo_ffi)
  {{module_name}}        -> C++模块名/FFI前缀 (如 demo)
  {{package_name|upper}} -> 大写包名,用于CMake选项 (如 DEMO_FFI)
  {{name}}               -> 项目显示名 (kebab-case, 如 demo-ffi)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).parent.resolve()

REPLACEMENTS = {}


def init_replacements(package_name: str, module_name: str) -> None:
    items = [
        ("{{package_name|upper}}", package_name.upper()),
        ("{{module_name|upper}}", module_name.upper()),
        ("{{package_name}}", package_name),
        ("{{module_name}}", module_name),
        ("{{name}}", package_name.replace("_", "-")),
    ]
    items.sort(key=lambda x: len(x[0]), reverse=True)
    REPLACEMENTS.clear()
    REPLACEMENTS.update(items)


def replace_in_text(text: str) -> str:
    for placeholder, value in REPLACEMENTS.items():
        text = text.replace(placeholder, value)
    return text


def replace_in_path(name: str) -> str:
    return replace_in_text(name)


def copy_and_render(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {"generate.py", "__pycache__", "*.pyc"}
    for item in src_dir.iterdir():
        if item.name == "generate.py" or item.name == "__pycache__":
            continue
        rel_name = replace_in_path(item.name)
        dst_path = dst_dir / rel_name
        if item.is_dir():
            copy_and_render(item, dst_path)
        else:
            content = item.read_text(encoding="utf-8")
            rendered = replace_in_text(content)
            dst_path.write_text(rendered, encoding="utf-8", newline="\n")
            print(f"  created: {dst_path}")


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <package_name> <module_name> <target_dir>")
        sys.exit(1)

    package_name = sys.argv[1]
    module_name = sys.argv[2]
    target_dir = Path(sys.argv[3]).resolve()

    init_replacements(package_name, module_name)

    if target_dir.exists():
        print(f"Error: target directory already exists: {target_dir}")
        sys.exit(1)

    print(f"Generating project from template:")
    print(f"  package_name: {package_name}")
    print(f"  module_name:  {module_name}")
    print(f"  target:       {target_dir}")
    print()

    copy_and_render(TEMPLATE_ROOT, target_dir)
    print()
    print(f"Done. Project generated at {target_dir}")


if __name__ == "__main__":
    main()
