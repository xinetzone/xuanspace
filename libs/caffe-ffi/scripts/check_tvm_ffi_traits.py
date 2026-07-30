#!/usr/bin/env python3
"""tvm-ffi TypeTraits 冲突预检脚本

基于 Phase 1 复盘洞察 I1（第三方依赖类型系统"勿重复实现已有功能"原则），
在 Phase 2 COW 实施前自动检测 tvm-ffi 已有 TypeTraits 特化，防止重复定义导致 SFINAE 冲突。

用法:
    python scripts/check_tvm_ffi_traits.py [--tvm-ffi-dir /path/to/tvm-ffi/include]
    python scripts/check_tvm_ffi_traits.py --json  # JSON 格式输出

退出码:
    0 - 所有检查通过，无冲突风险
    1 - 发现潜在冲突，需要人工审查
    2 - 脚本执行错误（tvm-ffi 未找到等）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def find_tvm_ffi_include_dir() -> Optional[Path]:
    """查找 tvm-ffi C++ 头文件目录。

    优先级:
    1. 环境变量 TVM_FFI_INCLUDE_DIR
    2. python -m tvm_ffi.config --includedir
    3. 自动检测 vendor/tvm-ffi/include
    4. 自动检测 conda 环境
    """
    # 1. 环境变量
    env_dir = os.environ.get("TVM_FFI_INCLUDE_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    # 2. Python 包查询
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import tvm_ffi; print(tvm_ffi.__path__[0])"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
        )
        if result.returncode == 0:
            pkg_dir = Path(result.stdout.strip())
            # 尝试 locate include 目录
            for candidate in [
                pkg_dir.parent.parent / "include",
                pkg_dir / "include",
                Path(str(pkg_dir).replace("python/tvm_ffi", "include")),
            ]:
                tvm_ffi_h = candidate / "tvm" / "ffi" / "tvm_ffi.h"
                if tvm_ffi_h.exists():
                    return candidate
    except (subprocess.TimeoutExpired, Exception):
        pass

    # 3. 自动检测 vendor
    script_dir = Path(__file__).resolve().parent
    vendor_include = script_dir.parent.parent.parent.parent / "vendor" / "tvm-ffi" / "include"
    if vendor_include.is_dir():
        return vendor_include

    # 4. 工作区相对路径
    cwd_vendor = Path.cwd() / "vendor" / "tvm-ffi" / "include"
    if cwd_vendor.is_dir():
        return cwd_vendor

    return None


def extract_type_traits_specializations(include_dir: Path) -> dict[str, list[str]]:
    """从 tvm-ffi 头文件中提取所有 TypeTraits 特化。

    Returns:
        {header_file: [specialization_lines]}
    """
    specializations: dict[str, list[str]] = {}
    traits_pattern = re.compile(
        r'(?:template\s*<>\s*)?struct\s+(?:TypeTraits|TypeSchema|TypeSchemaImpl|'
        r'IsObjType|storage_enabled|type_name_helper|'
        r'ArgTypeInfo|RetTypeInfo|IsContainerType|IsObjectRefType)\s*[<{]'
    )

    for root, dirs, files in os.walk(include_dir):
        for f in files:
            if not f.endswith((".h", ".hpp")):
                continue
            filepath = os.path.join(root, f)
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as fh:
                    lines = fh.readlines()
            except Exception:
                continue

            matches = []
            for i, line in enumerate(lines, 1):
                if traits_pattern.search(line):
                    matches.append(f"L{i}: {line.strip()[:120]}")
            if matches:
                rel_path = os.path.relpath(filepath, include_dir)
                specializations[rel_path] = matches

    return specializations


# caffe-ffi 中使用的类型，需要检查是否与 vendor TypeTraits 冲突
CAFFE_FFI_CRITICAL_TYPES = [
    "ObjectPtr",
    "ObjectRef",
    "Tensor",
    "Array",
    "Map",
    "List",
    "Dict",
    "String",
    "Shape",
    "Function",
    "Optional",
    "Variant",
    "Tuple",
    "Any",
    "AnyView",
    "DLTensor",
    "DataType",
    "Device",
    "RValueRef",
    "SmallStr",
    "SmallBytes",
]


def check_custom_types_in_codebase(codebase_dir: Path) -> list[str]:
    """检查 caffe-ffi 代码库中是否有自定义 TypeTraits 定义。"""
    warnings: list[str] = []
    traits_keywords = [
        "TypeTraits", "TypeSchema", "TypeSchemaImpl",
        "IsObjType", "storage_enabled", "type_name_helper",
        "IsContainerType", "IsObjectRefType",
    ]

    for root, dirs, files in os.walk(codebase_dir):
        # 跳过 vendor、build、.git 等目录
        dirs[:] = [d for d in dirs if d not in ("vendor", "build", ".git", "__pycache__", ".temp")]

        for f in files:
            if not f.endswith((".h", ".hpp", ".cpp", ".cc")):
                continue
            filepath = os.path.join(root, f)
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except Exception:
                continue

            for keyword in traits_keywords:
                if keyword in content:
                    # 排除注释中的引用
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if keyword in stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
                            if "template" in stripped or "struct" in stripped or "using" in stripped:
                                # 排除 vendor 头文件引用
                                if "#include" not in stripped:
                                    rel_path = os.path.relpath(filepath, codebase_dir)
                                    warnings.append(f"{rel_path}:L{i}: {stripped[:150]}")

    return warnings


def run_check(include_dir: Path, codebase_dir: Path, json_output: bool = False) -> int:
    """执行完整预检。"""
    if not json_output:
        print("=" * 70)
        print("tvm-ffi TypeTraits 冲突预检")
        print(f"  tvm-ffi include: {include_dir}")
        print(f"  caffe-ffi codebase: {codebase_dir}")
        print("=" * 70)

    results = {
        "status": "PASS",
        "tvm_ffi_include": str(include_dir),
        "caffe_ffi_codebase": str(codebase_dir),
        "vendor_specializations": {},
        "custom_traits_warnings": [],
        "critical_type_conflicts": [],
    }

    # 1. 提取 vendor TypeTraits 特化
    if not json_output:
        print("\n[1/3] 扫描 tvm-ffi 已有 TypeTraits 特化...")
    specializations = extract_type_traits_specializations(include_dir)
    results["vendor_specializations"] = specializations

    if not json_output:
        print(f"  发现 {len(specializations)} 个文件包含 TypeTraits 特化:")
        for fname, matches in sorted(specializations.items()):
            print(f"    {fname}: {len(matches)} 个特化")
            for m in matches[:3]:  # 只显示前3个
                print(f"      {m}")
            if len(matches) > 3:
                print(f"      ... 还有 {len(matches) - 3} 个")

    # 2. 检查 caffe-ffi 代码库中是否有自定义 TypeTraits
    if not json_output:
        print("\n[2/3] 检查 caffe-ffi 代码库中是否有自定义 TypeTraits 定义...")
    custom_warnings = check_custom_types_in_codebase(codebase_dir)
    results["custom_traits_warnings"] = custom_warnings

    if custom_warnings:
        if not json_output:
            print(f"  ⚠️  发现 {len(custom_warnings)} 处自定义 TypeTraits 定义:")
            for w in custom_warnings:
                print(f"    {w}")
        results["status"] = "WARN"
    elif not json_output:
        print("  ✅ 未发现自定义 TypeTraits 定义")

    # 3. 检查关键类型是否可能冲突
    if not json_output:
        print("\n[3/3] 检查关键类型潜在冲突...")
    for ctype in CAFFE_FFI_CRITICAL_TYPES:
        for fname, matches in specializations.items():
            for m in matches:
                if ctype in m:
                    conflict = f"{fname}: {m}"
                    results["critical_type_conflicts"].append(conflict)
                    break

    if results["critical_type_conflicts"]:
        if not json_output:
            print(f"  ℹ️  以下关键类型在 vendor 中已有特化（不要重复定义）:")
            for c in results["critical_type_conflicts"]:
                print(f"    {c}")
    elif not json_output:
        print("  ✅ 关键类型分析完成")

    # 汇总
    if not json_output:
        print("\n" + "=" * 70)
        if results["status"] == "PASS" and not custom_warnings:
            print("✅ 预检通过：无 TypeTraits 冲突风险，可以开始 Phase 2 COW 实施")
        else:
            print("⚠️  预检发现潜在问题，请审查上述警告后再开始 Phase 2 COW 实施")
        print("=" * 70)

    if json_output:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    if results["status"] == "PASS" and not custom_warnings:
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="tvm-ffi TypeTraits 冲突预检脚本 (Phase 2 COW 前置检查)"
    )
    parser.add_argument(
        "--tvm-ffi-dir",
        help="tvm-ffi include 目录路径",
    )
    parser.add_argument(
        "--codebase-dir",
        help="caffe-ffi 代码库根目录",
        default=str(Path(__file__).resolve().parent.parent),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="以 JSON 格式输出结果",
    )
    args = parser.parse_args()

    # 确定 tvm-ffi include 目录
    include_dir = Path(args.tvm_ffi_dir) if args.tvm_ffi_dir else find_tvm_ffi_include_dir()
    if include_dir is None or not include_dir.is_dir():
        print("❌ 错误: 无法找到 tvm-ffi 头文件目录", file=sys.stderr)
        print("   请通过 --tvm-ffi-dir 参数指定，或设置 TVM_FFI_INCLUDE_DIR 环境变量",
              file=sys.stderr)
        if args.json:
            print(json.dumps({"status": "ERROR", "error": "tvm-ffi include directory not found"}))
        return 2

    codebase_dir = Path(args.codebase_dir)
    if not codebase_dir.is_dir():
        print(f"❌ 错误: caffe-ffi 代码库目录不存在: {codebase_dir}", file=sys.stderr)
        return 2

    return run_check(include_dir, codebase_dir, args.json)


if __name__ == "__main__":
    sys.exit(main())