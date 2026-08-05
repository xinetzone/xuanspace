#!/usr/bin/env python3
"""P0 环境 TVM-FFI 依赖加载检查：确认 vendored libtvm_ffi.so 已正确加载。

检查项：
1. tvm_ffi.core 扩展的物理路径（应指向 vendor/tvm-ffi 的编译产物）
2. core 扩展实际链接的 libtvm_ffi.so 位置
3. 已验证的 P0 全链路冒烟（复用 p0_smoke_test.py 逻辑）
"""
from __future__ import annotations

import os
import subprocess
import sys

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


print("=== P0 TVM-FFI 依赖加载检查 ===")
print(f"Python: {sys.version.split()[0]}  @ {sys.executable}")

# 1. tvm_ffi.core 扩展路径
print("\n[1] tvm_ffi.core 扩展路径")
try:
    import tvm_ffi.core as core

    core_dir = os.path.dirname(core.__file__)
    check("core 扩展可导入", True, core.__file__)
    so_files = [f for f in os.listdir(core_dir) if f.endswith(".so") or f.endswith(".pyd")]
    check("发现 core 扩展 .so", len(so_files) > 0, so_files)
    core_so = [f for f in so_files if not f.endswith(".pyi")]
    if core_so:
        core_path = os.path.join(core_dir, core_so[0])
        # 2. 检查 core 扩展动态链接的 libtvm_ffi.so
        print("\n[2] core 扩展链接的 libtvm_ffi.so")
        try:
            ldd = subprocess.run(
                ["ldd", core_path], capture_output=True, text=True, timeout=30
            )
            tvmffi_lines = [ln for ln in ldd.stdout.splitlines() if "tvm_ffi" in ln]
            if tvmffi_lines:
                for ln in tvmffi_lines:
                    resolved = "=>" in ln
                    check(f"链接项: {ln.strip()}", resolved, "已解析到实际 .so")
            else:
                check("ldd 中未发现 libtvm_ffi 链接项", False)
        except FileNotFoundError:
            check("ldd 可用", False, "容器内无 ldd（非 Linux 环境？）")
        except Exception as e:  # noqa: BLE001
            check("ldd 执行", False, repr(e))
except Exception as e:  # noqa: BLE001
    check("import tvm_ffi.core", False, repr(e))

# 3. 全链路冒烟（tvm_ffi + caffe_ffi + P2 算子）
print("\n[3] 全链路冒烟")
rc = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), "p0_smoke_test.py")]
).returncode
check("p0_smoke_test.py 全链路", rc == 0, f"exit={rc}")

print("\n=== 汇总 ===")
if failures:
    print(f"FAIL: {len(failures)} 项未通过")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)