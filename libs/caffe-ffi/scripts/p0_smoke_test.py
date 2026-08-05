#!/usr/bin/env python3
"""P0 环境运行时冒烟测试：验证 tvm-ffi 加载 + caffe-ffi C++ 扩展 + P2 算子注册。

在 P0 环境（WSL Docker: caffe-ffi-jupyter）中运行：
    /opt/conda/envs/caffe-ffi/bin/python scripts/p0_smoke_test.py

退出码：0 = 全部通过；1 = 任一子项失败。
"""
from __future__ import annotations

import sys

import caffe_ffi
from caffe_ffi import _ffi_api

# ── P2 阶段新增/改动的算子（来自 gap_analysis_report.md / P2 tasks.md）──
P2_LAYERS = [
    "Data",
    "ImageData",
    "HDF5Data",
    "HDF5Output",
    "MemoryData",
    "WindowData",
    "DummyData",
    "Python",
    "ContrastiveLoss",
    "InfogainLoss",
    "MultinomialLogisticLoss",
    "Upsample",
]

# 不应注册的抽象基类 / GPU 专属层
NOT_REGISTERED = ["Recurrent"]  # Recurrent 为抽象基类，刻意不注册

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


print("=== P0 运行时冒烟测试 ===")
print(f"Python: {sys.version.split()[0]}  @ {sys.executable}")

# 1. tvm_ffi 可导入（Cython core 扩展已构建）
print("\n[1] tvm_ffi 导入")
try:
    import tvm_ffi

    check("import tvm_ffi", True, tvm_ffi.__file__)
    check("Function API (新版命名)", hasattr(tvm_ffi, "Function"))
    check("Object API", hasattr(tvm_ffi, "Object"))
    check("register_object", hasattr(tvm_ffi, "register_object"))
except Exception as e:  # noqa: BLE001
    check(f"import tvm_ffi", False, repr(e))

# 2. caffe_ffi FFI 初始化（C++ 扩展是否真正加载，而非 stub 回退）
print("\n[2] caffe_ffi FFI 初始化")
diag = _ffi_api.get_init_diagnostics()
check("is_available()==True", caffe_ffi.is_available())
if diag.success:
    check("init diagnostics success", True, str(diag.lib_path_found))
else:
    check("init diagnostics success", False, "\n" + diag.summary())

# 3. 版本
print("\n[3] 版本")
try:
    v = caffe_ffi.version()
    check("version()", True, str(v))
except Exception as e:  # noqa: BLE001
    check("version()", False, repr(e))

# 4. LayerTypeList 注册（含 P2 算子）
print("\n[4] LayerTypeList 算子注册")
fn = _ffi_api.get_global_func("caffe_ffi.LayerTypeList")
if fn is None:
    check("get LayerTypeList func", False, "global func not found")
    sys.exit(1)
try:
    types = list(fn())
    check("获得 LayerTypeList", len(types) > 0, f"共 {len(types)} 个算子")
    print(f"  已注册算子({len(types)}): {types}")
    registered_set = set(types)
    for p2 in P2_LAYERS:
        check(f"  P2 算子 {p2} 已注册", p2 in registered_set)
    for nr in NOT_REGISTERED:
        check(f"  抽象基类 {nr} 未注册", nr not in registered_set)
except Exception as e:  # noqa: BLE001
    check("LayerTypeList 调用", False, repr(e))

# 5. 汇总
print("\n=== 汇总 ===")
if failures:
    print(f"FAIL: {len(failures)} 项未通过")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)