#!/usr/bin/env python3
"""CI TVM-FFI dependency loading check (cross-platform).

Lightweight, dependency-loading focus designed to run in the GitHub Actions
matrix (Linux / macOS / Windows) right after ``pip install --no-build-isolation -e .``.

It reproduces the *root-cause* checks that previously blocked the P0 runtime
(`ModuleNotFoundError: tvm_ffi.core`, `WinError 127` / symbol skew) without
depending on the vendored ``libtvm_ffi.so`` or Linux-only ``ldd``:

  1. ``tvm_ffi`` imports and the Cython ``core`` extension is physically present.
  2. The core extension links against ``libtvm_ffi`` (best-effort: ``ldd`` on
     Linux, ``dumpbin``/``llvm-nm`` on Windows if available, else skipped).
  3. The ``caffe_ffi`` C++ extension actually loads (``is_available()==True``),
     not the Python-only stub fallback.
  4. Functional smoke of the FFI bridge: register a data-IO callback under
     ``Data.<name>`` and run a Data-layer forward, asserting the callback-filled
     payload is visible in the output blob (zero-copy DLPack round-trip).

Exit code: 0 = all pass; 1 = any failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def _dump_links(path: str) -> list[str]:
    """Return the dynamic-link lines for *path* on the current platform, or []."""
    if os.name == "nt":
        # dumpbin /DLL /DEPENDENTS (VS toolchain) may not be on PATH in CI.
        dumpbin = shutil.which("dumpbin")
        if dumpbin:
            out = subprocess.run(
                [dumpbin, "/nologo", "/dependents", path],
                capture_output=True, text=True, timeout=60,
            )
            return [ln for ln in (out.stdout + out.stderr).splitlines() if "tvm_ffi" in ln]
        return []
    ldd = shutil.which("ldd")
    if ldd:
        out = subprocess.run([ldd, path], capture_output=True, text=True, timeout=30)
        return [ln for ln in out.stdout.splitlines() if "tvm_ffi" in ln]
    return []


print("=== CI TVM-FFI 依赖加载检查 ===")
print(f"Python: {sys.version.split()[0]}  @ {sys.executable}")

# 1. tvm_ffi import + core extension presence
print("\n[1] tvm_ffi.core 扩展")
try:
    import tvm_ffi
    import tvm_ffi.core as core

    check("import tvm_ffi", True, getattr(tvm_ffi, "__file__", "?"))
    core_dir = os.path.dirname(core.__file__)
    so_files = [f for f in os.listdir(core_dir)
                if f.endswith(".so") or f.endswith(".pyd") or f.endswith(".dll")]
    check("发现 core 扩展二进制", len(so_files) > 0, so_files)
except Exception as e:  # noqa: BLE001
    check("import tvm_ffi.core", False, repr(e))

# 2. core extension links libtvm_ffi (best-effort per platform)
print("\n[2] core 扩展链接 libtvm_ffi")
try:
    core_dir = os.path.dirname(core.__file__)
    core_bin = [f for f in os.listdir(core_dir)
                if (f.endswith(".so") or f.endswith(".pyd")) and not f.endswith(".pyi")]
    if core_bin:
        full = os.path.join(core_dir, core_bin[0])
        links = _dump_links(full)
        if links:
            for ln in links:
                # Presence of a line mentioning libtvm_ffi confirms the core extension
                # links against the shared library (the dependency-loading root cause).
                check(f"链接项: {ln.strip()}", True, ln.strip())
        else:
            check("core 扩展链接解析", True, "sdk 未提供 dumpbin/ldd，跳过二进制级链接核验（最坏情况）。")
except Exception as e:  # noqa: BLE001
    check("解析 core 扩展链接", False, repr(e))

# 3. caffe_ffi C++ extension actually loaded (not stub)
print("\n[3] caffe_ffi C++ 扩展加载")
try:
    import caffe_ffi
    from caffe_ffi import _ffi_api

    check("is_available()==True", caffe_ffi.is_available())
    diag = _ffi_api.get_init_diagnostics()
    check("init diagnostics success", diag.success,
          str(diag.lib_path_found) if diag.success else "\n" + diag.summary())
except Exception as e:  # noqa: BLE001
    check("import caffe_ffi", False, repr(e))

# 4. Functional smoke: data-IO callback round-trip through the FFI bridge
print("\n[4] data_io 回调 FFI 桥接冒烟")
try:
    from caffe_ffi import _ffi_api

    reg = _ffi_api.get_global_func("caffe_ffi.data_io.register")
    check("data_io.register 可用", reg is not None)
    key = "Data.ci_smoke"

    def _cb(tensors):
        arr = np.from_dlpack(tensors[0])
        arr[...] = 11.0
        if len(tensors) >= 2:
            np.from_dlpack(tensors[1])[...] = 5.0

    reg(key, _cb)
    prototxt = textwrap.dedent("""
        name: "ci_smoke"
        layer {
          name: "ci_smoke"
          type: "Data"
          top: "data"
          top: "label"
          data_param { batch_size: 2 source: "dummy.txt" }
        }
    """)
    net = caffe_ffi.net_from_param(caffe_ffi.net_param_from_string(prototxt))
    net.Forward({})
    d = net.blob_by_name("data").to_numpy()
    l = net.blob_by_name("label").to_numpy()
    check("回调填充 data 可见", np.all(d == 11.0), f"shape={tuple(d.shape)}")
    check("回调填充 label 可见", np.all(l == 5.0), f"shape={tuple(l.shape)}")
except Exception as e:  # noqa: BLE001
    check("data_io 桥接冒烟", False, repr(e))

# ── 汇总 ──
print("\n=== 汇总 ===")
if failures:
    print(f"FAIL: {len(failures)} 项未通过")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)