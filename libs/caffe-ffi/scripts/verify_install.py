#!/usr/bin/env python3

import sys
import importlib.metadata as im


def _check_env() -> list[tuple[str, bool, str]]:
    results = []

    kmp_ok = True
    if sys.platform.startswith("win"):
        import os
        kmp_val = os.environ.get("KMP_DUPLICATE_LIB_OK", "")
        kmp_ok = kmp_val.upper() == "TRUE"
        results.append((
            "KMP_DUPLICATE_LIB_OK 环境变量",
            kmp_ok,
            "已设置为 TRUE" if kmp_ok else f"未设置或值不正确: {kmp_val!r} (Windows 必须设置为 TRUE)"
        ))
    else:
        results.append(("KMP_DUPLICATE_LIB_OK 环境变量", True, "非 Windows 平台，无需设置"))

    return results


def _check_python_version() -> tuple[str, bool, str]:
    version = sys.version_info
    ok = version >= (3, 14)
    return (
        "Python 版本",
        ok,
        f"{version.major}.{version.minor}.{version.micro} (要求 >= 3.14)" if ok
        else f"{version.major}.{version.minor}.{version.micro} (版本过低，要求 >= 3.14)"
    )


def _check_package(pkg_name: str) -> tuple[str, bool, str]:
    try:
        dist = im.distribution(pkg_name)
        version = dist.version
        return (f"包 {pkg_name}", True, f"已安装，版本: {version}")
    except im.PackageNotFoundError:
        return (f"包 {pkg_name}", False, "未安装")


def _check_tvm_ffi_import() -> tuple[str, bool, str]:
    try:
        import tvm_ffi
        version = getattr(tvm_ffi, "__version__", "unknown")
        return ("tvm_ffi 导入", True, f"成功，版本: {version}")
    except Exception as e:
        return ("tvm_ffi 导入", False, f"失败: {e!r}")


def _check_caffe_ffi_import() -> tuple[str, bool, str]:
    try:
        import caffe_ffi
        version = getattr(caffe_ffi, "__version__", "unknown")
        return ("caffe_ffi 导入", True, f"成功，版本: {version}")
    except Exception as e:
        return ("caffe_ffi 导入", False, f"失败: {e!r}")


def _check_caffe_import() -> tuple[str, bool, str]:
    try:
        from caffe_ffi import caffe
        return ("caffe 模块导入", True, "成功")
    except Exception as e:
        return ("caffe 模块导入", False, f"失败: {e!r}")


def _check_basic_functionality() -> tuple[str, bool, str]:
    try:
        import caffe_ffi
        from caffe_ffi.blob import Blob
        import numpy as np
        
        blob = Blob()
        blob.reshape(1, 1, 2, 2)
        data = blob.data
        assert data.shape == (1, 1, 2, 2), f"Expected shape (1,1,2,2), got {data.shape}"
        
        return ("Blob 基础功能测试", True, "成功，Blob 创建和形状设置正常")
    except Exception as e:
        return ("Blob 基础功能测试", False, f"失败: {e!r}")


def main() -> int:
    print("=" * 70)
    print("caffe-ffi 安装验证")
    print("=" * 70)

    checks: list[tuple[str, bool, str]] = []
    checks.append(_check_python_version())
    checks.extend(_check_env())
    checks.append(_check_package("apache-tvm-ffi"))
    checks.append(_check_tvm_ffi_import())
    checks.append(_check_package("caffe-ffi"))
    checks.append(_check_caffe_ffi_import())
    checks.append(_check_caffe_import())
    checks.append(_check_basic_functionality())

    passed = 0
    failed = 0

    for name, ok, msg in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"\n[{status}] {name}")
        print(f"       {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    if failed > 0:
        print("\n故障排除建议:")
        print("1. 确认已按顺序安装:")
        print("   cd libs/tvm-ffi && pip install --no-build-isolation -e .")
        print("   cd ../caffe-ffi && pip install --no-build-isolation -e .")
        print("2. Windows 用户确认设置 KMP_DUPLICATE_LIB_OK=TRUE")
        print("3. 如果导入失败，尝试重新构建 C++ 扩展:")
        print("   python scripts/dev.ps1 -Rebuild  (Windows)")
        print("   ./scripts/dev.sh -r              (Linux/macOS/WSL)")
        print("4. 运行 scripts/check_ffi_prefix.py 检查 FFI 前缀一致性")
        return 1
    else:
        print("\n🎉 所有检查通过！caffe-ffi 安装正确，可以正常使用。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
