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


def _check_npu_ffi_import() -> tuple[str, bool, str]:
    try:
        import npu_ffi
        version = getattr(npu_ffi, "__version__", "unknown")
        return ("npu_ffi 导入", True, f"成功，版本: {version}")
    except Exception as e:
        return ("npu_ffi 导入", False, f"失败: {e!r}")


def _check_vta_import() -> tuple[str, bool, str]:
    try:
        from npu_ffi import vta
        version = getattr(vta, "__version__", "unknown")
        has_buffer_alloc = hasattr(vta, "buffer_alloc")
        has_command = hasattr(vta, "tls_command_handle")
        return (
            "vta FFI 模块导入",
            True,
            f"成功，版本: {version}，buffer_alloc: {has_buffer_alloc}，tls_command_handle: {has_command}"
        )
    except Exception as e:
        return ("vta FFI 模块导入", False, f"失败: {e!r}")


def _check_buffer_alloc() -> tuple[str, bool, str]:
    try:
        from npu_ffi import vta
        buf_size = 1024
        buf = vta.buffer_alloc(buf_size)
        vta.buffer_free(buf)
        return ("Buffer 分配/释放测试", True, f"成功，分配 {buf_size} 字节缓冲区")
    except Exception as e:
        return ("Buffer 分配/释放测试", False, f"失败: {e!r}")


def _check_command_context() -> tuple[str, bool, str]:
    try:
        from npu_ffi.vta import CommandContext
        with CommandContext() as cmd:
            handle = cmd if isinstance(cmd, int) else getattr(cmd, 'handle', None)
        return ("CommandContext 上下文管理器", True, "成功")
    except Exception as e:
        return ("CommandContext 上下文管理器", False, f"失败: {e!r}")


def main() -> int:
    print("=" * 70)
    print("npu-ffi 安装验证")
    print("=" * 70)

    checks: list[tuple[str, bool, str]] = []
    checks.append(_check_python_version())
    checks.extend(_check_env())
    checks.append(_check_package("apache-tvm-ffi"))
    checks.append(_check_tvm_ffi_import())
    checks.append(_check_package("npu-ffi"))
    checks.append(_check_npu_ffi_import())
    checks.append(_check_vta_import())
    checks.append(_check_buffer_alloc())
    checks.append(_check_command_context())

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
        print("   cd vendor/tvm-ffi && pip install --no-build-isolation -e .")
        print("   cd ../../libs/npu-ffi && pip install --no-build-isolation -e .")
        print("2. Windows 用户确认设置 KMP_DUPLICATE_LIB_OK=TRUE")
        print("3. 如果导入 vta 失败，尝试重新构建 C++ 扩展:")
        print("   python scripts/dev.ps1 -Rebuild  (Windows)")
        print("   ./scripts/dev.sh                 (Linux/macOS)")
        print("4. 运行 scripts/check_ffi_prefix.py 检查 FFI 前缀一致性")
        return 1
    else:
        print("\n🎉 所有检查通过！npu-ffi 安装正确，可以正常使用。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
