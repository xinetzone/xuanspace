#!/usr/bin/env python3
"""
npu-ffi VTA Python API 基本使用示例

演示内容：
- Buffer RAII 用法（自动内存管理）
- CommandContext 自动同步
- 类型安全 API（buffer_copy_safe、MemcpyKind 枚举）
- from_foreign_pointer 包装外部指针
- reset() 显式释放资源

使用方法:
    python examples/basic_usage.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from npu_ffi import vta
from npu_ffi.vta import Buffer, CommandContext, MemcpyKind


def section(title: str) -> None:
    """打印分隔标题"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_buffer_raii() -> None:
    """演示 Buffer RAII 自动内存管理"""
    section("1. Buffer RAII - 自动内存管理")
    buf = Buffer(1024)
    print(f"  分配缓冲区: {buf!r}")
    print(f"  data=0x{buf.data:x}, size={buf.size}, owns={buf.owns_data}")
    print(f"  len(buf) = {len(buf)}")
    del buf
    print("  Buffer 已自动释放")


def demo_context_manager() -> None:
    """演示 with 上下文管理器"""
    section("2. Buffer 上下文管理器")
    with Buffer(512) as buf:
        print(f"  在 with 块内: {buf!r}")
        assert buf.data != 0
    print(f"  退出 with 块后: data=0x{buf.data:x} (已自动释放)")


def demo_buffer_reset() -> None:
    """演示显式 reset() 释放资源"""
    section("3. 显式 reset() 释放资源")
    buf = Buffer(256)
    print(f"  初始状态: {buf!r}")
    buf.reset()
    print(f"  reset() 后: {buf!r}")
    buf.reset()
    print("  再次 reset() 是安全的（幂等操作）")


def demo_from_foreign_pointer() -> None:
    """演示包装外部指针（不持有所有权）"""
    section("4. from_foreign_pointer - 包装外部指针")
    raw_ptr = vta.buffer_alloc(128)
    print(f"  原始指针: 0x{raw_ptr:x}")
    wrapped = Buffer.from_foreign_pointer(raw_ptr, 128)
    print(f"  包装后: {wrapped!r}")
    print(f"  owns_data={wrapped.owns_data} (不持有所有权)")
    del wrapped
    print("  包装对象销毁，但原始指针未被释放")
    vta.buffer_free(raw_ptr)
    print("  原始指针已手动释放")


def demo_double_free_protection() -> None:
    """演示 double-free 保护"""
    section("5. Double-free 保护")
    buf = Buffer(64)
    print(f"  初始: {buf!r}")
    buf.reset()
    print(f"  reset() 后: {buf!r}")
    del buf
    print("  __del__ 再次调用不会报错（安全）")


def demo_command_context() -> None:
    """演示 CommandContext 自动同步"""
    section("6. CommandContext - 自动同步")
    with CommandContext() as cmd:
        print(f"  在 with 块内: cmd=0x{cmd:x}")
        inp = Buffer(1024)
        wgt = Buffer(1024)
        acc = Buffer(1024)
        print(f"  分配了 3 个缓冲区")
        vta.uop_push(0, 1, 0, 0, 0, int(vta.ALUOpcode.ADD), 0, 0)
        print("  推送了 uop 命令")
    print("  退出 with 块时自动同步")


def demo_command_context_explicit_sync() -> None:
    """演示 CommandContext 显式同步"""
    section("7. CommandContext 显式 synchronize()")
    ctx = CommandContext()
    with ctx as cmd:
        print(f"  cmd=0x{cmd:x}")
        ctx.synchronize()
        print(f"  显式 synchronize() 后: ctx._cmd={ctx._cmd}")
        ctx.synchronize()
        print("  再次 synchronize() 是安全的")


def demo_get_command_handle() -> None:
    """演示 get_command_handle() 便捷函数"""
    section("8. get_command_handle() - command_handle() 的别名")
    cmd1 = vta.command_handle()
    cmd2 = vta.get_command_handle()
    print(f"  command_handle(): 0x{cmd1:x}")
    print(f"  get_command_handle(): 0x{cmd2:x}")
    print("  两者功能相同")


def demo_buffer_copy_safe() -> None:
    """演示类型安全的 buffer_copy_safe"""
    section("9. buffer_copy_safe - 类型安全缓冲区拷贝")
    print("  使用 MemcpyKind 枚举:")
    print(f"    H2D = {int(MemcpyKind.H2D)} (Host to Device)")
    print(f"    D2H = {int(MemcpyKind.D2H)} (Device to Host)")
    print(f"    D2D = {int(MemcpyKind.D2D)} (Device to Device)")

    with Buffer(256) as src, Buffer(256) as dst:
        vta.buffer_copy_safe(src, 0, dst, 0, 256, MemcpyKind.D2D)
        print("  buffer_copy_safe(src, 0, dst, 0, 256, MemcpyKind.D2D)")
        print("  - 自动从 Buffer 对象提取 data 指针")
        print("  - 自动将 MemcpyKind 枚举转为 int")

    raw_src = vta.buffer_alloc(128)
    raw_dst = vta.buffer_alloc(128)
    vta.buffer_copy_safe(raw_src, 0, raw_dst, 0, 128, MemcpyKind.H2D)
    print("  也支持直接传入 int 指针")
    vta.buffer_free(raw_src)
    vta.buffer_free(raw_dst)


def demo_runtime_shutdown() -> None:
    """演示 runtime_shutdown 幂等性"""
    section("10. runtime_shutdown() - 幂等安全")
    vta.runtime_shutdown()
    vta.runtime_shutdown()
    print("  runtime_shutdown() 可安全调用多次")


def demo_repr() -> None:
    """演示 __repr__ 输出"""
    section("11. __repr__ 调试输出")
    buf = Buffer(1024)
    print(f"  Buffer repr: {buf!r}")
    ctx = CommandContext()
    print(f"  CommandContext (未进入): {ctx!r}")
    with ctx as cmd:
        print(f"  CommandContext (已进入): {ctx!r}")
    buf.reset()


def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       npu-ffi VTA Python API - 类型安全使用演示             ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    demo_buffer_raii()
    demo_context_manager()
    demo_buffer_reset()
    demo_from_foreign_pointer()
    demo_double_free_protection()
    demo_command_context()
    demo_command_context_explicit_sync()
    demo_get_command_handle()
    demo_buffer_copy_safe()
    demo_runtime_shutdown()
    demo_repr()

    section("演示完成")
    print("  所有新特性已演示完毕。")
    print()
    print("  新增/改进特性清单:")
    print("    • Buffer.double-free 保护（__del__/__exit__/reset）")
    print("    • Buffer.from_foreign_pointer() 类方法")
    print("    • Buffer.reset() 显式释放")
    print("    • Buffer.__repr__() 调试输出")
    print("    • CommandContext.__repr__() 调试输出")
    print("    • CommandContext.synchronize() 显式同步（double-sync 保护）")
    print("    • vta.buffer_copy_safe() 类型安全拷贝")
    print("    • vta.get_command_handle() 便捷别名")
    print("    • tvm-ffi 版本号修正为 0.0.1")
    print()


if __name__ == "__main__":
    main()
