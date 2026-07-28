# Windows 平台 DLL 符号导出与宏冲突技术复盘

> **日期**: 2026-07-28
> **影响范围**: npu-ffi 在 Windows/MSVC 下的 C++ 测试链接
> **严重程度**: P0 (阻塞 Windows 平台所有 C++ 测试编译)
> **修复状态**: 已修复并验证 (45/45 测试通过)

---

## 1. 问题概述

在 Windows 平台使用 Visual Studio 2026 (MSVC) 编译 npu-ffi 并链接 C++ 测试时，遇到两类编译/链接错误：

1. **链接错误 LNK2019**: 测试可执行文件无法解析 `npu_ffi::vta::Buffer`、`CommandContext` 等类的构造/析构函数，以及所有 runtime 自由函数
2. **编译错误 C2143**: `MemoryType::OUT` 枚举值与 Windows SDK 的 `OUT` SAL 注解宏冲突

### 错误信息摘要

```
# 链接错误（11个无法解析的外部符号）
test_buffer.obj : error LNK2019: 无法解析的外部符号
  "public: __cdecl npu_ffi::vta::Buffer::Buffer(unsigned __int64)"
  "public: __cdecl npu_ffi::vta::CommandContext::CommandContext(unsigned int)"
  ...

# 编译错误
error C2143: 语法错误: 缺少"}"(在"="的前面)
  (由 Windows SDK 的 OUT 宏替换 MemoryType::OUT 导致)
```

---

## 2. 根因分析

### 2.1 DLL 符号导出问题（根因）

| 平台 | 默认符号可见性 | 问题 |
|------|--------------|------|
| Linux/macOS (GCC/Clang) | 默认导出所有符号 (`-fvisibility=default`) | 无问题 |
| Windows (MSVC) | 默认不导出任何符号 | **类/函数需要显式声明 `__declspec(dllexport/dllimport)`** |

**根本原因**：`npu_ffi_vta` 被构建为 SHARED 库 (DLL)，但公共 API 类和函数没有标记 DLL 导出属性。Linux/macOS 下默认导出所有符号所以能正常链接，Windows 下需要显式声明。

**技术细节链**：

1. `add_library(npu_ffi_vta SHARED ...)` 构建 DLL
2. C++ 类 `Buffer`、`CommandContext` 和 `runtime.h` 中的自由函数无 `__declspec(dllexport)` 标记
3. MSVC 编译时这些符号不进入导出表
4. 测试 EXE 链接 DLL 时找不到符号 → LNK2019

### 2.2 Windows SDK `OUT` 宏冲突（根因）

```
Windows SDK <sal.h>:
  #define OUT _Out_  // SAL (Source Annotation Language) 注解宏

npu-ffi types.h:
  enum class MemoryType : uint32_t {
    ...
    OUT = 6,  // 被预处理器替换为 _Out_ = 6, → 语法错误
  };
```

**根本原因**：Windows SDK 通过头文件链（最终通过 `<windows.h>` 或间接包含）定义了 `OUT` 宏，而 C++ 枚举 `MemoryType::OUT` 的枚举器名与该宏冲突。

---

## 3. 修复方案

### 3.1 DLL 符号导出：双层策略

**策略一：显式标记公共 API（精确控制）**

在 [handle.h](../include/npu_ffi/vta/handle.h) 中添加 `NPU_FFI_API` 宏的条件定义（作为独立后备，不依赖 `npu_ffi.h`）：

```cpp
#ifndef NPU_FFI_API
  #if defined(_WIN32) || defined(_WIN64)
    #ifdef NPU_FFI_EXPORTS
      #define NPU_FFI_API __declspec(dllexport)
    #else
      #define NPU_FFI_API __declspec(dllimport)
    #endif
  #elif defined(__GNUC__) || defined(__clang__)
    #define NPU_FFI_API __attribute__((visibility("default")))
  #else
    #define NPU_FFI_API
  #endif
#endif
```

在类和函数声明上添加 `NPU_FFI_API`：

```cpp
// buffer.h
class NPU_FFI_API Buffer { ... };

// command_context.h
class NPU_FFI_API CommandContext { ... };
NPU_FFI_API void synchronize(CommandHandle cmd, uint32_t wait_cycles);

// runtime.h
NPU_FFI_API CommandHandle tls_command_handle();
NPU_FFI_API void runtime_shutdown();
NPU_FFI_API void load_buffer_2d(...);
// ... 所有公共函数
```

**策略二：WINDOWS_EXPORT_ALL_SYMBOLS（自动兜底）**

在 [src/vta/CMakeLists.txt](../src/vta/CMakeLists.txt) 中启用自动导出：

```cmake
set_target_properties(npu_ffi_vta PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
```

这会自动导出所有非静态符号（包括 `extern "C"` 的 C API 函数如 `npu_ffi_vta_push_gemm_op`），弥补手动标记可能遗漏的内部 C ABI 函数。

**为什么需要双层策略？**

| 策略 | 优点 | 缺点 |
|------|------|------|
| 显式 `__declspec` | 精确控制导出表大小、清晰表达意图、跨编译器可移植 | 需逐个标记，可能遗漏 |
| `WINDOWS_EXPORT_ALL_SYMBOLS` | 零手动标记、自动导出所有 | 导出表较大、不能导出数据成员、仅限 MSVC |
| **双层结合** | **精确标记公共 C++ API + 自动兜底 C ABI 符号** | - |

### 3.2 Windows `OUT` 宏冲突：`#undef` 后置清理

在 [types.h](../include/npu_ffi/vta/types.h) 中，在所有 `#include` 之后取消 `OUT` 宏定义：

```cpp
#ifdef _WIN32
// Windows SDK defines OUT as a SAL annotation macro, which collides with
// MemoryType::OUT. Undefine it after includes to avoid this collision.
#ifdef OUT
#undef OUT
#endif
#endif
```

**为什么在 include 之后而不是之前？**

- 如果在 `#include` 之前 `#undef OUT`，Windows SDK 头文件处理过程中仍会通过条件编译重新定义它
- 在所有 include 之后 `#undef`，确保枚举定义时 `OUT` 不再是宏
- 这是处理 Windows SDK 宏污染的标准模式（类似 `#undef min`/`#undef max` 处理 `<windows.h>` 的 `min`/`max` 宏）

---

## 4. 关键修复文件清单

| 文件 | 修改内容 |
|------|---------|
| [include/npu_ffi/vta/handle.h](../include/npu_ffi/vta/handle.h) | 添加 `NPU_FFI_API` 宏的平台条件定义 |
| [include/npu_ffi/vta/buffer.h](../include/npu_ffi/vta/buffer.h) | `class Buffer` → `class NPU_FFI_API Buffer` |
| [include/npu_ffi/vta/command_context.h](../include/npu_ffi/vta/command_context.h) | `class CommandContext` → `class NPU_FFI_API CommandContext`；前向声明函数加 `NPU_FFI_API` |
| [include/npu_ffi/vta/runtime.h](../include/npu_ffi/vta/runtime.h) | 所有 14 个公共函数声明添加 `NPU_FFI_API` |
| [include/npu_ffi/vta/types.h](../include/npu_ffi/vta/types.h) | 添加 `#undef OUT` 宏清理 |
| [src/vta/CMakeLists.txt](../src/vta/CMakeLists.txt) | 添加 `WINDOWS_EXPORT_ALL_SYMBOLS ON` |

---

## 5. 避坑指南

### 5.1 Windows DLL 导出：通用检查清单

新建 C++ SHARED 库时，**必须**在第一个头文件中定义导出宏：

```cpp
// 推荐模板：放在项目的 common.h 或 api.h 中
#if defined(_WIN32) || defined(_WIN64)
  #ifdef <PROJECT>_EXPORTS
    #define <PROJECT>_API __declspec(dllexport)
  #else
    #define <PROJECT>_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define <PROJECT>_API __attribute__((visibility("default")))
#else
  #define <PROJECT>_API
#endif
```

CMake 侧必须配合：

```cmake
add_library(<project> SHARED ${SOURCES})
target_compile_definitions(<project> PRIVATE <PROJECT>_EXPORTS)
set_target_properties(<project> PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)  # 兜底
```

### 5.2 Windows SDK 宏污染：常见冲突列表

| 宏名 | 来源 | 冲突场景 | 解决方案 |
|------|------|---------|---------|
| `OUT` | `<sal.h>` | 枚举/变量名为 OUT | `#undef OUT` 后使用 |
| `IN` | `<sal.h>` | 枚举/变量名为 IN | `#undef IN` 后使用 |
| `OPTIONAL` | `<sal.h>` | 变量名 OPTIONAL | `#undef OPTIONAL` |
| `min`/`max` | `<windows.h>` | `std::min`/`std::max` | `#define NOMINMAX` 或 `#undef min`/`#undef max` |
| `DeleteFile` | `<windows.h>` | 函数名 DeleteFile | 使用 `#ifdef DeleteFile` 后 `#undef` |
| `CreateDirectory` | `<windows.h>` | 函数名 CreateDirectory | 同上 |
| `CopyFile` | `<windows.h>` | 函数名 CopyFile | 同上 |
| `ERROR` | `<wingdi.h>` | 日志级别 ERROR | 命名空间隔离或使用枚举类 |

**推荐模式**（在每个可能被 Windows 用户包含的公共头文件末尾）：

```cpp
#ifdef _WIN32
#ifdef OUT
#undef OUT
#endif
#ifdef IN
#undef IN
#endif
#ifdef OPTIONAL
#undef OPTIONAL
#endif
#endif
```

### 5.3 CMake 生成器切换陷阱

切换 CMake 生成器时（如 Ninja → Visual Studio），**必须删除旧构建目录**：

```powershell
# 错误：混用不同生成器
cmake -G Ninja -B build ...
cmake -G "Visual Studio 18 2026" -B build ...  # 错误！生成器不匹配

# 正确：清理后重新配置
Remove-Item -Recurse -Force build
cmake -G "Visual Studio 18 2026" -A x64 -B build-vs ...
```

否则会出现 "generator does not match previous generator" 错误。

---

## 6. 验证结果

| 测试套件 | 修复前 | 修复后 |
|---------|-------|-------|
| test_buffer (13 tests) | LNK2019 链接失败 | ✅ 13 passed |
| test_memory_ops (10 tests) | LNK2019 链接失败 | ✅ 10 passed |
| test_compute_ops (13 tests) | LNK2019 链接失败 | ✅ 13 passed |
| test_gemm_e2e (9 tests) | 未创建 | ✅ 9 passed |
| **合计** | **构建失败** | **45/45 通过** |

Debug 日志输出验证（`NPU_FFI_ENABLE_LOG=ON`）：

```
[npu-ffi][DEBUG] Buffer allocated: ptr=0x..., size=1024
[npu-ffi][DEBUG] CommandContext created: cmd=0x..., wait_cycles=0
[npu-ffi][DEBUG] CommandContext::synchronize: cmd=0x..., wait_cycles=0
[npu-ffi][DEBUG] Buffer freeing: ptr=0x..., size=1024
```

---

## 7. 预防措施

1. **CI 多平台构建**: GitHub Actions CI 已包含 Windows 构建 ([.github/workflows/ci.yml](../.github/workflows/ci.yml))，任何 DLL 导出遗漏会在 PR 阶段被捕获
2. **CMakePresets.json**: 使用预设（`debug-log`、`release` 等）避免手动指定生成器导致的配置错误
3. **头文件宏清理模板**: 公共头文件统一在末尾添加 Windows 宏 `#undef` 清理块
4. **WINDOWS_EXPORT_ALL_SYMBOLS**: 作为 C ABI 符号的兜底措施，始终为 SHARED 库启用

---

## 8. 经验总结

1. **Linux/macOS ≠ Windows**: GCC/Clang 默认导出所有符号是 GCC 的特性而非标准行为。编写跨平台 C++ 库时必须从第一天就考虑 DLL 导出。
2. **双层防御 > 单层**: 显式 `__declspec` + `WINDOWS_EXPORT_ALL_SYMBOLS` 的组合比单独使用任一种更健壮。
3. **宏污染是 Windows 特有痛点**: `<windows.h>` 定义了数百个宏（`OUT`、`IN`、`min`、`max`、`ERROR`、`DeleteFile` 等），在设计跨平台库的公共 API 命名时需要特别注意。
4. **enum class 不能防止宏替换**: 即使使用 `enum class MemoryType { OUT = 6 }`，预处理器仍会在编译器看到枚举定义之前替换 `OUT`，必须在头文件中 `#undef`。
