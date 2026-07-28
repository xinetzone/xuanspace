# 模式萃取：基于 tvm-ffi 构建类型安全 FFI 绑定

> **来源**: npu-ffi VTA FFI 优化项目
> **萃取日期**: 2026-07-28
> **方法论**: 七概念方法论 (I→F→A→E)

---

## 1. 模式概述

**模式名称**: 五层类型安全 FFI 绑定架构

**触发场景**: 当需要基于 tvm-ffi 框架为 C/C++ 硬件加速库（如VTA/NPU）构建 Python/C++ 双语言绑定时。

**核心思想**: 在 C ABI 稳定层之上构建多层类型安全包装，通过桥接模式消除后端重复代码，使用 RAII 自动管理资源，在 FFI 边界通过类型擦除（int64_t）统一传递，在上层恢复类型安全。

---

## 2. 核心架构

```
┌─────────────────────────────────────┐
│  应用层 (用户代码)                    │
├─────────────────────────────────────┤
│  类型安全API层 (新增/改进)            │  ← Python: buffer_copy_safe, Buffer.from_foreign_pointer
│  （接收类型化对象/枚举）              │  ← C++: Buffer, CommandContext, enum class
├─────────────────────────────────────┤
│  FFI注册层 (改进)                    │  ← ffi_registry.cc: int64_t类型参数，TVM_FFI_STATIC_INIT_BLOCK
│  （类型擦除：指针→int64_t）           │
├─────────────────────────────────────┤
│  公共宿主逻辑层 (新增)               │  ← runtime_common.cc: RAII, 移动语义, nullptr检查
│  （桥接模式消除重复）                 │
├─────────────────────────────────────┤
│  C ABI稳定层 (extern "C")           │  ← npu_ffi_vta_* 函数
│  （后端实现契约）                     │
├─────────────────────────────────────┤
│  后端实现层                          │  ← stub_rt.cc (mock) / real_rt.cc (硬件)
│  （仅实现alloc/free等原语）           │
└─────────────────────────────────────┘
```

---

## 3. 实施步骤

### 步骤1：定义稳定的 C ABI 边界

**关键规则**:
- 所有跨边界函数使用 `extern "C"` 声明
- 指针用 `void*` 传递，数值用明确宽度类型（`uint32_t`/`int32_t`/`size_t`）
- 错误通过 `nullptr` 返回值或整数错误码指示，不跨边界抛C++异常
- 函数名使用统一前缀（如 `npu_ffi_vta_`）

**反模式**:
```cpp
// ❌ 错误：C++类型跨边界，std::string引用
void npu_ffi_func(const std::string& name);
// ❌ 错误：窄类型传递宽值
void register_func(int size);  // size可能超过2^31
```

**正确模式**:
```cpp
// ✅ 正确：C ABI稳定
extern "C" {
  void* npu_ffi_vta_buffer_alloc(size_t size);
  void npu_ffi_vta_prepare_call_func(void* cmd, const char* name);
}
```

### 步骤2：提取公共宿主逻辑（桥接模式）

**关键规则**:
- 创建 `*_common.cc` 文件，实现RAII类（Buffer/Handle/Context）的所有成员函数
- 后端文件（stub/real）仅实现extern "C"的C函数
- 公共逻辑中进行参数验证（nullptr检查、范围检查）
- 移动语义必须正确实现：移动后原对象重置为empty状态

**反模式**:
```cpp
// ❌ 错误：两个后端文件重复实现Buffer成员函数
// stub_rt.cc:
Buffer::Buffer(size_t size) : data_(stub_alloc(size)), size_(size), owns_(true) {}
// real_rt.cc:
Buffer::Buffer(size_t size) : data_(real_alloc(size)), size_(size), owns_(true) {}
// （~50行重复代码，bug修复需要改两处）
```

**正确模式**:
```cpp
// ✅ 正确：公共层调用后端C函数
// runtime_common.cc:
Buffer::Buffer(size_t size)
    : data_(size > 0 ? npu_ffi_vta_buffer_alloc(size) : nullptr),
      size_(size), owns_(true) {
  if (size > 0 && data_ == nullptr) throw std::bad_alloc();
}
// stub_rt.cc: 仅实现 extern "C" npu_ffi_vta_buffer_alloc
// real_rt.cc: 仅实现 extern "C" npu_ffi_vta_buffer_alloc
```

### 步骤3：FFI注册层类型安全

**关键规则**:
- 所有指针/句柄参数用 `int64_t` 传递（不是 `int`）
- uint32_t类型参数用 `int64_t` 传递，内部 `static_cast<uint32_t>`
- 函数指针和签名指针也用 `int64_t` 传递指针整数值
- 使用 `TVM_FFI_STATIC_INIT_BLOCK()` 自动注册
- 添加前缀一致性注释提醒维护者

**反模式**:
```cpp
// ❌ 错误：int传递uint32_t，大值截断
void load_buffer_2d(int64_t cmd, int src_elem_offset, int x_size, ...) {
  npu_ffi_vta_load_buffer_2d(..., (uint32_t)src_elem_offset, ...);
}
```

**正确模式**:
```cpp
// ✅ 正确：int64_t传递所有数值参数
void load_buffer_2d(int64_t cmd, int64_t src_elem_offset, int64_t x_size, ...) {
  npu_ffi_vta_load_buffer_2d(
      reinterpret_cast<void*>(static_cast<intptr_t>(cmd)),
      static_cast<uint32_t>(src_elem_offset),
      static_cast<uint32_t>(x_size), ...);
}
```

### 步骤4：C++类型安全层

**关键规则**:
- 使用RAII类管理资源：构造获取，析构释放
- 禁止拷贝，允许移动（`= delete` 拷贝，`noexcept` 移动）
- 使用类型安全的包装类（`CommandHandle` 替代 `void*`）
- 使用 `enum class` 替代裸int枚举
- 提供 `const` 重载的访问器
- 添加上下文管理器类（CommandContext）自动同步/清理

**CommandContext RAII模式**:
```cpp
class CommandContext {
 public:
  explicit CommandContext(uint32_t wait_cycles = 0);
  ~CommandContext();  // 自动synchronize
  // 禁止拷贝，允许移动
  CommandContext(const CommandContext&) = delete;
  CommandContext(CommandContext&&) noexcept;
  void synchronize();  // 显式提前同步（幂等）
 private:
  CommandHandle cmd_;
  uint32_t wait_cycles_;
  bool active_;  // 防止重复synchronize
};
```

### 步骤5：Python类型安全层

**关键规则**:
- Python类实现上下文管理器（`__enter__`/`__exit__`）
- Double-free保护：析构前检查 `_data != 0`，释放后置0
- 提供类型安全的包装函数：接受Python对象/枚举，自动提取底层int
- 析构函数中捕获所有异常（Python不允许析构抛异常），但添加注释说明原因
- 添加 `__repr__` 便于调试

**Double-free保护模式**:
```python
def __del__(self):
    """Destructor - frees buffer if owns and data is valid.
    Note: We deliberately catch all exceptions here because Python
    destructors must never raise (raising in __del__ terminates the
    interpreter).
    """
    if self._owns and self._data != 0:
        try:
            _ffi_api.buffer_free(self._data)
        except Exception:
            pass
        self._data = 0
```

### 步骤6：构建系统现代化

**关键规则**:
- 使用 `target_include_directories()` 而非全局 `include_directories()`
- 使用 `#pragma once` 统一头文件保护
- 定义跨平台DLL导出宏
- 添加 `CMakePresets.json` 支持多配置
- 版本要求合理（Python >=3.13 而非 >=3.14）
- 添加tests/和examples/可选构建目标

---

## 4. 反模式清单

| 反模式 | 后果 | 正确做法 |
|-------|------|---------|
| 后端文件重复实现RAII逻辑 | 代码重复、bug修复需改多处 | 桥接模式提取到common层 |
| FFI边界用int传递uint32_t | 大数值静默截断 | int64_t传递+static_cast |
| 全局include_directories | 头文件路径泄漏、污染 | target_include_directories |
| 构造函数不检查alloc失败 | 空指针解引用→崩溃 | nullptr检查+抛bad_alloc |
| Python析构抛异常 | 解释器terminate | try/except包裹+注释说明 |
| #ifdef NDEBUG 条件编译用反 | Debug模式无检查 | 关键警告无条件输出 |
| push_op空实现丢弃参数 | 静默功能失效 | 正确转发所有参数 |
| 没有CommandContext RAII | 忘记synchronize→硬件状态不一致 | 析构自动同步 |
| git add . 提交 | 混入临时文件/无关修改 | 显式指定每个文件 |

---

## 5. 质量检查清单

- [ ] C ABI函数使用 `extern "C"` 声明，无C++类型跨边界
- [ ] RAII类正确实现移动语义（移动后原对象empty）
- [ ] 资源分配后检查nullptr/错误码
- [ ] FFI注册层所有数值参数用int64_t
- [ ] Python析构函数不抛异常（有try/except+注释）
- [ ] 后端文件仅实现extern "C"原语，无宿主逻辑
- [ ] 类型安全API层接受类型化对象/枚举，自动转换
- [ ] CMake使用target_*命令而非全局命令
- [ ] 有C++和Python测试覆盖核心逻辑
- [ ] 有examples展示用法
- [ ] 泄漏检测在stub模式无条件输出警告
- [ ] 提交遵循Conventional Commits，原子化

---

## 6. 迁移验证步骤

1. **编译验证**: stub模式编译通过无警告
2. **C++测试**: test_buffer所有用例通过
3. **Python测试**: pytest所有用例通过
4. **兼容性验证**: 原有API调用方式继续工作
5. **泄漏检查**: 正常流程退出stub无泄漏警告
6. **Double-free安全**: 重复reset/__del__不崩溃
7. **代码重复检查**: grep确认Buffer宿主逻辑只在一处

---

## 7. 参考案例

- **demo-ffi**: TVM FFI官方示例，展示基本绑定模式
- **npu-ffi (优化后)**: 本优化结果，完整展示五层架构
- **TVM FFI源码**: tvm/ffi/ 目录下的核心框架实现
