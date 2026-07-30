---
title: "从 std::shared_ptr 迁移到侵入式引用计数（ObjectPtr）的代码改造指南"
date: 2026-07-31
category: migration-guide
audience: developer
source: ffi-intrusive-refcount-zerocopy code-pattern (2026-07-31)
related:
  - ffi-intrusive-refcount-zerocopy.md
  - ZEROCOPY_ONBOARDING_CHECKLIST.md
---

# 从 `std::shared_ptr` 迁移到侵入式引用计数（ObjectPtr）的代码改造指南

> 适用场景：项目已使用 TVM FFI Object 体系，现有代码使用 `std::shared_ptr<T>` 管理自定义对象，需要统一迁移到 `ObjectPtr<T>`（侵入式引用计数）。

---

## 迁移全景图

```
阶段0: 审计 → 阶段1: 改对象定义 → 阶段2: 改源文件注册 → 阶段3: 改创建方式
→ 阶段4: 改参数传递 → 阶段5: 改裸指针恢复 → 阶段6: 改 enable_shared_from_this
→ 阶段7: 替换 weak_ptr/自定义删除器 → 阶段8: 编译验证 → 阶段9: 运行验证
```

---

## 步骤 0：审计现有代码（准备阶段）

**目标**：摸清有多少类需要迁移，它们的使用方式。

```bash
grep -rn "std::shared_ptr" --include="*.hpp" --include="*.cpp" src/ include/
grep -rn "shared_ptr<" --include="*.hpp" --include="*.cpp" src/ include/
grep -rn "class.*:.*public" --include="*.hpp" include/
```

列出清单，标注：
- ✅ 可以直接迁移：纯数据持有类，无多继承
- ⚠️ 需要调整：有多继承、自定义删除器、`enable_shared_from_this`
- ❌ 暂不迁移：第三方库类型、标准库类型

---

## 步骤 1：修改对象定义（让类继承 Object）

### Before（std::shared_ptr 版本）

```cpp
// include/foo.hpp
#pragma once
#include <memory>
#include <string>

class Foo {
 public:
  Foo(const std::string& name, int value);
  ~Foo();

  const std::string& name() const { return name_; }
  int value() const { return value_; }
  void set_value(int v) { value_ = v; }

 private:
  std::string name_;
  int value_;
};

using FooPtr = std::shared_ptr<Foo>;
```

### After（侵入式 refcount 版本）

```cpp
// include/foo.hpp
#pragma once
#include <string>
#include "caffe_ffi/common.hpp"

namespace myproject {

class Foo : public caffe_ffi::Object {
 public:
  static constexpr bool _type_mutable = true;  // 必须：标记为可变类型

  Foo(const std::string& name, int value);
  ~Foo() override;  // 析构函数加 override

  const std::string& name() const { return name_; }
  int value() const { return value_; }
  void set_value(int v) { value_ = v; }

  // 必须添加：FFI 类型注册宏
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "myproject.Foo", Foo, caffe_ffi::Object);

 private:
  std::string name_;
  int value_;
};

using FooPtr = caffe_ffi::ObjectPtr<Foo>;

}  // namespace myproject
```

**关键变化**：

| 项目 | Before | After |
|------|--------|-------|
| 基类 | 无（或自定义基类） | `public Object` |
| 析构函数 | `~Foo()` | `~Foo() override` |
| 类型宏 | 无 | `TVM_FFI_DECLARE_OBJECT_INFO_FINAL(...)` |
| 类型别名 | `std::shared_ptr<Foo>` | `ObjectPtr<Foo>` |
| `_type_mutable` | 无 | `static constexpr bool _type_mutable = true;` |

---

## 步骤 2：修改源文件中的 FFI 类型注册

在对应的 `.cpp` 文件中添加 FFI 类型注册：

### Before

```cpp
// foo.cpp
#include "foo.hpp"

Foo::Foo(const std::string& name, int value) : name_(name), value_(value) {}
Foo::~Foo() {}
```

### After

```cpp
// foo.cpp
#include "foo.hpp"

namespace myproject {

Foo::Foo(const std::string& name, int value) : name_(name), value_(value) {}
Foo::~Foo() {}

TVM_FFI_REGISTER_OBJECT(Foo)  // 添加：注册到 FFI 类型系统
    .def("name", &Foo::name)
    .def("value", &Foo::value)
    .def("set_value", &Foo::set_value);

}  // namespace myproject
```

---

## 步骤 3：修改对象创建方式

### Before

```cpp
// shared_ptr 创建
auto foo = std::make_shared<Foo>("hello", 42);
std::shared_ptr<Foo> foo2(new Foo("world", 100));
```

### After

```cpp
// ObjectPtr 创建
#include <tvm/ffi/container/array.h>  // make_object 声明

auto foo = caffe_ffi::make_object<Foo>("hello", 42);   // 推荐：类似 make_shared
caffe_ffi::FooPtr foo2(caffe_ffi::make_object<Foo>("world", 100));
```

**禁止**使用裸 `new` + 手动构造 ObjectPtr。

---

## 步骤 4：修改参数传递方式

### Before（shared_ptr 风格）

```cpp
// 按 const shared_ptr& 传递，避免拷贝控制块
void ProcessFoo(const std::shared_ptr<Foo>& foo) {
  if (!foo) return;
  int v = foo->value();
}

// 按值传递（共享所有权）
void StoreFoo(std::shared_ptr<Foo> foo) {
  owned_foos_.push_back(std::move(foo));
}
```

### After（ObjectPtr 风格）

```cpp
// 内部 C++ 代码：优先传裸指针（不增加 refcount）
void ProcessFoo(const Foo* foo) {
  if (foo == nullptr) return;
  int v = foo->value();
}

// 需要共享所有权时：按 ObjectPtr 传值（拷贝 = 原子 increment，极轻量）
void StoreFoo(FooPtr foo) {
  owned_foos_.push_back(std::move(foo));
}

// FFI 边界（Python/外部调用）：用 const ObjectPtr& + lambda 适配
.def("process_foo", [](Foo* self, const FooPtr& other) {
       self->ProcessFoo(other.get());  // ObjectPtr → const Foo*
     });
```

**原则**：

| 场景 | 推荐参数类型 | 原因 |
|------|------------|------|
| 内部函数，不持有引用 | `const Foo*` / `Foo*` | 零开销，不改动 refcount |
| 需要共享所有权/存储 | `FooPtr`（值传递） | 拷贝 = 原子 inc，move = 零开销 |
| FFI 注册方法参数 | `const FooPtr&` | FFI 系统要求，lambda 内 `.get()` 转裸指针 |
| 可能为空的可选参数 | `const Foo*`（nullable） | nullptr 语义清晰 |

---

## 步骤 5：修改裸指针恢复为智能指针的方式

**这是最关键的差异点**：`std::shared_ptr` 无法安全从裸指针恢复，而 `ObjectPtr` 可以。

### Before（shared_ptr：危险操作）

```cpp
// shared_ptr 从裸指针构造——UB！会创建第二个控制块导致 double-free
void BadCode(Foo* raw_foo) {
  std::shared_ptr<Foo> p(raw_foo);  // DANGER! double-free!
}
```

### After（ObjectPtr：支持从裸指针安全恢复）

```cpp
// ObjectPtr 从裸指针 GetRef——安全！因为 refcount 在对象内部
void SafeCode(Foo* raw_foo) {
  if (raw_foo == nullptr) return;
  FooPtr p = GetRef<ObjectPtr<Foo>>(raw_foo);  // 安全：refcount++
}
```

---

## 步骤 6：修改 enable_shared_from_this 模式

### Before

```cpp
class Bar : public std::enable_shared_from_this<Bar> {
 public:
  void RegisterCallback() {
    auto self = shared_from_this();
    scheduler_->Submit([self]() { self->DoWork(); });
  }
};
```

### After

```cpp
class Bar : public Object {
 public:
  static constexpr bool _type_mutable = true;

  void RegisterCallback() {
    auto self = ObjectPtr<Bar>(this);  // 从 this 直接构造 ObjectPtr（安全）
    scheduler_->Submit([self]() { self->DoWork(); });
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("myproject.Bar", Bar, Object);
};
```

---

## 步骤 7：替换自定义删除器/weak_ptr（如有）

### 自定义删除器

侵入式引用计数**不支持**自定义删除器。析构逻辑统一放在析构函数中：

```cpp
// Before: shared_ptr 自定义删除器
auto fp = std::shared_ptr<FILE>(fopen("a.txt", "r"), &fclose);

// After: 使用 RAII 包装类替代
struct FileCloser { void operator()(FILE* f) const { if (f) fclose(f); } };
using FilePtr = std::unique_ptr<FILE, FileCloser>;
FilePtr fp(fopen("a.txt", "r"));
```

### weak_ptr 替换

```cpp
// Before
std::weak_ptr<Foo> weak_foo = foo;
if (auto locked = weak_foo.lock()) { locked->value(); }

// After
#include <tvm/ffi/weak_ref.h>
tvm::ffi::WeakRef<Foo> weak_foo(foo);
if (auto locked = weak_foo.lock()) { locked->value(); }
```

---

## 步骤 8：全局替换与编译验证

```bash
# 推荐：手动审查每个文件，不使用 sed 全局盲替换
# 原因：shared_ptr 可能用于非 Object 子类（如标准库类型），需逐文件判断
cmake --build build
```

---

## 步骤 9：验证检查清单

| # | 验证项 | 命令/方法 |
|---|--------|----------|
| 1 | 编译通过 | `cmake --build build` 零 error |
| 2 | 无遗留 shared_ptr | `grep -rn "shared_ptr" src/ include/` 结果仅用于第三方/标准库类型 |
| 3 | 单元测试通过 | 运行现有测试套件 |
| 4 | 生命周期测试 | 添加 `SourceOutlivesDestination` / `DestinationOutlivesSource` 类测试 |
| 5 | 无内存泄漏 | 运行 ASAN 或项目内内存计数器验证 |
| 6 | FFI 绑定正常 | Python 端能正常调用新方法 |
| 7 | 性能无回归 | ObjectPtr 拷贝 vs shared_ptr 拷贝：侵入式快约 10-30%（少一次指针跳转） |

---

## 迁移陷阱速查表

| 陷阱 | 症状 | 修复 |
|------|------|------|
| 忘记加 `TVM_FFI_DECLARE_OBJECT_INFO_FINAL` | 编译报 `TypeTraits not specialized` / `storage_enabled_v` 错误 | 添加类型注册宏 |
| 自定义 TypeTraits 与 vendor 冲突 | 编译报 SFINAE 冲突、duplicate specialization | 删除自定义 TypeTraits，使用 vendor 内置 |
| 用 `new Foo()` + ObjectPtr 构造 | 偶发 double-free 或 refcount 异常 | 统一用 `make_object<Foo>(args...)` |
| 在栈上分配 Object 子类 | `ObjectPtr` 引用栈对象后析构导致 UB | Object 子类必须在堆上分配（make_object） |
| 多继承 Object | 编译错误或 RTTI 异常 | TVM FFI Object 使用单继承；多继承需其他基类为接口类 |
| 析构函数不是 virtual | 通过基类指针 delete 子类时资源泄漏 | `~Foo() override;` |
| FFI 方法直接传裸指针 | Python 调用时参数类型不匹配 | FFI 注册用 lambda 适配 ObjectPtr → raw pointer |

---

## 附录 A：最小可运行示例

```cpp
// migrate_example.cpp
#include <tvm/ffi/object.h>
#include <tvm/ffi/function.h>
#include <iostream>

namespace demo {

using namespace tvm::ffi;

class Greeter : public Object {
 public:
  static constexpr bool _type_mutable = true;
  explicit Greeter(std::string msg) : msg_(std::move(msg)) {}
  ~Greeter() override { std::cout << "Greeter destroyed\n"; }

  void Greet(const std::string& name) const {
    std::cout << msg_ << ", " << name << "!\n";
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("demo.Greeter", Greeter, Object);
 private:
  std::string msg_;
};

using GreeterPtr = ObjectPtr<Greeter>;

TVM_FFI_REGISTER_OBJECT(Greeter)
    .def("greet", &Greeter::Greet);

}  // namespace demo

int main() {
  using namespace demo;

  // 步骤 3：make_object 创建
  auto g = make_object<Greeter>("Hello");
  g->Greet("World");

  // 步骤 5：从裸指针安全恢复
  Greeter* raw = g.get();
  GreeterPtr g2 = GetRef<GreeterPtr>(raw);
  g2->Greet("from raw pointer");

  // 模拟源释放后目标仍有效（refcount 保证生命周期）
  {
    auto temp = make_object<Greeter>("Temporary");
    g2 = GetRef<GreeterPtr>(temp.get());
  }  // temp 析构，但 g2 持有 refcount
  g2->Greet("after temp destroyed");  // 安全

  return 0;
}
```

**预期输出**：
```
Hello, World!
Hello, from raw pointer!
Greeter destroyed  (第一个 g 的 "Hello" 对象)
Temporary, after temp destroyed!
Greeter destroyed  (temp 即 g2 指向的对象，main 退出时 g2 析构)
```

---

## 附录 B：make_object 调用示例（步骤 4 参数传递完整示例）

完整代码见独立示例文件：[`make_object_example.cpp`](make_object_example.cpp)

---

## 附录 C：编译/链接类陷阱对应的 CMake 配置片段

### C.1 禁用 Unity Build（对应陷阱 F3）

```cmake
# CMakeLists.txt（顶层）
# 必须在 project() 之后、任何 add_library() 之前设置
# 原因：CMake 4.x 对 MSVC 默认启用 Unity Build，会破坏
#       tvm::ffi::Array<ObjectPtr<T>> 的模板实例化顺序
set(CMAKE_UNITY_BUILD OFF CACHE BOOL "Unity build" FORCE)
```

**当前状态**：已在 [CMakeLists.txt#L6](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi/CMakeLists.txt#L6) 中配置。

### C.2 防止自定义 TypeTraits 冲突（对应陷阱 F2）

**无 CMake 配置可防止**——这是代码层面的问题。但可以通过以下方式在 CI 中检测：

```cmake
# cmake/CompilerConfig.cmake 中添加编译期检测
# 对 tvm-ffi vendor 源码中的 TypeTraits 进行编译期断言
add_compile_definitions(
  TVM_FFI_USE_BUILTIN_TYPETRAITS  # 强制使用 vendor 内置 TypeTraits
)
```

**推荐做法**：在 `common.hpp` 中添加编译期静态断言，防止自定义 TypeTraits 被定义：

```cpp
// common.hpp 中添加（放在所有 include 之后）
#ifdef TVM_FFI_USE_BUILTIN_TYPETRAITS
static_assert(
    tvm::ffi::detail::storage_enabled_v<tvm::ffi::ObjectPtr<tvm::ffi::Object>>,
    "vendor tvm-ffi TypeTraits must be enabled for ObjectPtr<T>");
#endif
```

### C.3 确保 C++17 标准（ObjectPtr 的 SFINAE 要求）

```cmake
# cmake/Options.cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用编译器扩展，确保标准兼容
```

**当前状态**：已在 [Options.cmake#L2-L4](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi/cmake/Options.cmake#L2-L4) 中配置。

### C.4 共享库符号可见性控制（防止 WEAK 符号泄漏）

当 TVM FFI 作为静态库链接时，模板实例化会产生大量 WEAK 符号，导致多副本冲突。通过以下配置隔离：

```cmake
# cmake/CompilerConfig.cmake 或 cmake/TargetBuild.cmake 中添加
if(NOT MSVC)
  target_compile_options(${target_name} PRIVATE
    -fvisibility=hidden           # 默认隐藏所有符号
    -fvisibility-inlines-hidden   # 隐藏内联/模板实例化产生的 WEAK 符号
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(${target_name} PRIVATE
    -Wl,--exclude-libs,ALL       # 排除所有静态库符号
  )
endif()
```

### C.5 编译警告升级为错误（捕获未使用变量/类型不匹配）

```cmake
# cmake/CompilerConfig.cmake
if(MSVC)
  target_compile_options(${target_name} PRIVATE /W3 /WX /utf-8)
else()
  target_compile_options(${target_name} PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)
endif()
```

### C.6 完整 CMake 配置参考

当前项目已有的编译配置（[CompilerConfig.cmake](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi/cmake/CompilerConfig.cmake)）包含：

```cmake
# 已配置项（无需额外操作）
# ✅ Unity Build 禁用：CMakeLists.txt#L6
# ✅ C++17 标准：Options.cmake#L2-L4
# ✅ 编译选项 /W3 /utf-8 (MSVC) / -Wall -Wextra (GCC/Clang)
# ✅ tvm-ffi 依赖查找：Dependencies.cmake (find_package + add_subdirectory fallback)
# ✅ 平台特定兼容：KMP_DUPLICATE_LIB_OK=TRUE (Windows OpenMP 多副本)
```

**需要额外添加的配置（建议）**：

```cmake
# 1. 添加到 cmake/Options.cmake 末尾
# 启用编译期 COW 开关（占位，Phase 2 使用）
option(CAFFE_FFI_ENABLE_COW "Enable Copy-on-Write optimization for N>=2 Split" OFF)
if(CAFFE_FFI_ENABLE_COW)
  add_compile_definitions(CAFFE_FFI_ENABLE_COW)
endif()

# 2. 添加到 cmake/CompilerConfig.cmake 的 caffe_ffi_configure_target() 函数中
# 强制使用 vendor 内置 TypeTraits（防止自定义 TypeTraits 冲突）
target_compile_definitions(${target_name} ${ARG_VISIBILITY}
  TVM_FFI_USE_BUILTIN_TYPETRAITS
)
```