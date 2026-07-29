# TVM FFI 零拷贝架构跨模块改造清单

> **生成日期**: 2026-07-29
> **参考实现**: [caffe-ffi](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi)（已完成零拷贝架构优化）
> **方法论**: 基于 caffe-ffi v0.1.0 优化实践的可复用模式沉淀

---

## 一、项目模块现状总览

| 模块 | 路径 | 当前 FFI 模式 | 零拷贝 | 双类模型 | @register_object | 三层日志 | 改造优先级 |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **caffe-ffi** | `vendor/caffe/caffe-ffi/` | TVM FFI 双类+反射 | ✅ | ✅ | ✅ | ✅ | —（已完成） |
| **caffe-slim** | `vendor/caffe/caffe-slim/` | TVM FFI handle 模式 | 🟡 部分 | ❌ | ❌ | ❌ | **P0 高** |
| **npu-ffi** | `libs/npu-ffi/` | TVM FFI int64 句柄 | ❌ | ❌ | ❌ | ❌ | **P0 高** |
| **demo-ffi** | `libs/demo-ffi/` | TVM FFI GlobalDef | ❌ | ❌ | ❌ | 🟡 简单perf log | **P2 低** |
| **caffex** | `vendor/caffe/caffex/` | 原生C++/boost::python | ❌ | ❌ | ❌ | ❌ | P3 不改造 |

> **图例**: ✅ 已实现 / 🟡 部分实现 / ❌ 未实现

---

## 二、各模块详细分析与改造方案

### 2.1 caffe-slim — P0 高优先级

**当前状态**：
- 使用 TVM FFI，但采用 `uintptr_t` 整数句柄模式（`Net_Init` 返回句柄，后续函数传递句柄）
- Blob 读取已实现零拷贝（`Blob_GetData` → `Tensor::FromNDAlloc` + 自定义 allocator → `np.from_dlpack`）
- Blob 写入仍使用 `memcpy`（`Blob_SetData` 内部 `std::memcpy`）
- Python 层手动管理 `__del__` 调用 `Net_Destroy`，无 RAII 安全保障
- 无反射注册对象，所有操作通过全局函数
- 无结构化日志
- 无参数校验（形状检查有，但缺少上下文信息）

**改造方案**：

| 改造项 | 工作量 | 预期收益 | 参考实现 |
|---|---|---|---|
| **双类对象模型** | 中 | 类型安全、自动生命周期、RAII | caffe-ffi 的 `BlobObj`/`Blob` 模式 |
| **@register_object 注册 Net/Blob** | 中 | 消除句柄泄漏风险、Python API 自然 | caffe-ffi `_core.py` |
| **Blob 写入零拷贝** | 低 | 输入数据无需 memcpy，写路径也零拷贝 | caffe-ffi `from_numpy` 直接拿指针 |
| **三层日志** | 中 | 全链路可观测性 | caffe-ffi `log.hpp` + `SetLogLevel` |
| **TVM_FFI_ICHECK 增强** | 低 | 错误上下文（层名、blob名） | caffe-ffi 各层 ICHECK 用法 |
| **Python API 重构** | 中 | 消除手动 handle 管理、`_mod.xxx()` 调用 | caffe-ffi `@register_object` 类方法 |

**关键文件**：
- C++: [_caffe.cpp](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-slim/src/caffe/_caffe.cpp) — FFI 注册层，需重构为双类模式
- Python: [__init__.py](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-slim/python/caffe/__init__.py) — Net 类封装，需改为 @register_object

**具体改造步骤**：
1. 创建 `include/caffe_ffi/net.hpp`、`blob.hpp`、`layer.hpp`（或复用 caffe-slim 已有头文件加 FFI 包装）
2. 实现 `NetObj : public Object`、`BlobObj : public Object` 双类模式
3. 替换 `CpuBlobDataAllocator` 为 caffe-ffi 风格的 `Tensor::FromBlob` 直接引用模式
4. `Blob_SetData` 改为零拷贝：直接检查 Tensor 兼容后设置指针或使用 DLPack
5. Python 层迁移到 `@register_object`，消除手动 handle 和 `__del__`
6. 添加三层日志系统
7. 补充 TVM_FFI_ICHECK 上下文

---

### 2.2 npu-ffi (VTA) — P0 高优先级

**当前状态**：
- VTA NPU 加速器绑定，使用 `int64_t` 传递原始指针（`buffer_alloc` 返回 int64_t，后续操作传 int64_t）
- Buffer 类在 Python 层手动管理 `__del__`/`__exit__` 释放
- 数据访问通过 `cpu_ptr(cmd)` 返回整数指针，需要 numpy ctypes 包装才能访问数据
- **完全无零拷贝**：CPU ↔ VTA SRAM/DRAM 传输通过 `buffer_copy`（memcpy 封装）
- 无 Tensor/DLPack 支持
- 无双类对象模型
- 无结构化日志（有基本 fprintf debug，但无统一控制）
- 错误处理较弱（buffer_alloc 返回 nullptr 不检查）

**改造方案**：

| 改造项 | 工作量 | 预期收益 | 参考实现 |
|---|---|---|---|
| **Buffer 双类模型** | 高 | 类型安全、RAII 自动管理、消除 int64 指针传递 | caffe-ffi `BlobObj`/`Blob` |
| **DLPack Tensor 零拷贝视图** | 高 | numpy/PyTorch 直接访问 VTA DRAM buffer，无需 ctypes 包装 | caffe-ffi `data_tensor` |
| **CommandContext 反射注册** | 中 | 消除 int64 cmd handle 手动管理 | caffe-ffi Net RAII 模式 |
| **buffer_cpu_ptr 零拷贝** | 中 | CPU 可访问指针直接暴露为 numpy 数组 | caffe-ffi `mutable_cpu_data` → numpy |
| **三层日志** | 中 | 调试 NPU 指令序列更方便 | caffe-ffi `log.hpp` |
| **错误处理增强** | 低 | buffer_alloc 失败抛异常、nullptr 检查 | caffe-ffi ICHECK guards |

**关键文件**：
- C++: [ffi_registry.cc](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/npu-ffi/src/vta/ffi_registry.cc) — 当前全是 int64 句柄转发
- C++: [buffer.h](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/npu-ffi/include/npu_ffi/vta/buffer.h) — RAII Buffer 类，需继承 Object
- Python: [buffer.py](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/npu-ffi/python/npu_ffi/vta/buffer.py) — 手动指针管理类
- Python: [_ffi_api.py](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/npu-ffi/python/npu_ffi/vta/_ffi_api.py) — FFI 初始化

**具体改造步骤**：
1. `Buffer` 类改为 `BufferObj : public Object` + `Buffer : public ObjectRef` 双类模式
2. 新增 `Buffer::tensor_view()` 方法，返回 DLPack Tensor 直接引用 buffer 内存
3. Python `Buffer` 使用 `@register_object` 装饰器，添加 `numpy()` 方法做零拷贝访问
4. `buffer_copy` 保留用于 VTA DRAM↔SRAM 传输，但 CPU↔numpy 路径改为零拷贝
5. `CommandContext` 也注册为 FFI 对象，消除 int64 cmd handle
6. 添加三层日志（标签：`[VTA-BUF]`、`[VTA-CMD]`、`[VTA-GEMM]`、`[VTA-DMA]`）
7. 所有 FFI 入口添加 TVM_FFI_ICHECK 检查

**注意事项**：
- VTA buffer 可能位于设备内存（SRAM/DRAM），DLPack 的 `device` 字段需正确设置
- `cpu_ptr()` 可能涉及 cache 同步操作（write_barrier/read_barrier），零拷贝视图需要在 barrier 之后获取
- 需要考虑 VTA 2D load/store 的 stride 信息，DLPack 支持 strides

---

### 2.3 demo-ffi — P2 低优先级

**当前状态**：
- 作为项目模板，展示 TVM FFI 基础用法
- 向量操作使用 `tvm::ffi::Array<double>`，Python 侧 list ↔ Array 转换有拷贝开销
- 无 Tensor/DLPack 支持（数学运算场景，非大块数据传递）
- 有简单的性能日志（`DEMO_FFI_PERF_LOG` 环境变量门控）
- 无双类对象模型（纯全局函数，不需要对象生命周期管理）

**改造方案**：

| 改造项 | 工作量 | 预期收益 | 是否建议 |
|---|---|---|---|
| 添加 Tensor/DLPack 向量操作 | 中 | numpy 零拷贝传递大向量 | 🟡 可选 |
| 双类对象模型 | 低 | demo 不需要复杂对象 | ❌ 不需要 |
| 三层日志统一 | 低 | 与其他模块一致 | 🟡 可选 |
| @register_object | 低 | demo 全是全局函数 | ❌ 不需要 |

**建议**：demo-ffi 作为入门模板，保持简洁即可。如果后续需要支持大规模向量运算，可以参考 caffe-ffi 添加 `vec_add_tensor`/`vec_dot_tensor` 等零拷贝版本的 Tensor 重载函数。

---

### 2.4 caffex（原始 Caffe）— P3 不改造

**当前状态**：
- 原始 BVLC Caffe 仓库，使用 boost::python/pybind11 风格绑定
- 是 caffe-ffi 和 caffe-slim 的上游参考，不直接作为 FFI 绑定使用
- 代码量大（含 GPU、Solver、Data Layer 等完整训练框架）

**建议**：不改造。caffex 是参考实现，caffe-ffi 和 caffe-slim 已经分别在不同层面提供了 TVM FFI 绑定。

---

## 三、通用改造模式（可复用模板）

### 模式 1：双类对象模型（XxxObj + Xxx）

```cpp
// include/xxx_ffi/xxx.hpp
#include <tvm/ffi/object.h>

namespace xxx_ffi {

class XxxObj : public tvm::ffi::Object {
public:
    // 数据成员、C++ 业务逻辑
    void SomeMethod();
    
    TVM_FFI_DECLARE_OBJECT_INFO(XxxObj, tvm::ffi::Object)
};

class Xxx : public tvm::ffi::ObjectRef {
public:
    TVM_FFI_DEFINE_OBJECT_REF_METHODS(Xxx, ObjectRef, XxxObj);
    
    // Python 可见方法
    void SomeMethod() { operator->()->SomeMethod(); }
};

} // namespace xxx_ffi
```

### 模式 2：零拷贝 Tensor 访问

```cpp
// 关键：使用自定义 allocator + Tensor::FromNDAlloc
// 让 numpy 数组直接指向 C++ 内存，持有 ObjectRef 保证生命周期
Tensor data_tensor() const {
    DLTensor dlt;
    dlt.data = data_;                          // 直接指向内部数据
    dlt.device = {kDLCPU, 0};
    dlt.ndim = shape_.size();
    dlt.dtype = {kDLFloat, 32, 1};
    dlt.shape = const_cast<int64_t*>(shape_.data());
    dlt.strides = nullptr;                     // compact
    dlt.byte_offset = 0;
    // 关键：用 ObjectRef 做生命周期托管
    return Tensor::FromDLTensor(dlt, ObjectRef(GetParentRef()));
}
```

### 模式 3：@register_object Python 绑定

```python
from tvm_ffi._ffi_runtime import register_object

@register_object("xxx_ffi.Xxx")
class Xxx(Object):
    @property
    def data_tensor(self):
        return _native_method(self, "data_tensor")()
    
    def reshape(self, shape):
        return _native_method(self, "Reshape")(shape)
```

### 模式 4：三层日志

```
Python 层: logging.getLogger("xxx_ffi") + set_log_level()
FFI 层:   TVM_FFI_DLL_EXPORT_TYPED_FUNC(SetLogLevel, SetLogLevel)
C++ 层:   RAII Logger + 编译期门控 + 组件标签
```

---

## 四、改造优先级排序与排期建议

### 第一阶段：高价值快速收益（1-2 周）

| 任务 | 模块 | 预估工时 |
|---|---|---|
| caffe-slim Blob 写入零拷贝 | caffe-slim | 0.5 天 |
| caffe-slim TVM_FFI_ICHECK 错误增强 | caffe-slim | 0.5 天 |
| npu-ffi buffer_alloc nullptr 检查 + 错误处理 | npu-ffi | 0.5 天 |
| npu-ffi Buffer 基础 numpy 访问（ctypes→DLPack） | npu-ffi | 2 天 |

### 第二阶段：架构对齐（2-3 周）

| 任务 | 模块 | 预估工时 |
|---|---|---|
| caffe-slim 双类对象模型重构 | caffe-slim | 3 天 |
| caffe-slim @register_object Python 重构 | caffe-slim | 2 天 |
| caffe-slim 三层日志 | caffe-slim | 1 天 |
| npu-ffi Buffer 双类模型 | npu-ffi | 3 天 |
| npu-ffi CommandContext 反射注册 | npu-ffi | 2 天 |

### 第三阶段：完善与扩展（按需）

| 任务 | 模块 | 预估工时 |
|---|---|---|
| npu-ffi 三层日志 | npu-ffi | 1 天 |
| npu-ffi DLPack 设备张量支持（VTA SRAM/DRAM） | npu-ffi | 3 天 |
| demo-ffi Tensor 重载（可选） | demo-ffi | 1 天 |

---

## 五、改造验收标准

每个模块改造完成后，需通过以下检查：

- [ ] **功能**: 所有现有测试通过（无回归）
- [ ] **零拷贝验证**: `np.from_dlpack(tensor)` 验证指针一致 + 写后读可见
- [ ] **编译**: 零编译警告（MSVC/GCC/Clang）
- [ ] **内存**: 创建/销毁对象后 `live_count` 回到基线
- [ ] **API 兼容**: 原有 Python API 保持可用（新 API 以 `_tensor` 后缀或新属性提供）
- [ ] **日志**: `set_log_level()` 可控制 C++ 日志输出
- [ ] **错误**: 错误信息包含上下文（对象名/ID/可用选项）
- [ ] **文档**: 公共 API 有 Doxygen 注释，Python 方法有 docstring

---

## 六、参考资源

- **caffe-ffi 优化报告**: [OPTIMIZATION_REPORT.md](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/docs/OPTIMIZATION_REPORT.md)
- **团队分享总结**: [TEAM_SHARING_SUMMARY.md](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/docs/TEAM_SHARING_SUMMARY.md)
- **零拷贝演示代码**: [zero_copy_vs_copy_demo.py](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/examples/zero_copy_vs_copy_demo.py)
- **TVM FFI 官方文档**: `vendor/tvm-ffi/docs/`
- **TVM FFI Tensor 概念**: `vendor/tvm-ffi/docs/concepts/tensor.rst`
