---
title: "CMake 配置与零拷贝测试构建验证报告（实测）"
date: 2026-07-31
session: sc-20260731-split-zerocopy
tags: [cmake, build, test, zerocopy, objectptr, migration]
source: .temp/debug.log (2026-07-31 PowerShell 实测)
---

# CMake 配置与零拷贝测试构建验证报告（实测）

## 执行环境

| 项目 | 值 |
|------|-----|
| OS | Windows 11 |
| Shell | PowerShell 5 (py314 conda) |
| 编译器 | MSVC 19.50.35717.0 (VS 18 Insiders) |
| CMake | 3.31.2 (conda py314) |
| 构建系统 | Ninja |
| Python | 3.14.3 (conda py314) |
| Protobuf | 33.5.0 |
| 构建类型 | Release |

## 构建结果总览

| 步骤 | 状态 | 耗时 |
|------|------|------|
| Step 0: 环境诊断 | ✅ 通过 | — |
| Step 1: 清理 CMake 缓存 | ✅ 通过 | — |
| Step 2: CMake 配置 | ✅ 通过 | 4.5s |
| Step 3: 构建 (34 目标) | ✅ 通过 | — |
| Step 4: 拷贝 DLL | ✅ 通过 | — |
| Step 5: C++ 单元测试 | ✅ 66/66 通过 | 876ms |

## 测试结果详情

### 按测试套件统计

| 测试套件 | 用例数 | 通过 | 失败 | 总耗时 | 平均耗时 |
|----------|--------|------|------|--------|----------|
| **BlobTest** | 23 | 23 | 0 | 51.08ms | 2.22ms |
| **ZeroCopyTest** | 14 | 14 | 0 | 1.96ms | 0.14ms |
| **ObjectPtrMigration** | 12 | 12 | 0 | 0.27ms | 0.02ms |
| **NetTest** | 17 | 17 | 0 | 822.17ms | 48.36ms |
| **合计** | **66** | **66** | **0** | **875.93ms** | — |

### Top 5 最慢测试

| 排名 | 测试 | 耗时 | 原因 |
|------|------|------|------|
| 1 | `NetTest.LayerByNameNotFoundThrows` | 281.86ms | 异常路径开销 |
| 2 | `NetTest.BlobByNameNotFoundThrows` | 270.39ms | 异常路径开销 |
| 3 | `NetTest.UnknownLayerTypeThrows` | 269.02ms | 异常路径开销 |
| 4 | `BlobTest.NegativeDimensionThrows` | 45.28ms | 异常路径开销 |
| 5 | `BlobTest.DefaultConstructor` | 4.78ms | 首次分配 |

> 慢测试均为异常路径测试（`.Throws`），异常抛出/捕获本身有显著开销，属正常现象。

## 零拷贝性能日志摘要

### N=1 零拷贝路径（ZeroCopyTest.SplitN1ZeroCopyViaNet）

```
[SPLIT-PERF] split1 Reshape: num_top=1 count=24 elem_size=4B
  bytes_copied_per_fwd=0B reshape_time=0.0058ms net_alloc=96B zerocopy_n1=yes

[SPLIT-PERF] split1 Reshape: num_top=1 count=24 elem_size=4B
  bytes_copied_per_fwd=0B reshape_time=0.0023ms net_alloc=0B zerocopy_n1=yes

[SPLIT-PERF] split1 Forward(N=1 ZEROCOPY): count=24 shared_bytes=96B
  share_time=3.2us data_ptr_equal=yes was_already_shared=no
  memcpy_saved=96B (zero-copy path)
```

### N=1 数据正确性（ZeroCopyTest.SplitN1DataCorrectnessThroughForward）

```
[SPLIT-PERF] split1 Forward(N=1 ZEROCOPY): count=120 shared_bytes=480B
  share_time=3.3us data_ptr_equal=yes was_already_shared=no
  memcpy_saved=480B (zero-copy path)
```

### N=2 传统 memcpy 路径（ZeroCopyTest.SplitN2StillCopiesData）

```
[SPLIT-PERF] split1 Forward(N=2): count=24 total_copied=192B
  total_memcpy_time=0.0004ms avg_per_copy=0.2us
  min_copy=0.1us max_copy=0.1us throughput=0.447035GB/s num_copies=2
```

## 关键验证结论

| 验证项 | 结果 | 证据 |
|--------|------|------|
| N=1 零拷贝生效 | ✅ | `data_ptr_equal=yes`, `memcpy_saved=96B/480B` |
| N=1 bytes_copied_per_fwd=0 | ✅ | `bytes_copied_per_fwd=0B` |
| N=2 仍走 memcpy | ✅ | `total_copied=192B`, `data_ptr` 不相等 |
| ShareData 指针共享 | ✅ | 14 个 ZeroCopyTest 全部通过 |
| ObjectPtr 引用计数 | ✅ | 12 个 ObjectPtrMigration 全部通过 |
| 构建零警告 | ✅ | `/WX` 下 34 个目标全部编译通过 |
| DLL 自拷贝修复 | ✅ | 从 `build/python/caffe_ffi/` 正确拷贝到 `build/` |
| py314 动态发现 | ✅ | 自动定位 `D:\Users\xinzo\anaconda3\envs\py314` |

## CMake 配置验证

| 配置项 | 状态 | 说明 |
|--------|------|------|
| `TVM_FFI_USE_BUILTIN_TYPETRAITS` | ✅ 生效 | 无 TypeTraits 冲突 |
| `/WX` (MSVC) | ✅ 生效 | 34 目标零警告编译 |
| `-fvisibility=hidden` (GCC/Clang) | N/A | Windows 构建不适用 |
| `CAFFE_FFI_ENABLE_COW` | OFF (默认) | Phase 2 预留 |
| OpenBLAS 检测 | ✅ 已修复 | 见下方附录 |

## 已知问题

| 问题 | 严重程度 | 状态 |
|------|----------|------|
| ZLIB 未检测到 | 低 | 不影响核心功能 |

## 测试覆盖分析

### 已覆盖场景

| 场景 | 测试 | 套件 |
|------|------|------|
| ShareData 基本功能 | `ShareDataMakesPointersEqual` | ZeroCopyTest |
| ShareDiff 基本功能 | `ShareDiffMakesDiffPointersEqual` | ZeroCopyTest |
| 共享后 Shape 保持 | `ShareDataPreservesShape` | ZeroCopyTest |
| 共享后双向写入可见 | `ShareDataMutationVisibleToBoth` | ZeroCopyTest |
| 不同 Blob 不共享 | `SharesDataWithFalseForDifferentBlobs` | ZeroCopyTest |
| Reshape 打破共享 | `ReshapeBreaksShare` | ZeroCopyTest |
| 自共享无操作 | `ShareDataFromSelfIsNoop` | ZeroCopyTest |
| 源长于目标生命周期 | `RefcountingSourceOutlivesDestination` | ZeroCopyTest |
| 目标长于源生命周期 | `RefcountingDestinationOutlivesSource` | ZeroCopyTest |
| 多次共享幂等 | `ShareDataMultipleTimesIdempotent` | ZeroCopyTest |
| Split N=1 端到端 | `SplitN1ZeroCopyViaNet` | ZeroCopyTest |
| Split N=1 数据正确性 | `SplitN1DataCorrectnessThroughForward` | ZeroCopyTest |
| Split N=2 memcpy | `SplitN2StillCopiesData` | ZeroCopyTest |
| 共享不泄漏 Blob | `LiveBlobCountStableAcrossShareData` | ZeroCopyTest |
| ObjectPtr 拷贝构造 | `CopyIncreasesRefcount` | ObjectPtrMigration |
| ObjectPtr 容器所有权 | `RegistryHoldsOwnershipAfterSourceOutOfScope` | ObjectPtrMigration |
| ObjectPtr 多次注册 | `MultipleRegistrationsShareSameObject` | ObjectPtrMigration |
| ObjectPtr 清空保留 | `RegistryClearPreservesOriginalObject` | ObjectPtrMigration |
| ObjectPtr 移动语义 | `MoveDoesNotIncreaseRefcount` | ObjectPtrMigration |
| ObjectPtr 拷贝恢复 | `CopyConstructorSharesPointerAndData` | ObjectPtrMigration |
| ObjectPtr 源析构安全 | `CopySurvivesSourceDestruction` | ObjectPtrMigration |
| ObjectPtr const& 传参 | `ConstRefParameterDoesNotModifyOriginal` | ObjectPtrMigration |
| ObjectPtr 值传递 | `ValuePassingForOwnershipTransfer` | ObjectPtrMigration |
| ObjectPtr 批量操作 | `VectorOfObjectPtrsBulkOperations` | ObjectPtrMigration |
| ObjectPtr 空指针 | `NullObjectPtr` | ObjectPtrMigration |
| ObjectPtr reset | `ResetReleasesOwnership` | ObjectPtrMigration |

### 建议补充的覆盖场景

| 优先级 | 场景 | 理由 |
|--------|------|------|
| **高** | `ShareData` 和 `ShareDiff` 分别来自不同源 | 当前测试 data/diff 均来自同一源；Split 场景下可能 data 来自 bottom、diff 来自不同 Blob |
| **高** | 共享后 Reshape 源 Blob | 验证 Reshape 源不会破坏已共享的目标数据 |
| **高** | 连续多次 Forward（N=1） | 验证重复调用不累积 refcount 泄漏 |
| **中** | COW 触发（`cpu_mutable_data`/`cpu_mutable_diff`） | Phase 2 COW 功能的前置测试，当前仅有代码无测试 |
| **中** | 空 Blob（count=0）共享 | 边界情况：未分配内存的 Blob 执行 ShareData |
| **中** | `set_data` 后共享关系 | 验证 `set_data` 是否打破共享关系 |
| **低** | 多个 Split 层同一 Net | 多 Split 层的 refcount 互不干扰 |
| **低** | 大数据量性能回归 | 非功能测试，但可作为 CI 性能基线 |

> **关于数据类型**：当前 Blob 的 `cpu_data()`/`cpu_diff()` 固定返回 `float*`，底层 Tensor 的 dtype 固定为 `float32`。`ShareData`/`ShareDiff` 在 Tensor 级别操作，不关心数据类型，因此不存在不同数据类型零拷贝验证的需求。未来若 Blob 支持模板化 dtype（如 `Blob<double>`），需补充对应测试。

## 附录：BLAS/OpenBLAS 检测修复（2026-07-31）

### 问题描述

CMake 配置时输出 `BLAS/OpenBLAS not found`，即使 OpenBLAS 已通过 `conda install -c conda-forge libopenblas` 安装在 `py314` 环境中。

### 验证结果

```
Found OpenBLAS: D:/Users/xinzo/anaconda3/envs/py314/Library/lib/libopenblas.lib
OpenBLAS include: D:/Users/xinzo/anaconda3/envs/py314/Library/include/openblas
```

### 根因分析

`cmake/DetectBLAS.cmake` 存在三个跨平台兼容问题：

1. **头文件搜索路径不匹配**：`find_path` 的 `PATH_SUFFIXES` 仅包含 `include include/openblas`，但 Windows conda 将头文件放在 `Library/include/openblas/` 下（如 `Library/include/openblas/cblas.h`）
2. **库名不匹配**：Phase 1 搜索 `openblas openblasp openblas.so.0`，其中 `.so.0` 是 Linux 专用名。Windows conda 提供的是 `libopenblas.lib` 和 `openblas.lib`
3. **环境前缀不一致**：`CONDA_PREFIX` 可能指向 base conda 环境，而 OpenBLAS 安装在 `py314` 子环境中

### 修复内容

修改 `cmake/DetectBLAS.cmake`，将平台差异集中到 `if(WIN32)` 分支：

| 修复项 | 修改前（统一） | 修改后（Windows） | 修改后（Linux/macOS） |
|--------|---------------|-------------------|----------------------|
| 头文件搜索路径 | `include include/openblas` | `include include/openblas Library/include Library/include/openblas` | `include include/openblas` |
| 库名 | `openblas openblasp openblas.so.0` | `libopenblas openblas` | `openblas openblasp openblas.so.0` |
| 库搜索路径 | `lib lib64` | `lib lib64 Library/lib Library/bin` | `lib lib64` |
| 环境前缀发现 | `CONDA_PREFIX` + `CMAKE_PREFIX_PATH` + `Python_SITEARCH` | 新增：从 `Protobuf_INCLUDE_DIR` 反推正确的 conda 环境前缀 | 同前 |
| 错误提示 | 统一提示 `libopenblas-dev` / `.so` | `conda install -c conda-forge libopenblas` | `libopenblas-dev` / `libopenblas` |