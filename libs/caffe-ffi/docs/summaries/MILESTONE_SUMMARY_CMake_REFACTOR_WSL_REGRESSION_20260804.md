---
title: CMake 重构 WSL 回归里程碑总结
date: 2026-08-04
category: caffe-ffi
task_type: milestone-summary
tags: [caffe-ffi, cmake, regression, milestone, wsl, p3]
status: verified
source: "CMake_REFACTOR_WSL_REGRESSION_LOG_20260804.md"
---

# CMake 重构 WSL 回归里程碑总结

> 基于 1646 个通过用例的完整回归结果，形成本文档，用于更新项目里程碑文档。

## 一、里程碑结论

CMake 原子化重构（**10 个模块化 cmake 文件**）在 WSL docker 环境验证通过，全量 P3 回归 **1646 个用例全部通过（1 skipped）**，确认 **M6（独立项目萃取迁移）** 与 **M9（P3 Backward 训练支持）** 里程碑稳定，为 **P4（优化/扩展）** 奠定可靠基础。

## 二、验证数据

| 指标 | 数值 |
|------|------|
| 通过用例 | **1646 (PASSED)** |
| 跳过用例 | 1 (SKIPPED) |
| 失败用例 | 0 |
| 测试文件 | 43 |
| 耗时 | 15.32s |
| 环境 | caffe-ffi-jupyter 容器 / Python 3.14.6 / cmake 4.4.1 / ninja 1.13.2 / gcc 14.3.0 |
| 宏 | CAFFE_FFI_ENABLE_COW=1, CAFFE_FFI_ENABLE_COW_PHASE3=1 |

## 三、关键要点

1. **CMake 重构验证通过**：10 个模块化 cmake 文件（Tests/WindowsDllCopy/TargetBuild/ProtoCompile/Options/Install/DetectOpenBLAS/DetectBLAS/Dependencies/CompilerConfig）构建与运行正常。
2. **P3 Backward 全量回归稳定**：43 个测试文件覆盖 19 类层 Backward 数值梯度、解析梯度、E2E 训练，全部通过。
3. **COW 机制 + lazy allocation 正常**：COW_PHASE3 宏编译进 `_caffe_ffi.so`，N≥16 时 Split 层触发 `SetShapeOnly` 延迟分配。
4. **P4 前置就绪**：性能优化（BLAS/多线程/COW 推广）在稳定基础上展开。

## 四、里程碑影响

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| M6（独立项目萃取迁移） | ✅ 验证通过 | CMake 原子化重构构建在 WSL 验证 |
| M7（COW 零拷贝共享） | ✅ 稳定 | COW + Phase3 全量回归通过 |
| M8（InsertSplits 图变换） | ✅ 稳定 | 25 层训练验证通过 |
| M9（P3 Backward 训练支持） | ✅ 全量回归通过 | 1646 用例覆盖 19 类层 Backward |
| P4（优化/扩展） | 🔄 规划中 | 基础已验证，可进入性能优化 |

## 五、关联文档

- 详细逐用例日志：[CMake_REFACTOR_WSL_REGRESSION_LOG_20260804.md](../setup/CMake_REFACTOR_WSL_REGRESSION_LOG_20260804.md)
- 任务记录：[tasks.md#Task18](../../../../../../.trae/specs/caffe-ffi-tvm-integration/tasks.md)
- P4 路线图：[p4-roadmap.md](../../../../../../.trae/specs/caffe-ffi-tvm-integration/p4-roadmap.md)