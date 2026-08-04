---
id: caffe-ffi-docs-index
title: caffe-ffi 文档索引
---
# caffe-ffi 文档中心

本目录包含 caffe-ffi 项目的所有技术文档、设计草稿、性能报告、复盘总结等。

## 目录结构

```
docs/
├── README.md              # 本文件（文档索引）
├── setup/                 # 构建指南与环境配置
├── checklists/            # 检查清单（上线/重构/迁移等）
├── design/                # 设计文档与API设计草稿
├── memory/                # 内存日志相关报告
├── migration/             # 迁移指南（shared_ptr→intrusive_refcount等）
├── performance/           # 性能优化报告与分析
├── plans/                 # 开发计划与技术规范
├── retrospectives/        # 项目复盘报告
├── summaries/             # 任务总结与简报
└── testing/               # 测试指南与规范
```

---

## 🔧 setup/ — 构建指南与环境配置

构建环境配置、Protobuf 兼容性、跨平台构建指南。

| 文档 | 说明 |
|------|------|
| [BUILD_VERIFICATION_REPORT_20260731.md](setup/BUILD_VERIFICATION_REPORT_20260731.md) | 构建验证报告（2026-07-31） |
| [CROSS_MACHINE_BUILD_SETUP_GUIDE.md](setup/CROSS_MACHINE_BUILD_SETUP_GUIDE.md) | 跨机器构建设置指南 |
| [WSL2_BUILD_SETUP_GUIDE.md](setup/WSL2_BUILD_SETUP_GUIDE.md) | WSL2 构建设置指南 |
| [PROTOBUF_COMPATIBILITY_AND_COMPILER_FLAGS.md](setup/PROTOBUF_COMPATIBILITY_AND_COMPILER_FLAGS.md) | Protobuf 兼容性与编译 flags 配置 |
| [PROTOBUF_COMPATIBILITY_AND_COMPILER_FLAGS_CHANGELOG_20260801.md](setup/PROTOBUF_COMPATIBILITY_AND_COMPILER_FLAGS_CHANGELOG_20260801.md) | Protobuf 兼容性变更日志（2026-08-01） |
| [ASAN_REPORT_READING_GUIDE.md](setup/ASAN_REPORT_READING_GUIDE.md) | ASan 报告堆栈解读指南 |
| [ASAN_VERIFICATION_REPORT_20260804.md](setup/ASAN_VERIFICATION_REPORT_20260804.md) | ASan 内存安全验证报告（2026-08-04） |

---

## ✅ checklists/ — 检查清单

重构、迁移、上线前的标准化检查清单。

| 文档 | 说明 |
|------|------|
| [COW_SHAREDATA_BOUNDARY_CHECKLIST.md](checklists/COW_SHAREDATA_BOUNDARY_CHECKLIST.md) | COW ShareData 边界检查清单 |
| [FFI_ZEROCOPY_REFACTOR_CHECKLIST.md](checklists/FFI_ZEROCOPY_REFACTOR_CHECKLIST.md) | FFI 零拷贝重构检查清单 |
| [ZEROCOPY_ONBOARDING_CHECKLIST.md](checklists/ZEROCOPY_ONBOARDING_CHECKLIST.md) | 零拷贝新手上路检查清单 |

---

## 🎨 design/ — 设计文档

架构设计草稿、API 设计、模式萃取等。

| 文档 | 说明 |
|------|------|
| [SPLIT_COW_PHASE2_DESIGN_DRAFT.md](design/SPLIT_COW_PHASE2_DESIGN_DRAFT.md) | Split COW Phase 2 设计草稿 |
| [SPLIT_COW_PHASE3_DESIGN_DRAFT.md](design/SPLIT_COW_PHASE3_DESIGN_DRAFT.md) | Split COW Phase 3 设计草稿 |
| [SPLIT_ZEROCOPY_DESIGN_DRAFT.md](design/SPLIT_ZEROCOPY_DESIGN_DRAFT.md) | Split 零拷贝设计草稿 |
| [SETSHAPEONLY_API_DESIGN.md](design/SETSHAPEONLY_API_DESIGN.md) | SetShapeOnly API 设计文档 |
| [FFI_ZEROCOPY_PATTERN_EXTRACTION.md](design/FFI_ZEROCOPY_PATTERN_EXTRACTION.md) | FFI 零拷贝模式萃取 |
| [caffe_slim_zerocopy_refactor_draft.md](design/caffe_slim_zerocopy_refactor_draft.md) | Caffe Slim 零拷贝重构草稿 |
| [INPLACE_MEMORY_SAFETY_STANDARD.md](design/INPLACE_MEMORY_SAFETY_STANDARD.md) | In-place 操作内存安全规范 |

---

## 💾 memory/ — 内存日志

内存日志相关报告与分析。

| 文档 | 说明 |
|------|------|
| [memory-logging-report.md](memory/memory-logging-report.md) | 内存日志报告 |

---

## 🔄 migration/ — 迁移指南

技术栈迁移、API 迁移指南。

| 文档 | 说明 |
|------|------|
| [SHARED_PTR_TO_INTRUSIVE_REFCOUNT_MIGRATION.md](migration/SHARED_PTR_TO_INTRUSIVE_REFCOUNT_MIGRATION.md) | shared_ptr → intrusive_refcount 迁移指南 |

---

## ⚡ performance/ — 性能优化

性能测试报告、优化前后对比分析。

| 文档 | 说明 |
|------|------|
| [OPTIMIZATION_REPORT.md](performance/OPTIMIZATION_REPORT.md) | 优化报告总览 |
| [P0_OPTIMIZATION_ADDENDUM_20260729.md](performance/P0_OPTIMIZATION_ADDENDUM_20260729.md) | P0 优化补充说明（2026-07-29） |
| [P1_OPTIMIZATION_REPORT_20260729.md](performance/P1_OPTIMIZATION_REPORT_20260729.md) | P1 优化报告（2026-07-29） |
| [P2B_SPLIT_PERFORMANCE_REPORT.md](performance/P2B_SPLIT_PERFORMANCE_REPORT.md) | P2B Split 性能报告 |
| [PHASE2_VS_PHASE1_PERFORMANCE_ANALYSIS.md](performance/PHASE2_VS_PHASE1_PERFORMANCE_ANALYSIS.md) | Phase 2 vs Phase 1 性能对比分析 |

---

## 📋 plans/ — 计划与规范

开发计划、技术规范、图变换设计。

| 文档 | 说明 |
|------|------|
| [ACTIVATION_PERF_MONITORING_SPEC.md](plans/ACTIVATION_PERF_MONITORING_SPEC.md) | 激活函数性能监控规范 |
| [BACKWARD_LOGGING_PLAN.md](plans/BACKWARD_LOGGING_PLAN.md) | 反向传播日志计划 |
| [INSERT_SPLITS_GRAPH_TRANSFORM.md](plans/INSERT_SPLITS_GRAPH_TRANSFORM.md) | InsertSplits 图变换设计 |

---

## 📝 retrospectives/ — 复盘报告

各阶段项目复盘、经验总结。

| 文档 | 说明 |
|------|------|
| [A3A5_COW_MIGRATION_RETROSPECTIVE_20260801.md](retrospectives/A3A5_COW_MIGRATION_RETROSPECTIVE_20260801.md) | A3/A5 COW 迁移复盘（2026-08-01） |
| [TS31-B4_COW_PROMOTION_BUG_FIXES_20260804.md](retrospectives/TS31-B4_COW_PROMOTION_BUG_FIXES_20260804.md) | TS31-B4 COW 推广两个核心 Bug 修复复盘（2026-08-04） |
| [BUILD_COMPATIBILITY_FIXES_RETROSPECTIVE_20260801.md](retrospectives/BUILD_COMPATIBILITY_FIXES_RETROSPECTIVE_20260801.md) | 构建兼容性修复复盘（2026-08-01） |
| [SOFTMAX_LOSS_BACKWARD_TEST_RETROSPECTIVE_20260801.md](retrospectives/SOFTMAX_LOSS_BACKWARD_TEST_RETROSPECTIVE_20260801.md) | SoftmaxLoss 反向测试复盘（2026-08-01） |
| [SPLIT_COW_PHASE3_RETROSPECTIVE_20260731.md](retrospectives/SPLIT_COW_PHASE3_RETROSPECTIVE_20260731.md) | Split COW Phase 3 复盘（2026-07-31） |
| [ZEROCOPY_PHASE1_RETROSPECTIVE_20260731.md](retrospectives/ZEROCOPY_PHASE1_RETROSPECTIVE_20260731.md) | 零拷贝 Phase 1 复盘（2026-07-31） |
| [memory-logging-retrospective.md](retrospectives/memory-logging-retrospective.md) | 内存日志复盘 |

---

## 📊 summaries/ — 总结与简报

任务执行总结、团队分享、产品简报。

| 文档 | 说明 |
|------|------|
| [BUILD_COMPATIBILITY_FIXES_FINAL_SUMMARY_20260801.md](summaries/BUILD_COMPATIBILITY_FIXES_FINAL_SUMMARY_20260801.md) | 构建兼容性修复最终总结（2026-08-01） |
| [PRODUCT_BRIEFING_BUILD_FIXES_20260801.md](summaries/PRODUCT_BRIEFING_BUILD_FIXES_20260801.md) | 构建修复产品简报（2026-08-01） |
| [TASK_EXECUTION_SUMMARY_20260729.md](summaries/TASK_EXECUTION_SUMMARY_20260729.md) | 任务执行总结（2026-07-29） |
| [TEAM_SHARING_SUMMARY.md](summaries/TEAM_SHARING_SUMMARY.md) | 团队分享总结 |

---

## 🧪 testing/ — 测试指南

测试规范、测试最佳实践。

| 文档 | 说明 |
|------|------|
| [TESTING_GUIDELINES.md](testing/TESTING_GUIDELINES.md) | 测试指南 |

---

## 相关资源

- 项目根目录：[../README.md](../README.md)
- 示例代码：[../examples/](../examples/)
- CMake 配置：[../cmake/](../cmake/)
