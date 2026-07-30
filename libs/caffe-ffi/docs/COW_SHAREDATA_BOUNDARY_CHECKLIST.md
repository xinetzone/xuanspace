---
title: "COW 共享机制边界条件检查清单"
date: 2026-07-31
category: testing
audience: developer
source: tests/cpp/test_blob_zerocopy.cpp ShareDataRefCount 测试套件 (15 用例)
related:
  - ZEROCOPY_ONBOARDING_CHECKLIST.md
  - SPLIT_COW_PHASE2_DESIGN_DRAFT.md
  - ZEROCOPY_PHASE1_RETROSPECTIVE_20260731.md
---

# COW 共享机制边界条件检查清单

> 基于 15 个 `ShareDataRefCount` 单元测试用例生成，覆盖 ShareData/ShareDiff/COW 的引用计数异常路径。
> 适用于 Split N≥2 COW 功能的 Code Review 自检、提交前检查、回归测试用例设计。

---

## A. ShareData 基本契约

| # | 检查项 | 对应测试 | 风险等级 | 判定标准 |
|---|--------|---------|---------|---------|
| A1 | **自共享是否幂等？** `blob->ShareData(blob)` 不崩溃，指针不变 | `SelfShareIsIdempotent` | 低 | `cpu_data()` 指针相等，`SharesDataWith(self)` 为 true |
| A2 | **链式共享 A→B→C 是否三向指针一致？** | `ChainShareThreeWay` | 高 | `cpu_data()` 三者指针相等，`SharesDataWith` 传递性成立 |
| A3 | **多次相同共享是否幂等？** 对同一对 Blob 多次调用 `ShareData` | `RepeatedShareDataIsIdempotent` | 中 | 每次调用后指针不变，数据值不变 |
| A4 | **双向共享是否幂等？** A→B 后再 B→A | `BidirectionalShareIsIdempotent` | 中 | 指针不变，不产生循环引用导致泄漏 |
| A5 | **不同形状 Blob 间共享是否正常？** | `ShareDataWithDifferentShapes` | 中 | shape 跟随源，`count()`/`num_axes()` 与源一致 |
| A6 | **零元素 Blob 共享是否崩溃？** shape {0} | `ShareDataZeroElementBlob` | 低 | `count() == 0`，`SharesDataWith` 为 true |

## B. 共享关系生命周期

| # | 检查项 | 对应测试 | 风险等级 | 判定标准 |
|---|--------|---------|---------|---------|
| B1 | **重共享是否覆盖旧引用？** B 先共享 A，再共享 C | `ReShareOverwritesPrevious` | 高 | B 与 A 断开，与 C 共享；A 数据不受影响 |
| B2 | **Reshape 是否打破共享？** | `ReshapeBreaksShare` | 高 | `SharesDataWith` 变为 false，源数据不受影响 |
| B3 | **链中段 Reshape 是否保护端点？** A→B→C，Reshape B | `ChainMiddleReshapePreservesEndpoints` | 高 | A 和 C 仍共享原数据，B 断开；数据值正确 |
| B4 | **源销毁后目标是否仍有效？** A→B，A 析构 | `SourceDestroyedDataStillValid` | 高 | 目标数据值不变，指针不悬空（refcount 保护） |
| B5 | **旧 Tensor 是否释放？** ShareData 覆盖旧独立 tensor | `OldTensorReleasedAfterShare` | 中 | `g_total_allocated_bytes` 不增加（或减少），无泄漏 |

## C. COW 触发与隔离

| # | 检查项 | 对应测试 | 风险等级 | 判定标准 |
|---|--------|---------|---------|---------|
| C1 | **COW 后共享是否重新建立？** A→B → B COW → B→C | `ShareDataAfterCOW` | 高 | C 与 B 共享，与 A 不共享；A 数据不变 |
| C2 | **COW 仅影响写入者？** A→B, A→C, B COW | `COWOnlyAffectsMutator` | 高 | B 与 A 断开，C 与 A 仍共享；A 数据不变 |
| C3 | **const 访问是否触发 COW？** `cpu_data()` 非 const 不自动触发 COW | 现有 `COWTest` 套件 | 高 | 仅 `cpu_mutable_data()` 触发 COW，`cpu_data()` 返回共享指针 |
| C4 | **COW 后数据是否完全隔离？** 写入者修改不影响原共享者 | 现有 `DataIsolationAfterCOW` | 高 | 逐元素 `EXPECT_NEAR` 验证，写入者新值与共享者旧值均正确 |

## D. Data/Diff 独立性与交叉共享

| # | 检查项 | 对应测试 | 风险等级 | 判定标准 |
|---|--------|---------|---------|---------|
| D1 | **ShareData 是否不影响 diff？** | `ShareDiffIndependentOfShareData` | 中 | data 共享、diff 独立；各自值正确 |
| D2 | **ShareDiff 是否不影响 data？** | `ShareDiffIndependentOfShareData` | 中 | diff 共享、data 独立；各自值正确 |
| D3 | **交叉共享是否隔离？** data 与 A 共享、diff 与 B 共享 | 现有 `DataDiffCrossShare` | 中 | `SharesDataWith(A)` 为 true，`SharesDiffWith(B)` 为 true，交叉关系为 false |

## E. 边界值与异常输入

| # | 检查项 | 对应测试 | 风险等级 | 判定标准 |
|---|--------|---------|---------|---------|
| E1 | **源 tensor 未定义时是否触发 CHECK？** | `ShareDataWithUndefinedTensorFails` | 低 | Debug 构建中 CHECK 失败；Release 中行为一致不崩溃 |
| E2 | **nullptr 源是否触发 CHECK？** | `ShareData` 实现中 `CAFFE_FFI_CHECK_TYPE(other != nullptr)` | 低 | Debug 构建中 CHECK 失败 |
| E3 | **单元素 Blob 共享是否正常？** shape {1} | 隐式覆盖（所有测试使用小 shape） | 低 | 指针相等、数据值正确 |
| E4 | **大 shape Blob 共享是否正常？** shape {100, 100} 等 | `OldTensorReleasedAfterShare` | 低 | 指针相等、数据值正确、无泄漏 |

---

## 快速排查决策表

按 N=2 测试失败的**症状**快速定位是哪个边界条件被违反：

| 症状 | 最可能违反的检查项 | 优先排查的测试 |
|------|-------------------|---------------|
| 指针不相等，`SharesDataWith` 为 false | A2 (链式共享) / B1 (重共享) | `ChainShareThreeWay` |
| 指针相等，但数据不一致 | C3 (const 触发 COW) / C4 (隔离失败) | `COWOnlyAffectsMutator` |
| COW 后 `SharesDataWith` 仍为 true | C1 (COW 未触发) | `ShareDataAfterCOW` |
| COW 后所有共享者都断开 | C2 (COW 影响范围过大) | `COWOnlyAffectsMutator` |
| Reshape 后共享意外保持 | B2 (Reshape 未打破共享) | `ReshapeBreaksShare` |
| 源析构后目标崩溃 | B4 (refcount 未保护) | `SourceDestroyedDataStillValid` |
| 内存持续增长 | B5 (旧 Tensor 未释放) | `OldTensorReleasedAfterShare` |
| 重复 Forward 后行为异常 | A3 (非幂等) / C1 (COW 累积) | `RepeatedShareDataIsIdempotent` |

---

## 使用方式

1. **提交前自检**：逐项对比 A1-E4，在代码变更涉及 ShareData/ShareDiff/cpu_mutable_data 时逐项打勾
2. **Code Review**：审阅者使用 C 类（COW 触发与隔离）和 B 类（生命周期）作为核心审查维度
3. **回归测试**：每次修改 `blob.hpp` 或 `blob.cpp` 后运行完整 `ShareDataRefCount` + `COWTest` 套件：
   ```bash
   ctest --test-dir build -R "ShareDataRefCount|COWTest" --output-on-failure
   ```
4. **CI 门禁**：A 类（基本契约）和 B 类（生命周期）测试应作为 CI 必过项

---

## 测试覆盖矩阵

| 测试用例 | A1 | A2 | A3 | A4 | A5 | A6 | B1 | B2 | B3 | B4 | B5 | C1 | C2 | D1 | D2 | E1 |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SelfShareIsIdempotent | x |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| ChainShareThreeWay |   | x |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| ReShareOverwritesPrevious |   |   |   |   |   |   | x |   |   |   |   |   |   |   |   |   |
| ReshapeBreaksShare |   |   |   |   |   |   |   | x |   |   |   |   |   |   |   |   |
| ShareDataAfterCOW |   |   |   |   |   |   |   |   |   |   |   | x |   |   |   |   |
| ChainMiddleReshapePreservesEndpoints |   |   |   |   |   |   |   |   | x |   |   |   |   |   |   |   |
| SourceDestroyedDataStillValid |   |   |   |   |   |   |   |   |   | x |   |   |   |   |   |   |
| ShareDataWithDifferentShapes |   |   |   |   | x |   |   |   |   |   |   |   |   |   |   |   |
| RepeatedShareDataIsIdempotent |   |   | x |   |   |   |   |   |   |   |   |   |   |   |   |   |
| BidirectionalShareIsIdempotent |   |   |   | x |   |   |   |   |   |   |   |   |   |   |   |   |
| OldTensorReleasedAfterShare |   |   |   |   |   |   |   |   |   |   | x |   |   |   |   |   |
| ShareDataWithUndefinedTensorFails |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | x |
| ShareDiffIndependentOfShareData |   |   |   |   |   |   |   |   |   |   |   |   |   | x | x |   |
| ShareDataZeroElementBlob |   |   |   |   |   | x |   |   |   |   |   |   |   |   |   |   |
| COWOnlyAffectsMutator |   |   |   |   |   |   |   |   |   |   |   |   | x |   |   |   |