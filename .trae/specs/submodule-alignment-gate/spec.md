# 子模块指针—文档一致性门禁 Spec（提案）

> **状态：提案（Specify 阶段产物，待批准）**
> 来源：[xuanspace 十月治理收口里程碑复盘](A-04 行动项)，2026-10-29。
> 本文件是给 xuanspace 子项目的正式变更提案，批准后进入 Plan/Implement。

## 文档元数据

```yaml
id: "submodule-alignment-gate"
version: "0.1.0"
status: proposal
created: 2026-10-29
```

## Why

xuanspace 现有「声明—实现一致性」门禁（`scripts/check_declaration_consistency.py`，
P0-4 落地）覆盖六类断言（A. CLI 命令清单 / B. 子项目清单 / C. 目录名口径 /
D. vendor 清单 / E. 子模块声明 / F. 项目索引），但**子模块相关断言（D/E/F）
只比对 `.gitmodules` 的路径与名称，不覆盖子模块指针（gitlink SHA）状态**。
`scripts/declaration-snapshot.json` 的 manifest 仅含 `cli_commands / lib_projects / submodules`
三个字段，无指针字段。

由此产生三类未被任何门禁拦截的漂移：

1. **指针漂移（上游快进）**：2026-10-13 提交 `81c29a6` 手动同步
   `vendor/tvm-ffi` 至 `0932956`——上游 main 已快进 **35 个提交**，指针更新依赖人工发现。
2. **裸指针变更提交**：2026-10-19 提交 `854d12d` 标题为 `modified: vendor`，
   无说明、无门禁审查，子模块指针变更可以无信息量地进入历史。
3. **文档失真链**：指针更新后，任何引用子模块版本的文档陈述（README 索引描述、
   CHANGELOG 版本号、CONTRIBUTING 状态说明）自动失效；本轮里程碑 10-28
   文档诚实化收口删除的「失效依赖报告」即为指针漂移沉积的产物。

**目标**：把子模块指针状态纳入声明—实现一致性门禁，使指针变更显式化、
可审查、可对账，阻断「裸 modified: vendor」类提交与文档—子模块版本失真。

## 总体范围

- **新增 G 类断言**（子模块指针对账），并入现有 `check_declaration_consistency.py`；
- **扩展声明快照**：`declaration-snapshot.json` 新增 `submodule_status` 字段；
- **CI 集成**：复用现有 Quality Gates 的 declaration 步骤，不新增独立步骤；
- **非目标**：不自动升级子模块（`git submodule update --remote`）、不改变
  vendor 子模块的使用策略、不动 .gitmodules 注册方式。

## What Changes

### G1：子模块状态可解析与初始化状态（rule）

`git submodule status` SHALL 可解析出 `.gitmodules` 登记的全部子模块。
**未初始化子模块（`-` 前缀）记 note、不阻断**——仓库采用延迟初始化策略
（CONTRIBUTING.md §3.5：克隆后子模块目录为空属正常态）；
**已初始化但与索引不一致（`+` 前缀，指针漂移）由 G2 阻断**。

### G2：子模块指针与声明快照一致（rule）

`git submodule status` 的每个**已初始化**子模块 SHALL 处于与索引一致状态
（无 `+` 前缀——`+` 表示 checkout 的 SHA 与索引/快照不一致，即指针漂移，
直接阻断）；其 HEAD SHA SHALL 与 `declaration-snapshot.json` 中
`submodule_status` 字段记录的 SHA 一致（快照过期同样阻断）。
指针变更必须显式运行 `--generate` 刷新快照并在提交说明中注明上游范围
（如「更新 vendor/tvm-ffi 至 0932956（上游 main 快进 35 提交）」），
禁止无说明的裸指针变更提交。

### G3：文档版本引用与指针标签对账（rubric）

README / CONTRIBUTING / CHANGELOG 中引用的子模块版本描述
（如 `mystx v0.3.8`）与 `git submodule status` 输出的描述标签
（`v0.3.8-9-g18f3486`）SHALL 一致；结构化引用（明确版本号）用断言强制，
非结构化引用（功能描述）以门禁提示（warn）方式暴露。

## 影响面

- **代码**：`scripts/check_declaration_consistency.py`（新增 G1-G3 与
  `extract_submodule_status()`）、`scripts/declaration-snapshot.json`（新增字段）；
- **文档**：`scripts/README.md`（门禁说明新增 G 类）、`CONTRIBUTING.md`
  （子模块指针变更流程说明）；
- **CI**：`.github/workflows/ci.yml` Quality Gates 的
  `Check declaration consistency` 步骤自动继承新断言（无需改动）；
- **规范**：AGENTS.md 的「子模块指针变更」条款（如有）同步。

## ADDED Requirements

### Requirement: 子模块指针纳入声明快照（rule）

`declaration-snapshot.json` SHALL 新增 `submodule_status` 字段，记录
每个子模块的当前 HEAD SHA；`--generate` 刷新时同步更新。

#### Scenario: 刷新快照
- **WHEN** 维护者执行 `python scripts/check_declaration_consistency.py --generate`
- **THEN** `submodule_status` 字段与 `git submodule status` 当前输出一致

### Requirement: G1-G3 断言并入一致性门禁（rule）

`check_declaration_consistency.py` SHALL 在既有 A-F 六类断言外新增
G1（初始化状态）、G2（指针=快照）、G3（文档版本引用对账）三类断言；
G1/G2 失败以非零退出码阻断 CI。

#### Scenario: 上游快进后未同步指针
- **WHEN** 上游子模块快进而本地指针未更新，运行门禁
- **THEN** G2 报告 `submodule_status` 与快照不一致，退出码 1，CI 阻断

#### Scenario: 裸指针变更提交
- **WHEN** 维护者更新子模块指针但未运行 `--generate` 刷新快照、提交说明无上游范围
- **THEN** G2 拦截（快照过期），要求先刷新快照并补充说明

### Requirement: 指针变更流程显式化（rubric，0-2 分）

子模块指针变更从「人工发现 + 裸提交」转为「门禁拦截 + 显式刷新 + 注明范围」。

- `2`：G2 完整拦截，历史中不再出现 `modified: vendor` 类裸提交；
- `1`：G2 落地但非结构化版本引用仍偶发失真（G3 仅 warn）；
- `0`：门禁未覆盖指针，依赖人工。

**通过阈值**：≥1。

### Requirement: 文档与门禁同步（rule）

`scripts/README.md` SHALL 记录 G 类断言与 `--generate` 刷新流程；
`CONTRIBUTING.md` SHALL 说明「子模块指针变更 = 更新指针 + 刷新快照 +
注明上游范围」三步。

#### Scenario: 新维护者更新子模块
- **WHEN** 新维护者按 CONTRIBUTING 流程更新子模块指针
- **THEN** 三步流程完成，门禁通过，提交说明含上游范围

---

## 参考证据（2026-10 月）

| 事实 | 来源 |
|---|---|
| tvm-ffi 上游 main 快进 35 提交，10-13 手动同步指针 | git log `81c29a6` |
| 10-19 裸提交 `modified: vendor`，无说明无审查 | git log `854d12d` |
| 10-12 mystx 指针更新（文档 CI 修复与双搜索框治理） | git log `7415796` |
| 门禁六类断言（A-F）不含指针状态 | `scripts/check_declaration_consistency.py` |
| 快照仅含 cli_commands/lib_projects/submodules | `scripts/declaration-snapshot.json` |
| 当前子模块指针：mystx v0.3.8-9-g18f3486 / tvm-book v0.1.4-10-g103bea7 / caffe heads/main f938d88 / tvm-ffi v0.1.15-6-g6628c46 | `git submodule status` |
