# tasks.md 实施台账

> 本文件为索引：正文已按语义增量原子化为 11 个分卷，共 34 个 `## Task` 块。
> 逐字节保真：各卷去掉页首横幅后顺序拼接，与冻结原件在「链接改指归一化」后 SHA-256 一致
> （`28e0c1e77ab6707b2ec24b9a71a2a542afe17c1981a183a03eca52f380265dba`，143421 字节 / 1891 行）。
> 改指记录：无。
> 冻结原件：[../../../../../../../../../.agents/archive/plugin-kernel-tasks-20261123-r4/raw/tasks.md](../../../../../../../../../.agents/archive/plugin-kernel-tasks-20261123-r4/raw/tasks.md)（`volumize.py check` 可复验）。
> 正文权威值在各分卷；本表由 `volumize.py check` 机检，**请勿手改**。

## 分卷索引

| 卷 | 主题 | 原文区间 | 字节 | 任务数 | 正文 SHA-256 前 12 位 | 文件 SHA-256 前 12 位 |
|---|---|---|---|---|---|---|
| 00-overview.md | 总览：范围、不变量、前置事实 | L1–L55 | 3311 | 0 | `1bc846f54cb1…` | `893aea0f7eac…` |
| 01-plan-p0-1.md | 计划本体 P0-1（Task 1–8 + 风险 + 完成判据） | L56–L361 | 21387 | 8 | `152b8976cd70…` | `eee0d19d0557…` |
| 02-p0-2-identity.md | 增量一 P0-2：epoch 摘要身份 | L362–L411 | 3841 | 1 | `2cd76d357653…` | `972985f53817…` |
| 03-p0-3-disposal.md | 增量二 P0-3：卸载异常聚合 | L412–L465 | 4484 | 1 | `a8a57232f5c3…` | `a4797e1d0f16…` |
| 04-p1-kernel-cli.md | 增量三 P1：清单、求解器、CLI 插件化 | L466–L659 | 16778 | 3 | `c645935de92a…` | `3aee9530626c…` |
| 05-p2-a-build-ffi.md | P2-A：构建后端、compose、FFI 试点 | L660–L840 | 14109 | 3 | `1c363a81ae2b…` | `64b3eeba347f…` |
| 06-p2-b-lifecycle.md | P2-B：生命周期、热替换、传播、摘要 | L841–L1111 | 20574 | 4 | `14f5f9aa5691…` | `1c6762842a1c…` |
| 07-p2-c-invariants.md | P2-C：不变量收口、互锁、ADR-05、同名装配、供应即撤销、诊断图同口径 | L1112–L1774 | 53169 | 11 | `1149660e7715…` | `3c99abcec859…` |
| 08-p2-d-doc-grouping.md | P2-D：设计文档按主题分组 | L1775–L1816 | 3149 | 1 | `4ec7d11d10d0…` | `756509070e1f…` |
| 09-p3-a-contract-gate.md | P3-A：H 类插件契约断言接入 CI | L1817–L1852 | 2797 | 1 | `9bd5f3e602e0…` | `a7e09c7a41d2…` |
| 10-p3-b-lock.md | P3-B：xuan.plugins.lock 生成器与 H4 生效 | L1853–L1891 | 3057 | 1 | `c702db065f9b…` | `3f2494a1dce5…` |

## 任务总表（派生视图）

> 状态列由各卷的 `- **Status**:` 行派生；改状态请改分卷，再跑 `volumize.py check` 复验。

| 编号 | 标题 | 状态 | 所在分卷 |
|---|---|---|---|
| `1` | 固化迁移前基线 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `2` | 新建 `libs/xuan-plugin` 骨架 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `3` | 迁移 kernel 四件套 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `4` | 改造 okf 源码消费点 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `5` | 改造 okf 测试导入 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `6` | 治理登记 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `7` | 验证与 CI 补口 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `8` | 提交与回滚预案 | `completed` | [01-plan-p0-1.md](tasks/01-plan-p0-1.md) |
| `P2-1` | epoch 改用摘要身份并纳入注入配置 | `completed` | [02-p0-2-identity.md](tasks/02-p0-2-identity.md) |
| `P3-1` | 卸载异常聚合与装配失败回滚 | `completed` | [03-p0-3-disposal.md](tasks/03-p0-3-disposal.md) |
| `P1-1-1` | K2 清单解析器与 `xs plugin list/graph` | `completed` | [04-p1-kernel-cli.md](tasks/04-p1-kernel-cli.md) |
| `P1-2-1` | 装配求解器、事务回滚与 `xs plugin apply --dry-run` | `completed` | [04-p1-kernel-cli.md](tasks/04-p1-kernel-cli.md) |
| `P1-3-1` | 命令注册切换为插件装配并接通真实 `apply` | `completed` | [04-p1-kernel-cli.md](tasks/04-p1-kernel-cli.md) |
| `P2-1-1` | 构建后端多提供者（`xuan:build/backend#dispatch`） | `completed` | [05-p2-a-build-ffi.md](tasks/05-p2-a-build-ffi.md) |
| `P2-2-1` | `xuan-compose` 指令登记化（处理器表 → 登记 + Disposable） | `completed` | [05-p2-a-build-ffi.md](tasks/05-p2-a-build-ffi.md) |
| `P2-3-1` | FFI 能力键与子进程隔离模式（试点 C） | `completed` | [05-p2-a-build-ffi.md](tasks/05-p2-a-build-ffi.md) |
| `P2-4-1` | 插件生命周期与状态机（§5.5 落地） | `completed` | [06-p2-b-lifecycle.md](tasks/06-p2-b-lifecycle.md) |
| `P2-5-1` | 插件热更新事务与效应归属（§5.4 第 6–8 步） | `completed` | [06-p2-b-lifecycle.md](tasks/06-p2-b-lifecycle.md) |
| `P2-6-1` | 下游传播批处理（事务内汇总 notify，事务结束统一刷新） | `completed` | [06-p2-b-lifecycle.md](tasks/06-p2-b-lifecycle.md) |
| `P2-7-1` | 身份摘要去地址化（无源码对象跨进程可复现） | `completed` | [06-p2-b-lifecycle.md](tasks/06-p2-b-lifecycle.md) |
| `P2-8-1` | 状态机不变量收口（error 语义与迁移表机检） | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-9-1` | ACTIVE 与可解性互锁 | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-10-1` | 落地 ADR-05 多提供者定序 | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-11-1` | 同名插件装配的确定性 | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-12-1` | 供应即撤销（`provide` 自动登记效应） | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-13-1` | 诊断图与求解器冲突同口径（`build_graph` 不再静默择一） | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-14-1` | 通知排空失败保留待处理集合并可重试 | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-15-1` | 排空轮次上界改为按 Fiber 数推导，消除对合法深链的误伤 | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-16-1` | 装配失败回滚半注册 Fiber，允许同名重装 | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-17-1` | `xs plugin graph` 冲突与必需未解析以非零退出码收口 | `completed` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-18-1` | `xs plugin why <id>` 输出失活因果链（设计 V4 / 度量 M5） | `未标注` | [07-p2-c-invariants.md](tasks/07-p2-c-invariants.md) |
| `P2-19-1` | 文档按主题分组（三目录 + 重编号 + 改链） | `completed` | [08-p2-d-doc-grouping.md](tasks/08-p2-d-doc-grouping.md) |
| `P3-2-1` | H 类插件契约断言接入 CI（设计 §5.7） | `completed` | [09-p3-a-contract-gate.md](tasks/09-p3-a-contract-gate.md) |
| `P3-1-2` | `xuan.plugins.lock` 生成器与 H4 视图一致性生效（设计 §7 P3-1） | `completed` | [10-p3-b-lock.md](tasks/10-p3-b-lock.md) |