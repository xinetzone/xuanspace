# 插件内核抽取（P0-1）— 实施台账

> 本文件为索引：正文已按增量原子化为 8 个分卷，共 25 个 `## Task` 块。
> 逐字节保真（截至分卷时刻 P2-11）：各卷去掉页首 3 行横幅后顺序拼接，与冻结原件在「4 处相对链接改指」归一化后 SHA-256 一致
> （`39b4c9e99f05d688b296b36821b3100c5e343d6c446f98a142d0652ec3ececf3`，101461 字节 / 1359 行）；
> 其后新增任务只追加到对应分卷末尾（P2-12-1 追加于 07 卷），不再与冻结原件逐字节相等。
> 链接改指：00 与 07 卷各 2 处 `../../../libs/...` → `../../../../libs/...`（目录下移一层，等价改写）。
> 冻结原件：`/media/pc/data/ai/.agents/archive/plugin-kernel-tasks-20261123/raw/tasks.md`。
> 文档位置：设计文档与复盘见 [libs/xuan-plugin/doc/README.md](../../../libs/xuan-plugin/doc/README.md)。

## 分卷索引

| 卷 | 主题 | 原文区间 | 字节 | 任务数 | SHA-256 前 12 位 |
|---|---|---|---|---|---|
| 00-overview.md | 总览：范围、不变量、前置事实 | L1–L55 | 3311 | 0 | `893aea0f7eac…` |
| 01-plan-p0-1.md | 计划本体 P0-1（Task 1–8 + 风险 + 完成判据） | L56–L361 | 21387 | 8 | `eee0d19d0557…` |
| 02-p0-2-identity.md | 增量一 P0-2：epoch 摘要身份 | L362–L411 | 3841 | 1 | `972985f53817…` |
| 03-p0-3-disposal.md | 增量二 P0-3：卸载异常聚合 | L412–L465 | 4484 | 1 | `a4797e1d0f16…` |
| 04-p1-kernel-cli.md | 增量三 P1：清单、求解器、CLI 插件化 | L466–L659 | 16778 | 3 | `3aee9530626c…` |
| 05-p2-a-build-ffi.md | P2-A：构建后端、compose、FFI 试点 | L660–L840 | 14109 | 3 | `64b3eeba347f…` |
| 06-p2-b-lifecycle.md | P2-B：生命周期、热替换、传播、摘要 | L841–L1111 | 20574 | 4 | `1c6762842a1c…` |
| 07-p2-c-invariants.md | P2-C：不变量收口、互锁、ADR-05、同名装配、供应即撤销 | L1112–L1359 | 24483 | 5 | `de2ee6c565f9…` |

## 任务总表（派生视图）

> 状态列由各卷的 `- **Status**:` 行派生，权威值在分卷内；本表随分卷变化须同步重生成。

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

---

> 复盘报告（P2-4～P2-11 全链，R → I → E → C，G1–G4 通过）：
> 已原子化并入 [libs/xuan-plugin/doc/README.md（第二部）](../../../libs/xuan-plugin/doc/README.md)，原独立文件已删除。
