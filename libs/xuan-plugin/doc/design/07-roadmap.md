> 原子化来源：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md) · 原文 §7（L414–L451）
> 文档索引：[README.md](../README.md)

## 7. C — 落地计划（原子任务）

> 每个任务：单一职责、可独立验证、有验收标准；均不改变现有四分区结构。

| 阶段 | 任务 | 产出 | 验收 |
|---|---|---|---|
| P0-1 | 抽取 kernel：`tools/okf/src/okf/{context,plugin,disposable,service}.py` → `libs/xuan-plugin` | 新库（零运行时依赖）+ `okf` 改为依赖它 | `okf` 全量测试通过；`check_dependency_direction` 通过 |
| P0-2 | 身份改造：epoch 由 `id(impl)` 改为 `(key, impl_digest, config_digest)` | 摘要函数 + 单测（同输入同输出、跨进程一致） | 两次独立进程装配摘要相等 |
| P0-3 | 撤销加固：卸载异常聚合上报；FAILED 前强制撤销已登记 effect | `DisposableList.dispose_all(report)` | 注入故障插件的测试断言"effect 数 = 0 且异常被上报" |
| P1-1 | 清单解析器：`[tool.xuan.plugin]` schema + `entry_points("xuan.plugins")` 合并 | `xs plugin list/graph` | 对 14 个内置命令生成等价图 |
| P1-2 | 求解器：拓扑 + 隔离 + 对账 + 回滚 + 静默断言 | `xs plugin apply --dry-run` | 对刻意构造的冲突/环/缺依赖样例给出确定性输出 |
| P1-3 | `xs` 命令插件化：14 个静态 import 改为内置插件声明，保留兼容入口 | `cli.py` 仅保留装配调用 | `xs --help` 输出与改造前逐字相同（快照比对） |
| P2-1 | 试点 A：构建后端 multi 提供者（python/native/static） | `xuan:build/backend#dispatch` | `xs new/build` 在三种模板上行为不变 |
| P2-2 | 试点 B：`xuan-compose` 指令登记化 | 处理器表 → 登记 + Disposable | 现有 compose 测试全绿；新增指令只改一个文件 |
| P2-3 | 试点 C：FFI 能力键 + 子进程隔离模式 | `xuan:ffi/<name>#api` | 切换后端不改调用方代码；子进程模式可重复装载 |
| P3-1 | `xuan.plugins.lock` 与快照合并 | lock 生成器 + 派生快照 | H4 断言在 CI 生效；删除人工快照后 CI 仍绿 |
| P3-2 | H 类断言接入 CI | `scripts/check_plugin_contract.py` | 6 个反例样例全部被拦截并给出可读信息 |
| P3-3 | 开发体验：`xs plugin new/verify/why` | 模板 + 诊断 | 新人在无文档情况下可新增一个命令插件 |

**里程碑判据**：P1 完成后，"新增一个 `xs` 子命令"的改动应只涉及 1–2 个文件（pyproject + 插件模块），且 `xs doctor --plugins` 能显示其 ACTIVE 与依赖链。

> **执行进度**：
>
> - P1-2 已于 2026-11-21 落地（提交 `a8be022`、`da39463`），事实、根因与后续行动见 §8；P1-3 为下一阶段。
> - P3-2（H 类断言入 CI）已于 2026-11-23 落地（提交 `e4f393e`）。
> - P3-1（lock 与快照合并）已于 2026-11-23 落地生成器与 H4 断言：`xuan.plugins.lock`
>   由当前清单重算（`check_plugin_contract.py --write-lock`），H4 在 CI 生效；
>   既有 `scripts/declaration-snapshot.json` 作为派生快照保留，退役与否待定。
> - P2-4 起为本方案落地期按真实缺口顺延的编号（设计表官方 P2 编号只到 P2-3），均已落地，并逐条回填 `.trae/specs/plugin-kernel-extraction/tasks.md`：
>
> | 顺延编号 | 缺口 / 主题 | 提交 |
> |---|---|---|
> | P2-4 | 生命周期七态化：迁移表、显式重试、静态插件抑制热换 | `f6fd5dd` |
> | P2-5 | 热替换事务化，修复 apply 期效应归属 | `8d3b173` |
> | P2-6 | 批处理下游通知并消除 INACTIVE 静默缺口 | `beb5fbb` |
> | P2-7 | 身份摘要去地址化，无源码对象跨进程可复现 | `62d42a0` |
> | P2-8 | 状态机不变量收口：`FAILED ⇔ error`，`error` 仅描述当前失败态 | `be6f6b7` |
> | P2-9 | ACTIVE 与可解性互锁：`ACTIVE ⇒ 输入快照可解` | `f42c37d` |
> | P2-10 | 落地 ADR-05 多提供者定序：运行时裁决 + 显式覆盖 + 同值报错 | `13c6127` |

---

相关页面：[文档索引](../README.md) · 上一篇：[V — 对抗审查（4 视角）](06-adversarial-review.md) · 下一篇：[P1-2 里程碑复盘（2026-11-21）](../retro/08-retrospective-p1-2.md)
