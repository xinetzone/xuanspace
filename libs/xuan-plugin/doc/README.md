# Xuan Plugin Contract 设计文档与复盘（分卷）

- 冻结原件（逐字节）：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md)
- 原位置：`/media/pc/data/ai/xuanspace-pluginization-design.md` 已于 2026-11-23 删除，正文由本目录承载
- 原文指纹：SHA-256 `21c021fdda1ef2d5c1758495cbe643b75f83acf49f8887138e673e1c9a8a0419`｜60,008 字节｜759 行
- 原子化：2026-11-23，设计 11 篇 + 复盘 7 篇 + 本索引；两次原子化的正文多重集均守恒（missing 0 / extra 0）
- 权威顺序：本目录为唯一权威；归档件只用于回滚，逐页行区间见 [MANIFEST.md](MANIFEST.md)

## 阅读路径

| 角色 | 建议顺序 |
|---|---|
| 决策者 | 00 摘要 → 07 落地计划 → 09 度量 |
| 实现者 | 00 摘要 → 05 方案 → 06 对抗审查 |
| 复盘者 | 01 事实 → 08 P1-2 复盘 → 11–17 P2-4～P2-9 加固链复盘 |

## 分卷索引

| 篇 | 主题 | 原文区间 |
|---|---|---|
| 0 | [结论摘要（TL;DR）](00-tldr.md) | L13–L38 |
| 1 | [R — 事实基线](01-facts.md) | L39–L104 |
| 2 | [I — 洞察（四元组）](02-insights.md) | L105–L143 |
| 3 | [E — 模式萃取（可跨场景迁移）](03-patterns.md) | L144–L208 |
| 4 | [F — 第一性原理](04-first-principles.md) | L209–L234 |
| 5 | [方案：Xuan Plugin Contract（XPC）](05-solution.md) | L235–L396 |
| 6 | [V — 对抗审查（4 视角）](06-adversarial-review.md) | L397–L413 |
| 7 | [C — 落地计划（原子任务）](07-roadmap.md) | L414–L451 |
| 8 | [P1-2 里程碑复盘（2026-11-21）](08-retrospective-p1-2.md) | L452–L631 |
| 9 | [度量与止损](09-metrics.md) | L632–L648 |
| 10 | [附录](10-appendix.md) | L649–L759 |

## 第二部：P2-4～P2-9 加固链复盘（2026-11-21）

> **来源**：`.trae/specs/plugin-kernel-extraction/retrospective-20261121.md`（23,651 字节 / 320 行），已于 2026-11-23 原子化并入本目录并删除。
> **时点**：事实截至提交 `f42c37d`（P2-9）。其后 P2-10（`13c6127`）落地 ADR-05 多提供者定序、P2-11（`eada0de`）收口同名装配，故文中「未决项 2」与 A11 的「ADR-05 未落地」属当时事实；现行口径见 [07-roadmap.md](07-roadmap.md) 与 [tasks.md](../../../.trae/specs/plugin-kernel-extraction/tasks.md)。
> **质量门**：G1 事实 28 条｜G2 四元组洞察 4 条｜G3 模式 1 个 + 反模式 4 条｜G4 交付与未决项齐备；V 门 4 视角 6 条异议。

| 篇 | 主题 | 原文区间 |
|---|---|---|
| 11 | [R — 客观事实（G1，28 条）](11-retro-facts.md) | L11–L58 |
| 12 | [I — 洞察（G2，4 条四元组）](12-retro-insights.md) | L61–L94 |
| 13 | [E — 可复用模式与反模式（G3）](13-retro-patterns.md) | L97–L125 |
| 14 | [V — 对抗审查（6 条异议）](14-retro-adversarial.md) | L128–L140 |
| 15 | [C — 交付与未决项（G4）](15-retro-delivery.md) | L143–L170 |
| 16 | [更新记录与复核方式](16-retro-changelog.md) | L173–L192 |
| 17 | [附录 A — 设计基线原子化合并（A0–A12）](17-retro-appendix-a.md) | L195–L320 |

## 原文导言（设计文档）

> **文档性质**：设计说明（Design Note），非代码变更
> **方法论链路**：R（事实采集）→ I（洞察）→ E（模式萃取）→ F（第一性原理）→ V（对抗审查）→ C（原子化行动）
> **质量门**：G1 事实 ≥20 条（本文 30 条）｜G2 四元组洞察 ≥3 条（本文 5 条）｜G3 模式 3–7 步骤 + ≥3 反模式（本文 5 个模式）｜V 门 4 视角 ≥5 意见（本文 8 条）｜G4 单一职责可验收（本文 12 个原子任务）
> **依据材料**：
> - 论文：[A Programming Paradigm for Spatiotemporal Composability](https://arxiv.org/abs/2608.25512)（Cordis，arXiv:2608.25512）
> - 目标项目：`notebook/dev/libs/AI/projects/xuanspace`
> **结论**：可以。论文不是提供"一个插件框架"，而是提供**四个可独立迁移的机制**；xuanspace 恰好已有一份不完整的同类实现（`tools/okf`），因此方案不是"引入新框架"，而是**把已有的碎片收敛成契约，并将契约下沉为可被 libs/apps 依赖的库**。
