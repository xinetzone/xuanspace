# Xuan Plugin Contract 设计文档与复盘（按主题分组）

- 冻结原件（逐字节）：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md)
- 原位置：`/media/pc/data/ai/xuanspace-pluginization-design.md` 已于 2026-11-23 删除，正文由本目录承载
- 原文指纹：SHA-256 `21c021fdda1ef2d5c1758495cbe643b75f83acf49f8887138e673e1c9a8a0419`｜60,008 字节｜759 行
- 原子化：2026-11-23，设计 11 篇 + 复盘 7 篇 + 本索引；两次原子化的正文多重集均守恒（missing 0 / extra 0）
- 主题分组：2026-11-23，按主题分为 `design/`、`retro/`、`appendix/` 三组并重编号；正文零改动，逐页行区间与指纹见 [MANIFEST.md](MANIFEST.md)
- 权威顺序：本目录为唯一权威；归档件只用于回滚

## 目录结构

```text
doc/
├── README.md      本索引
├── MANIFEST.md    原子化清单与指纹
├── design/        组 B 设计论证（00–07）
├── retro/         组 C 复盘与度量（08–15）
└── appendix/      组 D 附录与基线（16–17）
```

## 主题分组总览

| 组 | 目录 | 篇数 | 定位 |
|---|---|---:|---|
| B 设计论证 | `design/` | 8 | 从事实基线到落地计划的完整论证链（R→I→E→F→V→C） |
| C 复盘与度量 | `retro/` | 8 | P1-2 里程碑复盘、度量止损与 P2-4～P2-9 加固链复盘 |
| D 附录与基线 | `appendix/` | 2 | 设计附录（术语／ADR 索引）与设计基线 A0–A12 |

## 阅读路径

| 角色 | 建议顺序 |
|---|---|
| 决策者 | [00 摘要](design/00-tldr.md) → [07 落地计划](design/07-roadmap.md) → [09 度量](retro/09-metrics.md) |
| 实现者 | [00 摘要](design/00-tldr.md) → [05 方案](design/05-solution.md) → [06 对抗审查](design/06-adversarial-review.md) |
| 复盘者 | [01 事实](design/01-facts.md) → [08 P1-2 复盘](retro/08-retrospective-p1-2.md) → [10 事实](retro/10-retro-facts.md) → [14 交付](retro/14-retro-delivery.md) |

## 篇目索引（按分组）

### B 设计论证（`design/`）

| 篇 | 主题 | 原文节 | 原文区间 |
|---|---|---|---|
| 00 | [结论摘要（TL;DR）](design/00-tldr.md) | §0 | L13–L38 |
| 01 | [R — 事实基线](design/01-facts.md) | §1 | L39–L104 |
| 02 | [I — 洞察（四元组）](design/02-insights.md) | §2 | L105–L143 |
| 03 | [E — 模式萃取（可跨场景迁移）](design/03-patterns.md) | §3 | L144–L208 |
| 04 | [F — 第一性原理](design/04-first-principles.md) | §4 | L209–L234 |
| 05 | [方案：Xuan Plugin Contract（XPC）](design/05-solution.md) | §5 | L235–L396 |
| 06 | [V — 对抗审查（4 视角）](design/06-adversarial-review.md) | §6 | L397–L413 |
| 07 | [C — 落地计划（原子任务）](design/07-roadmap.md) | §7 | L414–L451 |

### C 复盘与度量（`retro/`）

| 篇 | 主题 | 来源节 | 原文区间 |
|---|---|---|---|
| 08 | [P1-2 里程碑复盘（2026-11-21）](retro/08-retrospective-p1-2.md) | 设计 §8 | L452–L631 |
| 09 | [度量与止损](retro/09-metrics.md) | 设计 §9 | L632–L648 |
| 10 | [R — 客观事实（G1，28 条）](retro/10-retro-facts.md) | 复盘 §一 | L11–L58 |
| 11 | [I — 洞察（G2，4 条四元组）](retro/11-retro-insights.md) | 复盘 §二 | L61–L94 |
| 12 | [E — 可复用模式与反模式（G3）](retro/12-retro-patterns.md) | 复盘 §三 | L97–L125 |
| 13 | [V — 对抗审查（6 条异议）](retro/13-retro-adversarial.md) | 复盘 §四 | L128–L140 |
| 14 | [C — 交付与未决项（G4）](retro/14-retro-delivery.md) | 复盘 §五 | L143–L170 |
| 15 | [更新记录与复核方式](retro/15-retro-changelog.md) | 复盘 §六 | L173–L192 |

### D 附录与基线（`appendix/`）

| 篇 | 主题 | 来源节 | 原文区间 |
|---|---|---|---|
| 16 | [附录](appendix/16-appendix.md) | 设计 §10 | L649–L759 |
| 17 | [附录 A — 设计基线原子化合并（A0–A12）](appendix/17-retro-appendix-a.md) | 复盘 §七 | L195–L320 |

## 编号规则

- 文件名编号 `00`–`17` 是分组后的**全局阅读序**；正文小节号（如 `## 10. 附录`、`## 一、R 阶段`）沿用**原文 § 号**，两者独立，重编号不改正文。
- 新增篇目追加到所属组末位，其后各组顺延编号，并同步更新本索引与 [MANIFEST.md](MANIFEST.md)。

## 第二部：P2-4～P2-9 加固链复盘（2026-11-21）

> **来源**：`.trae/specs/plugin-kernel-extraction/retrospective-20261121.md`（23,651 字节 / 320 行），已于 2026-11-23 原子化并入本目录并删除。
> **时点**：事实截至提交 `f42c37d`（P2-9）。其后 P2-10（`13c6127`）落地 ADR-05 多提供者定序、P2-11（`eada0de`）收口同名装配，故文中「未决项 2」与 A11 的「ADR-05 未落地」属当时事实；现行口径见 [07-roadmap.md](design/07-roadmap.md) 与 [tasks.md](../../../.trae/specs/plugin-kernel-extraction/tasks.md)。
> **质量门**：G1 事实 28 条｜G2 四元组洞察 4 条｜G3 模式 1 个 + 反模式 4 条｜G4 交付与未决项齐备；V 门 4 视角 6 条异议。
> **对应篇目**：10–15（复盘 R/I/E/V/C/更新记录）；其中 17（附录 A）按主题归入 `appendix/`。

## 原文导言（设计文档）

> **文档性质**：设计说明（Design Note），非代码变更
> **方法论链路**：R（事实采集）→ I（洞察）→ E（模式萃取）→ F（第一性原理）→ V（对抗审查）→ C（原子化行动）
> **质量门**：G1 事实 ≥20 条（本文 30 条）｜G2 四元组洞察 ≥3 条（本文 5 条）｜G3 模式 3–7 步骤 + ≥3 反模式（本文 5 个模式）｜V 门 4 视角 ≥5 意见（本文 8 条）｜G4 单一职责可验收（本文 12 个原子任务）
> **依据材料**：
> - 论文：[A Programming Paradigm for Spatiotemporal Composability](https://arxiv.org/abs/2608.25512)（Cordis，arXiv:2608.25512）
> - 目标项目：`notebook/dev/libs/AI/projects/xuanspace`
> **结论**：可以。论文不是提供"一个插件框架"，而是提供**四个可独立迁移的机制**；xuanspace 恰好已有一份不完整的同类实现（`tools/okf`），因此方案不是"引入新框架"，而是**把已有的碎片收敛成契约，并将契约下沉为可被 libs/apps 依赖的库**。
