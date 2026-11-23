> 原子化来源：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md) · 原文 §9（L632–L648）
> 文档索引：[README.md](README.md)

## 9. 度量与止损

| 指标 | 定义 | 基线（现状） | 目标 |
|---|---|---|---|
| M1 扩展成本 | 新增一个能力需改动的文件数 | 2–4（`cli.py` + 命令模块 + 可能的注册表） | ≤ 2 |
| M2 装配可复现性 | 同 lock 两次装配的视图摘要是否相等 | 不可比（`id()`） | 100% 相等 |
| M3 卸载残留 | 禁用插件后残留 effect 数 | 未测量（异常被吞） | 0 |
| M4 变更半径 | 一次提供方变更触发的重载节点占比 | 100%（重启） | ≤ 依赖闭包占比 |
| M5 定位时间 | 装配失败到定位根因的耗时 | 人工排查日志 | < 1 分钟（`xs plugin why`） |

**止损线**：
- 若 P0-1 抽取导致 `okf` 行为变化（非等价重构），暂停后续阶段，先补等价性测试。
- 若 P2 任一试点需要改动调用方业务代码超过 20 行，判定该试点"能力边界不成立"，退回 P1 重新划分键。
- 若 H4 在真实仓库上误报率 > 5%，先修锁生成器，不放开动态发现。

---

相关页面：[文档索引](README.md) · 上一篇：[P1-2 里程碑复盘（2026-11-21）](08-retrospective-p1-2.md) · 下一篇：[附录](10-appendix.md)
