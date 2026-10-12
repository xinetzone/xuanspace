# xuan-compose 文档

> 本目录是 xuan-compose 的文档中心。xuan-compose 是对
> [containers/podman-compose](https://github.com/containers/podman-compose) v1.6.0
> 的分层重构（translate-then-move，翻译式搬迁），行为对等、结构现代化、证据完备。
>
> 本页只承担**区级导航与阅读路径**。文档地图、编号体系、统计口径与证据总账的
> 权威陈述统一在 [dev/ 文档索引](dev/index.md)，本页不重复任何数字与结论
> （遵从本文档集自身的「同一事实只保留一个权威陈述位置」纪律）。

## 文档结构

本目录只有一个子区：

| 区域 | 内容 | 入口 |
|---|---|---|
| `dev/` | 重构过程的对等性证据、时点报告、常青台账与模式沉淀 | [dev/index.md](dev/index.md) |

## 阅读路径

按读者意图任选一条，不必从头读：

| 我想…… | 从这里开始 |
|---|---|
| 先看结论、统计口径与全部证据总账 | [dev/index.md](dev/index.md)（文档地图 + 证据与复现总账） |
| 查某个上游符号迁到了哪个模块 | [dev/parity/symbol-map.md](dev/parity/symbol-map.md) |
| 查所有已登记问题的处置结论 | [dev/findings-ledger.md](dev/findings-ledger.md) |
| 复用单体重构的可迁移模式 | [dev/patterns/2026-09-16-refactor-patterns.md](dev/patterns/2026-09-16-refactor-patterns.md) |
| 复核交付、审查与独立复验过程 | [dev/delivery/](dev/delivery/2026-09-16-delivery-summary.md)、[dev/review/](dev/review/2026-09-16-t11-adversarial-review.md)、[dev/verification/](dev/verification/2026-09-16-full-parity-verification.md) |

## 上游基线

基线 commit、许可证与术语口径的**唯一权威位置**在
[dev/index.md「上游基线」](dev/index.md#上游基线单一事实源)，本页不再另行声明。

## 相关入口

- 库使用手册（API / CLI / 架构 / 测试 / 集成验证）：[../README.md](../README.md)
- 版本历史：[../CHANGELOG.md](../CHANGELOG.md)