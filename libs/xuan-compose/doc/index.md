# xuan-compose 文档

> 本目录是 xuan-compose 的文档中心。xuan-compose 是对
> [containers/podman-compose](https://github.com/containers/podman-compose) v1.6.0
> 的分层重构（translate-then-move，翻译式搬迁），行为对等、结构现代化、证据完备。
>
> 本页只承担**区级导航与阅读路径**。文档地图、编号体系、统计口径与证据总账的
> 权威陈述统一在 [dev/ 文档索引](dev/index.md)，本页不重复任何数字与结论
> （遵从本文档集自身的「同一事实只保留一个权威陈述位置」纪律）。

## 文档结构

| 区域 | 面向 | 内容 | 入口 |
|---|---|---|---|
| `guide/` | 使用者 | xc 是什么、怎么装、24 个命令与全局选项、库 API、compose 处理管线 | [guide/index.md](guide/index.md) |
| `examples/` | 使用者 | 可照做的完整用例：单服务栈、双服务栈、无 podman 纯解析演练 | [examples/index.md](examples/index.md) |
| `reference/` | 使用者 | 无叙述参数信源：全局选项全表、24 个命令逐选项表（实测 `xc --help`） | [reference/index.md](reference/index.md) |
| `dev/` | 维护者 | 重构过程的对等性证据、时点报告、常青台账与模式沉淀 | [dev/index.md](dev/index.md) |

> 使用者文档（`guide/`、`examples/`、`reference/`）统一以短命令 `xc` 讲解；
> `xc` 与规范名 `xuan-compose`、`python -m xuan_compose` 三入口等价，
> 说明见 [guide/00-introduction.md](guide/00-introduction.md)。

## 阅读路径

按读者意图任选一条，不必从头读：

**使用 xc**

| 我想…… | 从这里开始 |
|---|---|
| 安装并跑通第一个栈 | [guide/00 认识 xc](guide/00-introduction.md) → [examples/00 nginx 栈](examples/00-nginx-quickstart.md) |
| 查某个命令有哪些选项 | [reference/commands.md](reference/commands.md) |
| 查全局选项（`-f`/`-p`/`--dry-run` 等） | [reference/global-options.md](reference/global-options.md) |
| 写多服务 compose 文件 | [guide/04 处理管线](guide/04-compose-processing.md) → [examples/01 Web+Redis](examples/01-web-redis.md) |
| 无 podman 时解析/校验文件 | [guide/03 库 API](guide/03-library-api.md) → [examples/02 纯解析演练](examples/02-parse-without-podman.md) |

**理解重构与对等性（维护者）**

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

```{toctree}
:hidden:
:maxdepth: 2

guide/index
examples/index
reference/index
dev/index
```
