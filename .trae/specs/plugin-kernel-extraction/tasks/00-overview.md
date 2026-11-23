> 分卷来源：`tasks.md` 原文 L1–L55（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：（无） · 下一卷：01-plan-p0-1.md
> 主题：总览：范围、不变量、前置事实
# 插件内核抽取（P0-1）— 实施计划

> Plan 阶段产物，2026-11-20。自工作区级设计文档 `xuanspace-pluginization-design.md` §7 P0-1 派生；
> 决策依据见该文档 §9.4 ADR-01。
>
> **范围声明**：本计划只做「kernel 换位置」，不做任何行为改造。`id(impl)` → 摘要身份（P0-2）、
> 卸载异常聚合（P0-3）、`ServiceRegistry` 的 priority 选择（ADR-05，属 P2）、`xs` 命令改造（P1-3）
> 全部不在本计划内（P0-2 与 P0-3 已于 2026-11-21 另行执行，记录见文末「增量一」「增量二」；
> P1-1 见「增量三」；P1-3 见「增量四」，已于 2026-11-21 完成）。
>
> **文档位置**：设计文档与复盘均已原子化至 [libs/xuan-plugin/doc/README.md](../../../../libs/xuan-plugin/doc/README.md)；
> 原 `xuanspace-pluginization-design.md` 与 `retrospective-20261121.md` 已删除，逐字节归档与逐页指纹见
> [doc/MANIFEST.md](../../../../libs/xuan-plugin/doc/MANIFEST.md)。
>
> **三条不变量**：
> 1. 零行为变更——`Context` / `Fiber` / `Disposable` / `ServiceRegistry` 的公开 API 与运行语义逐项保持原样，
>    包括 `id(impl)` 这一已知缺陷；
> 2. 零运行时依赖——新库只允许标准库，与 `tools/okf` 现状一致；
> 3. 不触碰业务模块——`frontmatter.py`、`links.py`、`trust.py`、`synthesis.py`、`conformance.py`、
>    `attested.py`、`loader.py`、`models.py`、`plugins/` 全部不动。

## 前置事实（计划制定时已核实）

**依赖闭包**（四者无外部依赖、无环，可整体平移）：

```text
disposable.py  ← 无内部依赖
plugin.py      ← disposable（运行时）+ context（仅 TYPE_CHECKING，防环）
context.py     ← disposable + plugin
service.py     ← context（仅 TYPE_CHECKING）
```

**待迁移文件**（`tools/okf/src/okf/`，合计约 15.7 KB）：

| 文件 | 字节 | 说明 |
|---|---:|---|
| `disposable.py` | 1504 | `Disposable` / `EffectMeta` / `DisposableList` |
| `plugin.py` | 4587 | `FiberState` / `_INACTIVE` / `InjectSpec` / `Plugin` / `Fiber` |
| `context.py` | 7103 | `Context`（effect/provide/get/plugin/notify/事件分发） |
| `service.py` | 2927 | `ServiceDefinition` / `ServiceProvider` / `ServiceRegistry` |

**消费点**（已用 grep 核实，共 21 行 import）：

- 源码 4 处：`context.py:8-9`（随文件迁移）、`plugin.py:9` 与 `plugin.py:12`（随文件迁移）、
  `service.py:11`（随文件迁移）、`harness.py:9-10`（需改）；
- 测试 9 文件 15 行：`test_context.py`、`test_context_extra.py`、`test_disposable.py`、
  `test_events.py`、`test_harness.py`、`test_plugin.py`、`test_plugin_extra.py`、
  `test_plugins_keyerror.py`、`test_service.py`（需改）。

**已知治理缺口**：`.github/workflows/ci.yml` 的 quality job 目前只执行 `xuan-cmd` 与
`xuan-compose` 的测试，`tools/okf/tests` 不在任何 CI 步骤中。本次等价性只能先靠本地基线，
持续保障须由 Task 7 补齐。

---

