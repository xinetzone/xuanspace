> 原子化来源：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md) · 原文 §1（L39–L104）
> 文档索引：[README.md](../README.md)

## 1. R — 事实基线

### 1.1 论文事实（F1–F12）

| 编号 | 事实 | 出处 |
|---|---|---|
| F1 | 论文把"可组合性"拆成两个正交维度：**空间**（组件之间互相引用、循环依赖、全局可变状态、框架对对象的所有权）与**时间**（运行期增/删/替换组件）。 | §1 |
| F2 | 论文指出既有插件系统三类系统性缺口：能加载但不能卸载；插件之间没有依赖关系表达；插件产生的效果不可逆。 | §1.2.1 |
| F3 | 核心抽象是"**上下文**（context）作为一等值"：上下文是服务名到实现的映射，插件通过上下文取能力，而非直接 import 具体实现。 | §3.1 |
| F4 | "**可撤销效果**"（revertible effect）：每个效果函数返回逆操作；组件被撤下时按逆序执行逆操作，从而在时间维上恢复此前状态。 | §3.1 |
| F5 | "**反应式协效应**"（reactive coeffect）：组件显式声明自己读取哪些服务（依赖集）；当某个依赖的实现发生变化，依赖集命中的组件被退役并以新上下文重新插入，未命中的保持不动。 | §3.2 |
| F6 | 组件被建模为三元组：依赖声明、提供集合、效果函数；运行时维护注册表，**插入与退役是唯一的外部输入**，其余状态变化均由依赖传播推出。 | §4 |
| F7 | **隔离性**定义为：同一装配中任意两组件提供的服务集合互不相交；隔离性是并发/顺序无关的组合正确性前提。 | §4 |
| F8 | 引入**目标视图 / 提交视图**（target view / committed view）：目标视图是期望的插件配置，提交视图是实际生效的视图；提交视图由解析算法从目标视图推出。 | Definition 44, §5.1 |
| F9 | 解析算法（Algorithm 6）逐项解析注入：依赖可满足则激活；声明了依赖但当前不可得则置于 **INACTIVE**（不报错）；访问未声明服务则记录 **UNDECLARED_ACCESS**。 | §5.1 |
| F10 | 加载器把插件条目定义为七字段声明（id、来源、隔离、拦截、配置、禁用等），对配置做**逐字段对账**，只有字段变化的插件才重载。 | Definition 74, §5.2 |
| F11 | 论文证明三条装配性质：加载**顺序无关**（Theorem 63）、对账过程**终止**（Theorem 66）、对账后系统处于**静默**状态且静默性等价于"最终配置的函数"（Theorem 73）；热替换以备份—恢复实现事务性。 | §5.2 |
| F12 | 生产案例 Koishi：4000+ 社区插件、运行四年；插件即组件，禁用插件等价于撤销其全部效果；依赖变化只重载受影响的下游；依赖暂不可得的插件静默待激活而非崩溃。 | §5.3 |

补充讨论章（对工程决策直接有用）：

| 编号 | 事实 | 出处 |
|---|---|---|
| F13 | 服务允许多提供者（multiplex）与显式选择；单提供者假设在扩展场景下不成立。 | §6.2 |
| F14 | 访问控制通过作用域（scope）限定"某插件可见哪些服务"，能力即权限。 | §6.3 |
| F15 | 互相依赖不应靠运行期打补丁解决，而应拆分为更小组件，或显式声明 peer 依赖并由装载器保证同时在场。 | §6.5 |
| F16 | 语义化版本不足以表达"结构兼容"：需要按能力契约（方法集/签名）而非字符串版本比较。 | §6.6 |

### 1.2 xuanspace 现状事实（S1–S18）

| 编号 | 事实 | 证据位置 |
|---|---|---|
| S1 | 采用 monorepo + PDM workspace，四分区：`apps/`、`libs/`、`tools/`、`vendor/`，每个子项目是独立发行包（独立 `pyproject.toml`、独立版本）。 | 仓库根 `pyproject.toml`、`libs/README.md` |
| S2 | 治理以**声明一致性**为核心：CI 断言 pyproject 声明（名称/版本/入口/依赖）与实现（目录/模块/入口函数）一致，编号 A–F 类。 | `.trae/specs/submodule-alignment-gate/spec.md`、`.github/workflows/ci.yml` |
| S3 | 存在**依赖方向**断言：`libs/` 不得依赖 `apps/`、`scripts/`、`attic/`、`vendor/`。 | `scripts/check_dependency_direction.py`；CI |
| S4 | 存在**声明快照**（JSON），记录 `cli_commands`、`lib_projects`、`submodules` 三类事实，作为"声明 ↔ 实现"比对的基线。 | `scripts/`、`xs meta` 命令族 |
| S5 | `tools/xs` 是 14 个命令模块，用**静态 import** 在 `cli.py` 中逐一装配（`archive/build/deps/doc/doctor/init/lfs/list/meta/new/py-compat/toolchain/update/version`）。 | `tools/xs/src/xs/cli.py:13-26` |
| S6 | 项目发现是**文件系统扫描 + pyproject 探测**（按 build-backend 判定 python/native/static），无运行期注册表。 | `tools/xs/src/xs/discovery.py` |
| S7 | `libs/xuan-core` 仅含 `path.py`；`libs/xuan-cmd` 含 `errors/paths/projects/quoting`，定位为"命令装配与工程路径解析"；两者均无插件/生命周期概念。 | 各库 `pyproject.toml` 与源码清单 |
| S8 | `libs/xuan-compose` 是 podman-compose 1.6.0 的分层重构，命令分派采用**静态处理器表**（`COMMAND_HANDLERS` / `COMMAND_PARSERS`）。 | `libs/xuan-compose` |
| S9 | `libs/xuan-ext-demo` 是 C/C++ 原生扩展示例库，说明原生扩展路径已存在，但无统一加载/卸载契约。 | `libs/xuan-ext-demo` |
| S10 | `tools/okf` 是论文范式的**在库实现**：`Context` 提供 `effect/provide/get/plugin/notify/dispose/on/emit/bail/parallel/serial/waterfall`。 | `tools/okf/src/okf/context.py` |
| S11 | `okf.plugin.Plugin` 声明 `name/apply/inject/provide`；`InjectSpec` 携带服务名（与可选性）；`Fiber` 有六态：`PENDING/LOADING/ACTIVE/UNLOADING/DISPOSED/FAILED`。 | `tools/okf/src/okf/plugin.py:15-59` |
| S12 | `Fiber._compute_epoch()` 以 `f"{spec.name}:{id(impl)}"` 拼接依赖身份；epoch 不变即"中性"，变化则激活/停用/先卸后载。 | `tools/okf/src/okf/plugin.py:86-117` |
| S13 | `Fiber._unload()` 逆序执行 disposable，但用 `contextlib.suppress(Exception)` 吞掉卸载异常；`_reload()` 失败仅置 `FAILED`，无回滚。 | `tools/okf/src/okf/plugin.py:119-137` |
| S14 | `Harness.from_config()` 从 `[tool.okf.plugins]`（`name = "module:Class"`）读取静态映射，实例化后按 inject 做**拓扑排序**再逐个 `ctx.plugin()`；无环检测报告、无隔离校验、无配置对账。 | `tools/okf/src/okf/harness.py` |
| S15 | `okf` 位于 `tools/`，因此按 S3 的红线，`libs/` 与 `apps/` **不能**复用其 kernel。 | S3 + 目录归属 |
| S16 | 子模块（`vendor/mystx`、`vendor/tvm-book` 等）存在指针漂移，已被立项治理（A–F 断言之外的 G 类提案）。 | `.trae/specs/submodule-alignment-gate/` |
| S17 | 多环境（conda/pdm/uv × 3 OS）矩阵在 CI 中执行，说明装配必须**跨环境可复现**。 | `.github/workflows/ci.yml` |
| S18 | 文档站（Sphinx/MyST）与 `xs doc` 已存在，`doc/okf/` 有 architecture/plugins/loader 等章节，说明"装配视图可视化"有现成落点。 | `doc/okf/`、`tools/README.md` |

### 1.3 差距矩阵（论文机制 × xuanspace 现状）

| 论文机制 | xuanspace 现状 | 差距判定 |
|---|---|---|
| 上下文 / 服务缝（F3） | `okf.Context` 已有，但仅 `tools/` 内可见；库之间仍是编译期 import | **有实现、无分发** |
| 可撤销效果（F4） | `Disposable`/`DisposableList` 已有；卸载吞异常、无残留检测 | **有骨架、无保证** |
| 反应式协效应（F5） | `Fiber._refresh` + epoch 已有；epoch 用进程内 `id()` | **有机制、不可复现** |
| 隔离性（F7） | 无 provide 冲突检查；多提供者无显式语义 | **缺失** |
| 目标/提交视图与对账（F8–F10） | 有声明快照，但与磁盘是"双真相"；无逐字段对账 | **形态相似、语义不同** |
| 顺序无关/终止/静默（F11） | 有拓扑排序；无终止与静默断言 | **缺证明性门禁** |
| 多提供者与作用域（F13–F14） | 无 | **缺失** |
| peer 依赖与结构兼容（F15–F16） | 仅字符串依赖方向断言 | **缺失** |

---

相关页面：[文档索引](../README.md) · 上一篇：[结论摘要（TL;DR）](00-tldr.md) · 下一篇：[I — 洞察（四元组）](02-insights.md)
