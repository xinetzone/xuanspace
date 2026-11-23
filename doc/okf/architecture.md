# 架构原理

`okf` 工具链的代码可分为两层：**机制层**（如何组织可插拔能力）与**领域层**（何谓合法 Bundle）。机制层由 Harness / 插件 / 上下文 / 可逆效应等组成，领域层由加载器 / 校验 / 合成 / 信任 / 链接等组成。

## 模块地图

```text
src/okf/
├── cli.py            # CLI（argparse，6 条命令）
├── harness.py        # Harness 自举：pyproject.toml 驱动的插件装配
├── context.py        # 统一上下文：效应追踪 + 服务注册 + 插件装配 + 事件分发
├── plugin.py         # 插件与 Fiber 生命周期状态机
├── service.py        # Capability Seam：ServiceDefinition / Provider / Registry
├── disposable.py     # 可逆效应：Disposable / DisposableList / EffectMeta
├── events.py         # 兼容模块（事件分发已内聚到 Context）
├── models.py         # 核心数据模型（frozen dataclass）
├── loader.py         # Bundle 加载（领域层）
├── frontmatter.py    # 自研 YAML 子集解析（领域层）
├── links.py          # 链接解析与断链检测（领域层）
├── conformance.py    # 一致性校验（领域层）
├── synthesis.py      # index/log 合成（领域层）
├── trust.py          # 信任与保鲜（领域层）
├── attested.py       # Attested Computation（领域层）
└── plugins/          # 默认插件实现
```

## 设计主线一：一切皆插件、无特权内核

Harness（`src/okf/harness.py`）借鉴 DeepSeek Harness，核心理念是：

- **无特权内核**：`Harness` 本身不含业务逻辑，只负责装配。
- **一切皆插件**：加载、校验、合成、信任、链接、CLI 适配全部是插件。
- **配置驱动**：装配顺序由 `pyproject.toml` 的 `[tool.okf.plugins]` 配置节驱动（或用默认清单）。
- **可替换**：任意插件可被第三方实现替换（见[插件开发](plugins.md)）。

## 设计主线二：插件生命周期状态机（Fiber）

每个插件被包装为一个 `Fiber`（内核位于 `libs/xuan-plugin/src/xuan_plugin/plugin.py`，`okf` 通过 `xuan-plugin` 复用），它管理单个插件的完整生命周期，具有七个状态：

```text
PENDING ──→ INACTIVE ──→ LOADING ──→ ACTIVE ──→ UNLOADING ──→ PENDING
              ↑                           │
              └──── 依赖消失 ─────────────┘
LOADING ──apply 异常──→ FAILED ──retry()──→ LOADING
任意状态（撤销成功时） ──dispose──→ DISPOSED（终态，二次 dispose 幂等）
```

- 插件通过 `Plugin.provide` 声明「提供哪些服务」，通过 `Plugin.inject` 声明「依赖哪些服务」。
- `Fiber._compute_epoch` 计算依赖快照（依赖服务实现的摘要拼接）；摘要由「模块 + 限定名 + 源码哈希」构成，源码不可得时以「类型身份 + 归一化 `repr`」兜底，repr 中的堆地址一律折叠为固定占位符，故同一装配在两次独立进程中摘要相等（度量 M2）。快照变化时触发 `_refresh`，实现**响应式热替换**：先撤销旧效应、再应用新实现。成功计入 `Fiber.reloads`，`Fiber.reload_reason` 给出变化的依赖字段（`impl` / `config`）；新实现失败时撤销其已登记的部分效应（零残留），以 `FAILED` + 原始异常暴露，可经 `retry()` 恢复。
- `INACTIVE` 表示必需依赖暂不可得：不抛错、不持有任何效应，并通过 `Fiber.reason` 给出缺失键；依赖出现后自动迁移到 `LOADING`。
- 依赖方只在**最终视图**上判定：`Fiber._refresh`、`retry`、`Context.plugin` 各自处于一次传播事务内，多键提供方的卸载/装载通知合并排空，依赖方不会看到「键暂时缺失」的中间态；一次逻辑变更只产生一轮替换（度量 M4）。
- `INACTIVE` 且依赖已全部可解时立即重新装配（含「撤销后重供同一实现」），不会停在可激活而未激活的静默缺口。
- `ACTIVE` 与可解性互锁：装配期间必需依赖被移除时，`_reload` 不在 `ACTIVE` 上停留——先撤销刚登记的效应，再收敛到 `INACTIVE` 并由 `Fiber.reason` 给出缺失键；故 `ACTIVE ⇒ 输入快照可解` 恒成立。热替换中发生该情形既不计 `reloads`（无新身份生效）也不计 `rollbacks`（无待重试目标）。
- 加载成功进入 `ACTIVE`，失败进入 `FAILED`（已登记效应强制撤销）；`retry()` 提供显式重试通道。
- `error` 与状态互锁：`FAILED ⇔ error is not None`。卸载成功后清零，`error` 只描述当前失败态；`Context.dispose` 因此只汇总真正的处置失败，不会把调用方已读取的装配失败二次上报。
- 迁移表之外的调用（如对 `ACTIVE` 调用 `retry()`、对 `DISPOSED` 再次加载）抛 `FiberTransitionError`，不做静默降级；7×7 状态对由 `test_lifecycle.py` 穷举守护，表与设计 §5.5 漂移即失败。
- `lifecycle="static"` 的插件在 `ACTIVE` 后抑制热换（ADR-03）；`dispose` 仍执行逻辑退役，撤销能力句柄与效应，但不承诺卸载已映射的 `.so`。

## 设计主线三：可逆效应（Disposable）

`src/okf/disposable.py` 定义「可逆效应」抽象：任一副作用（服务注册、事件监听、资源分配）都返回一个 `undo` 逆函数。`DisposableList` 以栈式语义（后注册先回收）收集这些逆函数，实现资源的确定性清理。

`Context.dispose` 会逆序卸载所有 Fiber 并清空服务与效应，全程幂等安全。

## 设计主线四：统一上下文（Context）

`Context`（`src/okf/context.py`）同时承载四类职责：

1. **服务注册**：`provide` / `get`。`provide` 自带撤销登记——它返回的逆函数同时登记为效应（`apply` 作用域内归属当前 Fiber，否则归属根 `DisposableList`），因此调用方无需再包 `effect(lambda: ctx.provide(...))`；同名重复供应只保留最后一次登记，注销责任随「最后写入」的持有者。`provide` 后自动 `notify` 依赖方；`Context.batch()` 事务内的通知只登记，最外层退出时去重、分轮排空（`dispose` 使用 `flush=False` 丢弃）。事务外保持同步语义。
2. **效应追踪**：`effect` 注册可逆效应；处于某插件 `apply` 作用域时归属该 Fiber（随卸载/热替换逆序回收），其余归属根 `DisposableList`。`provide` 与 `effect` 共用同一套归属规则，故热替换或 apply 失败时供应键随效应一并撤销，不会残留孤儿。
3. **插件装配**：`plugin` 创建 Fiber 并启动加载；名称在旧实例 `DISPOSED` 之前被独占，同名重复装配抛 `PluginNameConflictError` 而非静默覆写——覆写会让旧实例在 `dispose` 遍历中缺席，其供应与效应成为孤儿。
4. **事件分发**：五种分发模式。

`Context` 实现了上下文管理器协议，`with` 退出时自动回收资源。

## 事件分发的五种模式

事件方法已内聚到 `Context`（`src/okf/events.py` 仅为向后兼容保留）：

| 方法 | 语义 |
|------|------|
| `on` | 注册监听器，返回逆函数取消监听 |
| `emit` | 纯通知：同步执行所有监听器，无返回值 |
| `bail` | 短路查找：第一个返回非 `None` 的结果获胜 |
| `parallel` | 并行执行（含异步），返回结果列表 |
| `serial` | 串行执行，第一个返回非 `None` 的结果获胜 |
| `waterfall` | 中间件链：监听器接收 `(*args, next)`，调用 `next()` 继续 |

## Capability Seam 三角色

`src/okf/service.py` 定义能力的接口与实现分离：

| 角色 | 职责 |
|------|------|
| `ServiceDefinition` | 声明「能做什么」（name + interface） |
| `ServiceProvider` | `ServiceDefinition` 的具体实现（factory + config + `priority` + `id`） |
| `ServiceRegistry` | 注册与查找 `definition` / `provider` |

多个 Provider 并存时，`lookup` 按 ADR-05 三段规则裁决，不做隐式「取首个」：

1. 默认取 `priority` 最大者，结果与注册顺序无关；
2. 调用方可传 `provider="<id>"` 显式指定，覆盖 `priority` 结果；`id` 不存在时抛 `LookupError`；
3. 最高 `priority` 并列且未显式指定时抛 `AmbiguousProviderError`——不猜、不取注册顺序首个。

无任何 Provider 时 `lookup` 返回 `None`（「依赖不可得不是异常」的既有语义）。

## 数据模型

`src/okf/models.py` 用 `@dataclass(frozen=True, slots=True)` 定义全部核心类型（`Concept` / `Bundle` / `Source` / `AttestedComputation` / `ConformanceReport` 等），保证不可变与类型安全。`slots=True` 降低内存占用。

## 参见

- [插件开发](plugins.md)
- [一致性校验](conformance.md)
- [索引、日志与链接](synthesis.md)