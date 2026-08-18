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
- **可替换**：任意插件可被第三方实现替换（见[插件开发](plugins)）。

## 设计主线二：插件生命周期状态机（Fiber）

每个插件被包装为一个 `Fiber`（`src/okf/plugin.py`），它管理单个插件的完整生命周期，具有六个状态：

```text
PENDING → LOADING → ACTIVE → UNLOADING → DISPOSED
                     ↘ FAILED
```

- 插件通过 `Plugin.provide` 声明「提供哪些服务」，通过 `Plugin.inject` 声明「依赖哪些服务」。
- `Fiber._compute_epoch` 计算依赖快照（依赖服务实现的 `id` 拼接）；快照变化时触发 `_refresh`，实现**响应式重载**。
- 加载成功进入 `ACTIVE`，失败进入 `FAILED`；卸载时逆序回收本插件注册的资源。

## 设计主线三：可逆效应（Disposable）

`src/okf/disposable.py` 定义「可逆效应」抽象：任一副作用（服务注册、事件监听、资源分配）都返回一个 `undo` 逆函数。`DisposableList` 以栈式语义（后注册先回收）收集这些逆函数，实现资源的确定性清理。

`Context.dispose` 会逆序卸载所有 Fiber 并清空服务与效应，全程幂等安全。

## 设计主线四：统一上下文（Context）

`Context`（`src/okf/context.py`）同时承载四类职责：

1. **服务注册**：`provide` / `get`，其中 `provide` 后自动 `notify` 依赖方。
2. **效应追踪**：`effect` 注册可逆效应到根 `DisposableList`。
3. **插件装配**：`plugin` 创建 Fiber 并启动加载。
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
| `ServiceProvider` | `ServiceDefinition` 的具体实现（factory + config） |
| `ServiceRegistry` | 注册与查找 `definition` / `provider` |

`lookup` 使用首个 Provider 的 factory 实例化服务。

## 数据模型

`src/okf/models.py` 用 `@dataclass(frozen=True, slots=True)` 定义全部核心类型（`Concept` / `Bundle` / `Source` / `AttestedComputation` / `ConformanceReport` 等），保证不可变与类型安全。`slots=True` 降低内存占用。

## 参见

- [插件开发](plugins)
- [一致性校验](conformance)
- [索引、日志与链接](synthesis)