> 原子化来源：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md) · 原文 §10（L649–L759）
> 文档索引：[README.md](../README.md)

## 10. 附录

### 10.1 术语对照

| 论文术语 | XPC 术语 | 落地位置 |
|---|---|---|
| context | 上下文 / 能力容器 | `libs/xuan-plugin` Context |
| service / effect | 能力键 / 能力句柄 | K1、K3 |
| revertible effect | 可撤销登记 | `Disposable` |
| reactive coeffect | 依赖声明 + epoch 重载 | K4、Fiber |
| component | 插件（清单 + apply） | K2 |
| isolation | provide 不相交 | K5、H2 |
| target / committed view | 期望视图 / 生效视图 | `xuan.plugins.lock` |
| entry（Definition 74） | 插件条目 | `[tool.xuan.plugin]` |
| INACTIVE / UNDECLARED_ACCESS | 待激活 / 越界访问 | H1、H5 |
| multiplex | 多提供者 + priority 定序 / 显式覆盖 | P5、ADR-05 |

### 10.2 论文机制 → 方案条目速查

| 论文 | 方案 |
|---|---|
| §1 二维可组合性 | §4 公理 A1–A3 |
| §1.2.1 三类缺口 | I1–I3 |
| §3.1 上下文 / 可撤销效果 | K1 / K3、P2 |
| §3.2 反应式协效应 | K4、P3 |
| §4 组件与隔离 | K2、K5、H2 |
| §5.1 解析算法 | §5.4 步骤 1–4 |
| §5.2 加载器与对账 | §5.4 步骤 7–10、H4 |
| §5.3 Koishi 经验 | M1–M5 指标设计 |
| §6.2 多提供者 | P5、P2-1 |
| §6.3 作用域/权限 | K5、V6 |
| §6.5 互相依赖 | D2、V2 |
| §6.6 结构兼容 | K2 `constraint`、H5 |

### 10.3 参考文件

- 论文：[A Programming Paradigm for Spatiotemporal Composability](https://arxiv.org/abs/2608.25512)（arXiv:2608.25512；[PDF v1](https://arxiv.org/pdf/2608.25512v1)）
- 既有 kernel：`notebook/dev/libs/AI/projects/xuanspace/tools/okf/src/okf/`（`context.py`、`plugin.py`、`harness.py`、`service.py`、`disposable.py`）
- CLI 装配现状：`notebook/dev/libs/AI/projects/xuanspace/tools/xs/src/xs/cli.py:13-26`
- 项目发现：`notebook/dev/libs/AI/projects/xuanspace/tools/xs/src/xs/discovery.py`
- 治理基线：`notebook/dev/libs/AI/projects/xuanspace/.trae/specs/submodule-alignment-gate/spec.md`、`.github/workflows/ci.yml`
- 既有可见性体系：`notebook/dev/libs/AI/projects/xuanspace/doc/okf/`

### 10.4 决策记录（ADR）

> 本节为方案定稿的决策存档，替代原「遗留问题」清单。每条记录状态、结论、依据、代价与回退条件；回退条件写明是为了让后续复盘能核对"当初的假设是否仍成立"，而非预留反悔空间。

#### ADR-01　kernel 归属：新建 `libs/xuan-plugin`

- **状态**：Accepted（对应 D1）
- **结论**：新建独立库，只下沉内核四件套（`Context` / `Fiber` / `Disposable` / `ServiceRegistry`）；`tools/okf` 保留原位，并反向依赖该库。**不并入 `xuan-core`。**
- **依据**：
  1. `xuan-core` 实际只有 4 个文件、源码合计 840 B（`path.py` 689 B + `__init__.py` 151 B），零依赖，且全仓无任何包依赖它——是尚未成形的占位库。
  2. 内核是低频变更、高稳定性要求的组件；`xuan-core` 是高频扩展的通用工具。二者变更频率相差一个数量级，必须分包，否则每次内核改动都会牵动全部 `xuan-core` 使用者。
  3. 位置必须落在 `libs/`：`tools → libs` 是允许方向；若内核留在 `tools/`，可复用库将反向依赖内部工具链。
  4. 内核的首个消费者是真实存在的 okf，而非凭空设计——下沉路径可由 okf 既有的 20 个测试文件直接验证。
- **代价**：多一个发行包。缓解：`check_declaration_consistency.py` 的 `LIB_PROJECT_META` 增加一行登记，与既有 9 个子项目同模式。
- **回退条件**：若三个月内内核公开 API 变动不超过一次，且无第二个消费者出现，可评估并回 `xuan-core`。

#### ADR-02　动态发现：默认关闭

- **状态**：Accepted（对应 D6）
- **结论**：默认不启用自动发现；第三方扩展须通过 `xs plugin add` 显式登记，并写入 `xuan.plugins.lock`。
- **依据**：
  1. 当前 `xs` 装配为静态 import（`tools/xs/src/xs/cli.py:13-26`），同一 commit 在任何机器上命令集一致。默认开启发现会使"同一 commit 在不同机器跑出不同命令集"，直接摧毁 `declaration-snapshot.json` 的对账基础。
  2. 论文 Theorem 73（静默性）保证的是**给定插件集合后结果与插入顺序无关**，不保证"集合本身的发现过程确定"。发现环节必须显式，这是对论文边界的准确理解，而非保守取舍。
- **代价**：第三方扩展接入多一步显式登记。
- **回退条件**：若开发期本地调试插件成为反复出现的真实痛点，增加命令行级开关（如 `--dev-plugin-dir`），仍不改为默认开启。

#### ADR-03　原生后端：静态生命周期，隔离显式启用

- **状态**：Accepted（对应 D7，细化 §0.3 边界第 1 条）
- **结论**：原生插件声明 `lifecycle = "static"`，进程内只装配一次、不卸；子进程隔离仅在两类场景显式启用——需反复装卸做 A/B 对比，或插件崩溃需可恢复。
- **依据**：
  1. CPython 对 extension module 不做真正的 `dlclose`。从 `sys.modules` 删除后，`.so` 的静态区、全局变量、已注册的 C 回调仍然驻留。这是解释器边界，不是实现缺陷。
  2. 论文 §3.1 的"可撤销效果"针对**状态**——撤销 `provide()` 注册、撤销事件监听、撤销文件句柄；从未承诺撤销已映射进地址空间的机器码。原生层的"卸载"只能是逻辑退役（停止分发调用）。
  3. 重载原生插件等价于重启进程。§5.3 Koishi 的热重载结论不可外推——其插件全为 JS。
- **代价**：原生插件调试需重启进程。原生扩展本就低频变更，可接受。
- **回退条件**：无。此为物理约束，不因需求变化而改变。

#### ADR-04　锁与快照：单向派生，不做双写过渡

- **状态**：Accepted（对应 D4 的时序细化）
- **结论**：不设双写过渡期。新增 `xuan.plugins.lock` 为唯一装配真相；`declaration-snapshot.json` 降级为从 lock 派生的只读产物。
- **依据**：
  1. 双写的本质是保留两个真相，正是洞察 I4 诊断的病根。以双写过渡等于以病根治病，并会制造"两份都不对"的中间态。
  2. 回滚需求有更简且更强的解法：lock 是纯文本、可 diff、可 review，回滚即 `git revert` 该文件，审计链完整。
  3. 双写时间窗还需额外的到期删除与一致性校验逻辑，复杂度高于收益。
- **代价**：旧快照格式的外部消费者需迁移。
- **回退条件**：若存在仓库外消费者且无法改造，用 `xs plugin export --format=snapshot` 生成旧格式到 `.temp/`，仍不入版本控制。

#### ADR-05　多提供者的选择策略

- **状态**：Accepted，**已落地**（P2-10，提交 `13c6127`）
- **背景**：复核发现 `tools/okf/src/okf/service.py:64-86`（P0-1 后迁至 `libs/xuan-plugin/src/xuan_plugin/service.py`）的 `ServiceRegistry.lookup()` 实现为"取 `providers[0]`"——多提供者只有注册能力、没有选择策略，论文 §6.2 的 multiplex 在 okf 中实际不可用。P5 落地前必须先定规则，否则 multiplex 只是纸面能力。
- **结论**：「priority 定序 + 显式覆盖 + 同值报错」三段规则：
  1. `multi` 键按清单 `priority` 降序取最高者；
  2. 消费者可在 lock 中写 `provider = "<id>"` 强制指定，覆盖 priority 结果；
  3. `priority` 相同且未显式指定时，装配期报错——不猜、不取首个。
- **落地（P2-10）**：
  1. 运行时：`ServiceProvider` 新增 `priority` / `id`；`lookup(name, ctx, config=None, *, provider=None)` 委托 `_select` 裁决——显式 id 优先（未知 id 抛 `LookupError`，同 id 重复抛 `AmbiguousProviderError`），否则取最高 `priority`，并列即 `AmbiguousProviderError`；无提供者仍返回 `None`；
  2. 求解侧：`solver._find_conflicts` 对同键 `multi` 的最高 `priority` 并列报 `SolveError(CONFLICT)`；`_provider_index` 按 `(-priority, id)` 定序；
  3. 诊断图：`manifest.build_graph` 保留 `(-priority, id)` 过渡定序（对应本条「回退条件」），不作为装配裁决依据；
  4. 待办边界：清单 schema 的 `provider = "<id>"` 字段尚未引入（属 P5 / lock 范围），规则 2 目前由运行时关键字参数承载，未写入清单；
  5. 证据：`libs/xuan-plugin/tests/test_service.py` 13 条用例覆盖规则 1–3 与边界；`tools/okf/tests/test_service.py` 的「插入到最前」用例改为「抬优先级透明切换 + 显式回退」；`okf` 的 `architecture.md` Capability Seam 小节同步为三段规则。
- **依据**：
  1. 论文 Thm 63/73 要求最终配置与插入顺序无关。`providers[0]` 使结果依赖注册顺序，同一插件集合换一种加载顺序即得到不同实现，与顺序无关性直接冲突，因此**不能作为降级方案保留**。
  2. `priority` 是 K2 契约的既有字段（清单示例中的 `priority = 100`）。选择依据属集合固有属性，复用而不新增语法。
  3. 同值报错把歧义前移到装配期，与论文"依赖不可得报 INACTIVE、越界报 UNDECLARED_ACCESS"的显式失败风格一致，优于运行期静默选中。
- **代价**：`multi` 提供者须标注 `priority`；若 schema 不强制，同值报错会被频繁触发。缓解：schema 校验要求 `multi` 键的每个 provider 显式写 `priority`。
- **回退条件**：若存量插件难以补齐 `priority`，对未标注者按 `id` 字典序定序作为过渡（字典序同样与插入顺序无关），并在装配告警中提示补标。**不再回退到 `providers[0]`。**

---

相关页面：[文档索引](../README.md) · 上一篇：[更新记录与复核方式](../retro/15-retro-changelog.md) · 下一篇：[附录 A — 设计基线原子化合并（A0–A12）](17-retro-appendix-a.md)
