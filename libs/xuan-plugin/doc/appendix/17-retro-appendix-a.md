> 原子化来源：`retrospective-20261121.md` 原文 §七（L195–L320）；该独立文件已于 2026-11-23 并入本目录并删除，唯一权威为本目录
> 时点声明：本页为 2026-11-21 时点记录（截至提交 `f42c37d`）。此后清单已变化——设计文档于 2026-11-23 原子化为本目录（原件归档后删除），ADR-05 由 P2-10（`13c6127`）落地，同名装配由 P2-11（`eada0de`）收口；文中「547 行 / md5」「ADR-05 未落地」均属当时事实，现行口径以 [README.md](../README.md) 与 [tasks.md](../../../../.trae/specs/plugin-kernel-extraction/tasks.md) 为准
> 文档索引：[README.md](../README.md)

## 七、附录 A：设计基线原子化合并

> **合并锚点**：`\\10.16.11.3\ai\xuanspace-pluginization-design.md`，**547 行**，md5 `26778c085a45e9bd254ce466341be17d`（2026-11-21 读取）。
> **原子化规则**：按源文章节切分为 A1–A12 十二个原子块，每块只保留「判据 + 实现对照 + 证据」；原理论证不复述，源文件仍是唯一规范源。
> **防第二真相**：本附录不产出新判据；「实现对照」是唯一新增内容，每条给出可复核证据。源文件 md5 变化时本附录失效，须重新生成。

### A0 合并方式与复核注记

- 合并粒度：源 §0→A1、§1→A2、§2→A3、§3→A4、§4→A5、§5.2→A6、§5.4→A7、§5.5→A8、§7→A9、§8→A10、§9.4→A11、§0.3→A12。
- 复核注记 1：源文自述「G1 事实 ≥20 条（本文 30 条）」，实际编号为 F1–F16（16 条）+ S1–S18（18 条）共 **34 条**；条目数应以编号为准。
- 复核注记 2：源 §5.5 状态图含 **11 条边**，代码迁移表含 **15 条边**（差异见 A8）。

### A1 结论与七项决策（源 §0）

| 决策 | 判据 | 实现对照（2026-11-21） |
|---|---|---|
| D1 kernel 下沉 `libs/xuan-plugin` | `tools → libs` 为允许方向；okf 反向依赖 | **已落地**：库存在，六次提交在其上加固（F1–F6） |
| D2 能力键 `xuan:<domain>/<owner>#<api>` | 键稳定、命名空间隔离 | **部分**：`ServiceDefinition.name` 为自由字符串，命名空间语法未在 kernel 强制 |
| D3 登记返回 `Disposable` | 禁止裸副作用 | **已落地**：`Fiber.effect` 返回 undo，`_unload` 逆序撤销、异常聚合 |
| D4 `xuan.plugins.lock` 单一权威视图 | 快照由系统派生 | **未落地**：现为 `scripts/declaration-snapshot.json`，`check_declaration_consistency.py` 2 项失败（F18） |
| D5 H 类断言入 CI | 可解/隔离/撤销/视图一致 | **未落地**：现行门禁为 A–G 类 + ruff/mypy（F12–F17） |
| D6 默认关闭自动发现 | 集合发现须显式 | 本轮未核验 |
| D7 原生 `lifecycle="static"` | `.so` 不可卸载 | **已落地**：`test_static_plugin_suppresses_hot_reload`、`test_static_plugin_still_retires_handles_on_dispose` |

### A2 事实基线与差距矩阵（源 §1）

源列 34 条事实，与 kernel 直接相关的 6 条已被本轮改动超越：

| 源编号 | 源描述 | 当前事实 |
|---|---|---|
| S11 | `Fiber` 六态 | 七态（追加 `INACTIVE=6`，F21） |
| S12 | epoch 用 `id(impl)` | 改为「类型身份 + 源码/归一化 repr 摘要」（`62d42a0`） |
| S13 | 卸载 `suppress(Exception)` 吞错 | `dispose_all` 聚合上报（P2-8 增补 2 条用例） |
| S14 | Harness 拓扑装配无环检测、无对账 | 运行期由 `_TRANSITIONS` 15 边机检约束（F22） |
| S15 | okf 在 `tools/` 致 libs 不可复用 | 已解（D1 落地） |
| S5 / S8 | 静态 import / 静态处理器表 | 试点已登记化（`46f0e47`、`ea2fc0e`），其余未核验 |

其余差距项（隔离性、多提供者、作用域、peer 依赖、结构兼容）保持源判定：**缺失或未核验**。

### A3 五条洞察（源 §2）

| 编号 | 一句话判据 | 本轮验证状态 |
|---|---|---|
| I1 | 声明一致 ≠ 组合一致 | 已验证：P2-4～P2-6 建立运行期组合判定 |
| I2 | 缺撤销语义使可选能力退化为编译期开关 | 已验证：K3 落地（D3） |
| I3 | 静态装配使新增能力成本落在核心文件 | 部分：试点 B 已登记化 |
| I4 | 快照 vs 磁盘是双真相 | 未解决：D4 未落地 |
| I5 | okf 的位置与身份限制其实用性 | 已验证：P0-1 + P0-2 落地 |

### A4 五个可迁移模式（源 §3）

P1 能力键（5 步）｜P2 可撤销装配（5 步）｜P3 版本化上下文（5 步）｜P4 声明视图对账（5 步）｜P5 多提供者与作用域（4 步）。
反模式合计 4 类：模块路径/包名当身份、`atexit` 兜底、`id()` 身份、`providers[0]` 隐式取首。与主报告 E 阶段互补：源模式给**设计骨架**，主报告模式给**缺陷收敛流程**。

### A5 三条公理（源 §4）

| 公理 | 判据 | 对应本轮提交 |
|---|---|---|
| A1 身份公理 | 被依赖物须有稳定、可复现身份 | P2-7（`62d42a0`） |
| A2 可逆公理 | 状态变更须伴随可执行逆操作 | P2-4 / P2-5 / P2-8 |
| A3 局部公理 | 影响范围 = 声明依赖的传递闭包 | P2-6 批处理 |

推论三句：清单是可执行契约；装配器是求解器；快照是解的输出。

### A6 五张契约（源 §5.2）

| 契约 | 判据 | 违约检测 | 落地 |
|---|---|---|---|
| K1 能力键 | 语法 + `single/multi` + 域归属 | 键名正则 + 域表 | 部分（见 A1-D2） |
| K2 插件清单 | 9 字段，`lifecycle ∈ {reloadable, static}` | schema + 清单/实现双向一致 | `lifecycle` 已落地；schema 未核验 |
| K3 可撤销 | 登记返回句柄、逆序、异常聚合不吞 | AST 扫描 + 残留断言 | 已落地（无 AST 扫描） |
| K4 版本化上下文 | `(key, impl_digest, config_digest)`；同 lock 两次装配摘要相等 | 双进程摘要比对 | 已落地（P2-7 测试） |
| K5 隔离与作用域 | provide 两两不交、越界记 UNDECLARED_ACCESS | 装配期冲突 + 访问审计 | 未核验 |

### A7 装配算法十步（源 §5.4）

1 collect → 2 normalize → 3 graph（缺依赖标 INACTIVE）→ 4 cycles → 5 order（确定性）→ 6 apply（隔离校验）→ 7 reconcile（逐字段 diff）→ 8 rollback（逆序撤销）→ 9 quiesce（静默断言）→ 10 commit（写 lock）。

落地对照：步骤 3（`_deactivate` + `reason`）、5（`Harness` 拓扑）、7（`_describe_change`）、8（`_unload` + `rollbacks`）、9（P2-8 迁移表机检 + P2-9 可解性互锁）已具备；步骤 10 的 lock 未落地（D4）。

### A8 生命周期状态机：设计 11 边 vs 代码 15 边

源 §5.5 图边 11 条：PENDING→INACTIVE、PENDING→LOADING、INACTIVE→LOADING、LOADING→ACTIVE、LOADING→FAILED、FAILED→LOADING、ACTIVE→UNLOADING、UNLOADING→PENDING、ACTIVE→DISPOSED、PENDING→DISPOSED、FAILED→DISPOSED。

代码 `_TRANSITIONS` 额外 4 条及理由：INACTIVE→DISPOSED（dispose 未激活节点）、FAILED→INACTIVE（依赖消失）、FAILED→UNLOADING（dispose 回收失败节点）、UNLOADING→FAILED（撤销异常）。

五态不变量：PENDING / INACTIVE / FAILED / DISPOSED 效应数为 0；ACTIVE 依赖全可解（P2-9 互锁后成立）。

### A9 落地计划 12 个原子任务（源 §7）

| 任务 | 状态 |
|---|---|
| P0-1 kernel 抽取 | 已完成（库存在、`okf` 依赖它） |
| P0-2 身份改造 | 已完成（`62d42a0`） |
| P0-3 撤销加固 | 已完成（`test_disposal_errors.py`，P2-8 增补） |
| P1-1 清单解析器 | 本轮未核验 |
| P1-2 求解器 | 部分（`xs plugin apply` 可用：3 节点 ACTIVE、20 键一致） |
| P1-3 CLI 插件化 | 本轮未核验 |
| P2-1 构建后端 multi | 台账 P2-1-1 已登记完成 |
| P2-2 compose 指令登记化 | 已完成（`46f0e47`） |
| P2-3 FFI 能力键 | 已完成（`ea2fc0e`） |
| P3-1 lock 与快照合并 | 未开始（D4 未落地） |
| P3-2 H 类断言入 CI | 未开始（D5 未落地） |
| P3-3 `xs plugin new/verify/why` | 本轮未核验 |

### A10 度量与止损（源 §8）

| 指标 | 目标 | 当前值 |
|---|---|---|
| M1 扩展成本 | ≤ 2 文件 | 未测量 |
| M2 装配可复现性 | 100% 相等 | 已由 P2-7 测试覆盖（同输入同摘要） |
| M3 卸载残留 | 0 | 0（`test_non_active_states_hold_no_effects`） |
| M4 变更半径 | ≤ 依赖闭包 | 结构性满足（按 inject 传播），未量化 |
| M5 定位时间 | < 1 分钟 | 未测量 |

止损线三条维持原文：P0-1 行为变化即暂停；试点改动调用方 >20 行即退回 P1；H4 误报 >5% 先修锁生成器。

### A11 决策记录 ADR-01～05（源 §9.4）

ADR-01 新建 `libs/xuan-plugin`（Accepted，已落地）；ADR-02 默认关闭动态发现（Accepted，本轮未核验）；ADR-03 原生静态生命周期（Accepted，已落地，回退条件为「无」）；ADR-04 锁单向派生、不做双写（Accepted，未落地）；ADR-05 多提供者「priority 定序 + 显式覆盖 + 同值报错」（Accepted，**未落地**：`ServiceRegistry.lookup` 仍取 `providers[0]`，与 ADR-05 直接冲突，见未决项 2）。

### A12 边界（源 §0.3）

1. 不承诺 C 扩展热卸载，只承诺「句柄可撤销 + 无残留引用」，必要时子进程隔离兜底。
2. 不引入 JS 生态依赖、不重写 Cordis；在 okf 之上做减法与加固。
3. 不改变 `apps/libs/tools/vendor` 四分区与 PDM workspace 结构。

---

相关页面：[文档索引](../README.md) · 上一篇：[附录](16-appendix.md)
