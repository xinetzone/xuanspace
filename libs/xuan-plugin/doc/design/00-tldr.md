> 原子化来源：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md) · 原文 §0（L13–L38）
> 文档索引：[README.md](../README.md)

## 0. 结论摘要（TL;DR）

### 0.1 一句话命题

xuanspace 当前的"管理"是**声明期治理**（目录约定 + pyproject 声明 + CI 断言 + 快照），缺的是**运行期的可组合性**：能力没有稳定身份、装配没有可撤销效果、变更没有传播规则。论文的四机制正好补齐这三处：**能力键（空间）→ 可撤销效果（时间）→ 版本化上下文（变更传播）→ 声明视图对账（可复现装配）**。

### 0.2 七项关键决策

| # | 决策 | 依据 |
|---|---|---|
| D1 | 新增 `libs/xuan-plugin`：只下沉 `tools/okf` 的 kernel 四件套（Context/Fiber/Disposable/ServiceRegistry），okf 保留原位并反向依赖它 | `check_dependency_direction.py` 现仅拦截 `apps/scripts/attic/vendor`，`tools/` 未列入——但 `libs` 反向依赖内部工具链违背分区语义，规则日后收紧即断裂；`xuan-core` 现为零依赖占位库（仅 `path.py`），与内核的变更频率不同级（ADR-01） |
| D2 | 依赖单元从"发行包"改为"能力键（capability key）"，采用命名空间 `xuan:<domain>/<owner>#<api>` | 论文 §6.5：互相依赖须拆分为组件或显式 peer 依赖；键冲突须命名空间隔离 |
| D3 | 一切登记必须返回 `Disposable`，禁止裸副作用；卸载残留纳入门禁 | 论文 §3.1 可撤销效果；现状 `plugin.py` 卸载处 `suppress(Exception)` 静默吞错 |
| D4 | 装配视图 = `committed view`，由 `xuan.plugins.lock` 单一权威产出，禁止快照与磁盘双真相 | 论文 §5.1 算法 6、§5.2 逐字段对账与静默性定理 |
| D5 | CI 新增 H 类断言（可解性 / 隔离性 / 可撤销性 / 视图一致性），与既有 A–G 类声明一致性断言同级 | 论文 §4 隔离性与 §5.2 顺序无关性、终止性、静默性 |
| D6 | 关闭自动发现：第三方扩展须显式 `xs plugin add` 登记并写入 lock | 论文 Thm 73 只保证"给定集合后与插入顺序无关"，不保证集合发现过程确定；默认发现会摧毁跨机器可复现性（ADR-02） |
| D7 | 原生插件声明 `lifecycle = "static"`，进程内不卸；子进程隔离仅在反复装卸或崩溃可恢复场景显式启用 | CPython 不对 extension module 做 `dlclose`，`.so` 静态状态无法撤销；§3.1 的可撤销效果针对状态而非机器码（ADR-03） |

### 0.3 不做什么（边界）

- **不承诺 C 扩展的热卸载**：`tvm-ffi`、`caffe-ffi` 的 `.so` 无法真正回滚，只承诺"能力句柄可撤销、进程内无残留引用"，必要时以子进程隔离兜底（见 V1；决策依据见 ADR-03）。
- **不引入 JS 生态依赖**，不重写 Cordis；`okf` 已是零运行时依赖的 Python 移植，方案在其上做减法与加固。
- **不改变现有 `apps/libs/tools/vendor` 四分区与 PDM workspace 结构**，插件化是叠在其上的"运行视图"。

---

相关页面：[文档索引](../README.md) · 下一篇：[R — 事实基线](01-facts.md)
