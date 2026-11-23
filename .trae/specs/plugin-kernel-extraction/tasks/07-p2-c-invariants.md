> 分卷来源：`tasks.md` 原文 L1112–L1359（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：06-p2-b-lifecycle.md · 下一卷：（无）
> 主题：P2-C：不变量收口、互锁、ADR-05、同名装配
## Task P2-8-1: 状态机不变量收口（error 语义与迁移表机检）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-7-1（`62d42a0`）；处置 P1-2 / P1-3 遗留的 A4 待办
- **Description**:
  - 缺口一：`Fiber._unload` 成功后不清空 `_error`，卸载成功（→PENDING）的 Fiber 仍携带
    上一次装配失败的异常；`Context.dispose` 因此把调用方早已读取的失败当成「处置错误」
    二次上报，装配器只能在 `assembly.py` 中按 `id(exc)` 去重兜底（第二真相，且依赖地址）。
  - 缺口二：`_TRANSITIONS` 是手写常量，缺少与设计 §5.5 绑定的机检；7×7 状态对中任何
    「多一条或漏一条」都不会被现有路径测试发现。
  - 改动：
    - `plugin.py::_unload` 成功分支清空 `_error`；`Fiber.error` 口径改为「仅当前失败态，
      `FAILED ⇔ error is not None`」；
    - `assembly.py` 删除按异常对象去重，直接并入 `ctx.get_disposal_errors()`；
    - `test_lifecycle.py` 新增迁移表逐项对账、7×7 穷举与四态零效应不变量；
      `test_disposal_errors.py` 新增「卸载成功清零」「处置不重复上报装配失败」；
    - `doc/okf/architecture.md`：状态图 dispose 边改为「任意状态（撤销成功时）」，
      补充 `error` 互锁与穷举守护说明。
- **Acceptance Criteria Addressed**: §5.5 状态不变量可机检；`Context.dispose` 只汇总真正的
  处置失败；迁移表与设计漂移即失败
- **Test Requirements**:
  - `rule` TR-P2.8-1: 迁移表与 §5.5（含三条显式补边）逐项一致；证据：
    `test_transition_table_matches_documented_edges`
  - `rule` TR-P2.8-2: 7×7 穷举：表内迁移成功、表外抛错且状态不变；证据：
    `test_transition_rejects_every_edge_outside_the_table`
  - `rule` TR-P2.8-3: PENDING/INACTIVE/FAILED/DISPOSED 效应数恒为 0；证据：
    `test_non_active_states_hold_no_effects`
  - `rule` TR-P2.8-4: 卸载成功后 `error` 为 None；证据：
    `test_error_is_cleared_after_successful_unload`
  - `rule` TR-P2.8-5: `Context.dispose` 不重复上报已读取的装配失败；证据：
    `test_context_dispose_does_not_repeat_apply_failure`
  - `rule` TR-P2.8-6: 去掉去重后装配回滚的失败清单仍无重复；证据：
    `test_failure_rolls_back_in_reverse_and_leaves_no_residue`
- **Completion Evidence**（2026-11-21 实测）:
  - 第一批 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests`
    → **194 passed**（P2-7 后基线 189，新增 5）；
  - `tools/okf` 独立运行 → **259 passed**；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 58 文件已格式化；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src tools/xs/src`
    → `Success: no issues found in 40 source files`；
    `check_pyproject_style.py`（16 个 pyproject）、`check_dependency_direction.py`
    （337 文件）、`check_spec_integrity.py`（34 个任务编号）通过；
    `check_declaration_consistency.py` 仍为既有 2 项失败（`libs/mystx` 指针漂移与快照过期）；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - **V1 error 语义变化**：`error` 不再是历史账本。装配失败在 `ctx.plugin()` 返回时即可读，
    调用方须就地取证；需要历史排障应依赖日志而非该属性。
  - **V2 去重删除**：卸载失败会产生新的 `DisposalError`，与 apply 原始异常是两条不同事实，
    同时上报是正确的；真正重复的来源（同一异常对象）已由清零消除。
  - **V3 穷举测试直改 `state`**：靶子是 `_transition` 守卫本身；路径级状态覆盖由既有
    `test_lifecycle.py` / `test_notification_batch.py` 承担，两者互补而非替代。
  - **V4 三条补边**：`FAILED→INACTIVE`（依赖消失）、`FAILED→UNLOADING`（dispose 回收
    失败节点）、`UNLOADING→FAILED`（撤销异常）在代码注释中已有说明，本轮纳入机检基线。
  - **V5 不变量边界**：`FAILED ⇔ error is not None` 在 `_deactivate` 从 FAILED 直接转
    INACTIVE 时同样成立（该分支显式清零）。

## Task P2-9-1: ACTIVE 与可解性互锁

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-8-1（`be6f6b7`）
- **Description**:
  - 缺口：`Fiber._reload` 在 `apply` 成功后重算输入快照，但 `snapshot is None`（apply 期间
    必需依赖被移除）时只跳过 epoch 更新，仍迁移到 `ACTIVE`，形成「`ACTIVE` + 输入不可解」；
    此时 `Fiber._inputs is None`、`ctx.get(必需键)` 抛 `KeyError`。不能指望后续传播补救：
    触发撤销的那次 `notify` 可能在 `LOADING` 期被 `_refresh` 早退吞掉。
  - 缺口二：`_replace` 在 `_reload` 后只区分 `ACTIVE` / 其它，若 `_reload` 自行降级为
    `INACTIVE`，会被误计为一次回滚并留下无意义的 `_pending`。
  - 改动：
    - `plugin.py::_reload`：`snapshot is None` 时置 epoch 为 `_INACTIVE`，依迁移表先落
      `ACTIVE` 再由 `_deactivate` 撤销刚登记的效应并收敛到 `INACTIVE`；
    - `plugin.py::_replace`：`_reload` 后 `state is INACTIVE` 直接返回，不计 `reloads` /
      `rollbacks`，不设 `_pending`、不记 `reload_reason`；
    - `test_hot_reload.py` 新增两条重入用例（初次装配、热替换各一）；`test_lifecycle.py`
      新增「`ACTIVE ⇒ 输入可解」不变量机检；
    - `doc/okf/architecture.md`：状态机小节补充互锁说明。
- **Acceptance Criteria Addressed**: §5.5 不变量「`ACTIVE` 必持有可解输入」可机检；热替换中
  依赖消失不再产生伪回滚与失效 `_pending`
- **Test Requirements**:
  - `rule` TR-P2.9-1: apply 期间依赖消失时，事务未排空即已收敛到 INACTIVE 且零效应；证据：
    `test_reload_converges_to_inactive_when_inputs_vanish_during_apply`
  - `rule` TR-P2.9-2: 热替换的新装配期间依赖消失：终态 INACTIVE，`reloads`/`rollbacks`
    均为 0、`pending_reload` 为 None、效应归零；证据：
    `test_hot_reload_with_vanishing_dependency_is_neither_reload_nor_rollback`
  - `rule` TR-P2.9-3: `ACTIVE ⇒ _inputs is not None 且 epoch ≠ _INACTIVE`；证据：
    `test_active_implies_resolvable_inputs`
- **Completion Evidence**（2026-11-21 实测）:
  - 第一批 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests`
    → **197 passed**（P2-8 后基线 194，新增 3）；
  - `tools/okf` 独立运行 → **259 passed**（未受影响）；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 58 文件已格式化；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src tools/xs/src`
    → `Success: no issues found in 40 source files`；
    `check_pyproject_style.py`（16 个 pyproject）、`check_dependency_direction.py`
    （337 文件）、`check_spec_integrity.py`（35 个任务编号）通过；
    `check_declaration_consistency.py` 仍为既有 2 项失败（`libs/mystx` 指针漂移与快照过期）；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - **V1 瞬时 ACTIVE**：`LOADING → ACTIVE → INACTIVE` 是迁移表约束下的唯一合法路径，
    `_transition` 不发布事件，方法返回时外部只见终态；语义正确性优先于瞬时状态。
  - **V2 卸载失败优先级**：`_deactivate` 中撤销失败会保留 `FAILED`，`_replace` 遂按回滚记录。
    这是 P2-8 的既有语义（卸载失败优先暴露），`retry()` 会重新计算 epoch，故不会误重试。
  - **V3 为何不直接 `LOADING → INACTIVE`**：迁移表不允许该边；放宽会削弱「装配必有结论」
    的约束，故选择委托 `_deactivate` 复用既有撤销逻辑，而不是新增边或复制卸载代码。
  - **V4 副带修正**：`_epoch` 置为 `_INACTIVE` 后，「撤销后重供同一实现」从
    `new_epoch == _epoch` 分支落到 `else` 分支，两条分支都会重新装配，行为等价；
    后者同时把 epoch 更新为真实身份，消除 epoch 滞后于输入快照的隐患。
  - **V5 测试用外层事务**：`with ctx.batch(): ctx.plugin(...)` 用的是公共 API，对应
    「装配嵌套在既有传播事务内」的真实调用形态；断言在事务未排空时成立，正好证明
    收敛不依赖通知补救。

## Task P2-10-1: 落地 ADR-05 多提供者定序

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-9-1（`f42c37d`）
- **Description**:
  - 缺口：ADR-05 三段规则在运行时侧零落地——`ServiceRegistry.lookup` 直接取 `providers[0]`，
    注册顺序成了语义身份；`ServiceProvider` 无 `priority` / `id` 字段，消费者无从显式指定，
    并列时既不报错也无法覆盖。清单/求解侧已部分落地（`_provider_index` 按 `(-priority, id)`
    定序；`_find_conflicts` 对最高优先级并列报 `SolveError(CONFLICT)`），唯独运行时与 ADR-05
    直接冲突。
  - 改动：
    - `service.py`：新增 `AmbiguousProviderError(RuntimeError)`（携带 `name` 与候选 id 元组）；
      `ServiceProvider` 追加 `priority: int = 0`、`id: str = ""`（`config` 保持第三位，
      兼容既有位置参数调用）；
    - `service.py::lookup`：新增仅关键字参数 `provider`，委托 `_select` 按三段规则裁决；
      无任何提供者仍返回 `None`；
    - `service.py::_select`：显式 id 优先（未知 id → `LookupError`；同 id 重复 →
      `AmbiguousProviderError`）；否则取最高 `priority`，并列即 `AmbiguousProviderError`
      （缺省 id 以 `priority=<n>` 标注候选）；
    - `xuan_plugin/__init__.py` 导出 `AmbiguousProviderError`；
    - 新建 `libs/xuan-plugin/tests/test_service.py`（13 条用例）；
      `tools/okf/tests/test_service.py` 的「插入到最前」用例改为「抬优先级透明切换 +
      显式回退」；
    - `doc/okf/architecture.md`：Capability Seam 小节改写为 ADR-05 三段规则。
- **Acceptance Criteria Addressed**: ADR-05 三条规则在运行时全部可机检；「最终配置与插入
  顺序无关」得以成立
- **Test Requirements**:
  - `rule` TR-P2.10-1: 多提供者按 `priority` 取最高，正序/逆序注册结果一致；证据：
    `test_highest_priority_wins_regardless_of_registration_order`（2 参数）
  - `rule` TR-P2.10-2: 最高 `priority` 并列且未显式指定 → `AmbiguousProviderError`，
    候选 id 全量列出，绝不返回首个；证据：`test_tied_priority_raises_instead_of_taking_first`、
    `test_tied_priority_lower_levels_do_not_matter`、`test_blank_id_falls_back_to_priority_label_in_error`
  - `rule` TR-P2.10-3: 显式 `provider` 覆盖 priority；并列可被显式指定解除；未知 id →
    `LookupError`；同 id 重复 → `AmbiguousProviderError`；证据：
    `test_explicit_provider_overrides_lower_priority`、`test_explicit_provider_resolves_tie`、
    `test_explicit_provider_unknown_raises_lookup_error`、`test_explicit_provider_duplicate_id_raises_ambiguous`
  - `rule` TR-P2.10-4: 无提供者返回 `None`；显式 config 优先于 Provider config；证据：
    `test_lookup_without_provider_returns_none`、`test_explicit_config_overrides_provider_config`
- **Completion Evidence**（2026-11-23 实测）:
  - `libs/xuan-plugin/tests` → **114 passed**（HEAD 基线 101，新增 13）；
  - 第一批 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests
    libs/xuan-project/tests` → **234 passed**（44 + 114 + 7 + 34 + 35）；
  - `tools/okf` 独立运行 → **259 passed**；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 103 文件已格式化；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src libs/xuan-build/src tools/xs/src`
    → `Success: no issues found in 41 source files`；
    `check_pyproject_style.py`（18 个 pyproject）、`check_dependency_direction.py`
    （347 文件）、`check_spec_integrity.py`（P2-9 后 35 个任务编号）通过；
    `check_declaration_consistency.py` 仍为既有 2 项失败（`libs/tao` 指针漂移与快照过期）；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - **V1 破坏性**：同值直接报错会否把原本能跑的装配变砖？`ServiceRegistry` 当前无生产消费者
    （全仓检索仅 `__init__` 导出与文档提及），且求解侧 `_find_conflicts` 早已对同值 `multi`
    报 `SolveError(CONFLICT)`；运行时与求解侧对齐后口径唯一，不新增线上风险。
  - **V2 为何不按 id 字典序兜底**：ADR-05 的「未标注者按 id 字典序过渡」是清单静态图的迁移
    口（`build_graph`），不是运行时终态；运行时按名字猜会让「改个 id」静默换实现，规则 3
    明确要求报错。装配期权威判定在 `solve`，它已对同值报错。
  - **V3 端到端**：实机 `xs plugin apply` 三节点 ACTIVE、16 条命令、20 键对账一致；
    `tools/okf` 259 条测试全绿。
  - **V4 兼容性**：`ServiceProvider` 的 `config` 保持第三位，`priority` / `id` 追加在尾部且
    带默认值，位置参数与关键字参数调用均不变；`tools/okf` 既有用例仅一处「插入到最前」
    按新语义改写，其余原样通过。
  - **V5 同 id 重复的提示信息**：候选元组会出现 `("dup", "dup")`，信息量偏低；该状态是注册方
    违约（id 应唯一），错误已阻断而非静默取首个；把唯一性校验前移到 `register` 属新增约束，
    超出本轮范围，留待需要时评估。
  - **落地口径边界**：清单侧规则 2 的 `provider=` 覆盖字段仍属 lock/P5 范围，本轮只落地运行时
    参数 `lookup(..., provider="<id>")`；`build_graph`（诊断图）保留 `(-priority, id)` 过渡
    定序，不以它作为装配裁决依据。`retrospective-20261121.md`（已并入
    [doc 第二部 · C 阶段](../../../../libs/xuan-plugin/doc/15-retro-delivery.md)）中「ADR-05 未落地」是 P2-9
    时点记录，不改写历史复盘。

## Task P2-11-1: 同名插件装配的确定性

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-10-1（`13c6127`）
- **Description**:
  - 缺口：`Context.plugin` 以 `self._fibers[plugin.name] = fiber` 直接覆写。同名旧 Fiber
    若仍处于 ACTIVE / INACTIVE / FAILED 等非终态，其键与效应仍挂在旧实例上，而
    `dispose` 只遍历字典现值——旧实例既不卸载也不上报，成为孤儿：键留在 `_store`、
    效应永不撤销。原实现无任何冲突检测，装配顺序决定谁在册。
  - 改动：
    - `context.py`：新增 `PluginNameConflictError(RuntimeError)`（携带 `name` 与旧实例
      `state`，消息给出处置建议）；`Context.plugin` 在构造 Fiber 之前检查同名实例，
      非 `DISPOSED` 即拒绝覆写，已 `DISPOSED` 的名称可重新占用；
    - `xuan_plugin/__init__.py` 导出 `PluginNameConflictError`；
    - `libs/xuan-plugin/tests/test_assembly.py` 追加 4 条用例（ACTIVE 冲突、INACTIVE 冲突、
      DISPOSED 后复用、被拒装配零残留）；
    - `doc/okf/architecture.md`：Context 职责第 3 条补「名称独占与冲突异常」。
- **Acceptance Criteria Addressed**: 同名装配不再产生孤儿 Fiber 与键残留；冲突在写入任何
  状态之前抛出，失败路径零副作用
- **Test Requirements**:
  - `rule` TR-P2.11-1: 同名且未销毁 → `PluginNameConflictError`，首个实例状态与键不受影响；
    证据：`test_duplicate_plugin_name_is_rejected`
  - `rule` TR-P2.11-2: `DISPOSED` 为惰性终态，名称可被新实例重新占用；证据：
    `test_duplicate_name_reusable_after_dispose`
  - `rule` TR-P2.11-3: `INACTIVE` 同样占名，不得被覆写；证据：
    `test_conflict_also_applies_to_inactive_fiber`
  - `rule` TR-P2.11-4: 被拒绝的第二次装配不写入键或效应，整体 `dispose` 无失败项；证据：
    `test_rejected_conflict_leaves_no_orphan_effects`
- **Completion Evidence**（2026-11-23 实测）:
  - `libs/xuan-plugin/tests` → **118 passed**（P2-10 基线 114，新增 4）；
  - 第一批 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests
    libs/xuan-project/tests` → **238 passed**（44 + 118 + 7 + 34 + 35）；
  - `tools/okf` 独立运行 → **259 passed**（未改动 okf，与 P2-10 持平）；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 77 文件已格式化；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src libs/xuan-build/src tools/xs/src`
    → `Success: no issues found in 41 source files`；
    `check_pyproject_style.py`（18 个 pyproject）、`check_dependency_direction.py`
    （347 文件）通过；`check_spec_integrity.py`（本轮后 **37** 个任务编号）通过；
    `check_declaration_consistency.py` 既有 1 项失败（清单快照过期，与本轮无关）；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - **V1 为何不自动「先卸旧再装新」**：自动替换会让调用方持有的 Fiber 句柄与状态引用
    全部过期，且旧实例卸载失败时无法回滚已建的新实例；报错把决策交还调用方，与 P2-10
    「不猜、不静默兜底」一致。
  - **V2 为何放行 DISPOSED**：终态实例无键、无效应、无待重试目标，`_refresh_dependents`
    也不会再传播到它；继续封锁名称只会让「重装同一插件」永远失败。
  - **V3 实机影响**：装配器按清单 `id` 唯一构造节点，正常路径不触发；冲突仅出现在手工
    重装与测试中，故不构成线上行为变更。
  - **V4 相邻问题（测试暴露，未修）**：`Context.provide` 返回的 undo 若不显式登记为
    effect，Fiber 卸载不会回收该键（本轮用例一度按「dispose 后键消失」断言而失败，已
    改为按真实语义断言）。是否让 `provide` 自动登记效应，留作后续候选。
  - **V5 报错时机**：检查位于 Fiber 构造与 `batch` 之前，拒绝路径不产生任何键、效应或
    Fiber 实例；异常携带旧状态，便于区分 ACTIVE（需先卸载）与 INACTIVE（需先补依赖）。

## Task P2-12-1: 供应即撤销（`provide` 自动登记效应）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-11-1（`eada0de`）
- **Description**:
  - 缺口：`Context.provide` 只写 `_store` 并返回一个**游离**的 undo——它不被任何
    `DisposableList` 持有。插件若在 `apply` 里直接 `ctx.provide(...)`（`tools/okf` 的 7 个
    插件即如此），Fiber 卸载/热替换后键仍留在 `_store`：新版本若少供该键，依赖方会读到
    上一代实现，形成「已卸载仍可供」的隐式后门。此前只能靠作者记得写
    `ctx.effect(lambda: ctx.provide(...))` 维持，属纪律而非内核保证（P2-11 V4 已记录）。
  - 改动：
    - `context.py`：`provide` 自行登记撤销——处于 `apply` 作用域时 `push` 到当前 Fiber 的
      `_disposables`，否则推入根 `DisposableList`；新增 `_provide_removers` 记录每个键的
      「登记摘除器」，同名重复供应先摘除旧登记再登记新撤销，保证**每键恰好一条有效登记**；
      新增 keyword-only 参数 `label`，非空时写入效应观测树（与原 `effect(label=...)` 同口径）；
      `dispose` 末尾清空该映射。
    - 调用点去重：`xuan_ffi._provide`、`xuan_build.apply`、`xs/plugins/commands.py` 由
      「`effect` 包 `provide`」改为「`provide(label=...)` + 必要的 `effect`」；测试中 11 处
      同样改写，避免双重登记。
    - `doc/okf/architecture.md`：Context 职责第 1、2 条改写为「供应自带撤销登记」。
- **Acceptance Criteria Addressed**: apply 期供应键的注销责任与效应归属同源；卸载、apply
  失败、热替换三条路径均不残留键；同名重复供应不累积登记
- **Test Requirements**:
  - `rule` TR-P2.12-1: apply 中直接 `provide`（不包 effect）归属当前 Fiber，`_unload` 后
    `registered_keys()` 不含该键且 `_disposables` 归零；证据：
    `test_bare_provide_is_registered_as_fiber_effect`
  - `rule` TR-P2.12-2: apply 中途抛错时，已写入的供应随 Fiber 回滚撤销；证据：
    `test_bare_provide_is_rolled_back_when_apply_fails`
  - `rule` TR-P2.12-3: 热替换后下游读到的是新装配写入的实现；证据：
    `test_bare_provide_follows_hot_swap`
  - `rule` TR-P2.12-4: 同名重复供应只保留最后一次登记（根层 2 次 → 1 条），`dispose` 后键与
    登记均归零；证据：`test_repeated_provide_keeps_one_registration_per_key`
  - `rule` TR-P2.12-5: `provide(label=...)` 的 label 进入 `get_effects()`；证据：
    `test_provide_label_is_recorded_as_effect_meta`
- **Completion Evidence**（2026-11-23 实测）:
  - `libs/xuan-plugin/tests` → **123 passed**（P2-11 基线 118，新增 5）；
  - 第一批 → **243 passed**（44 + 123 + 7 + 34 + 35）；
  - `tools/okf` 独立运行 → **259 passed**（okf 源码未改，其 7 个插件现在真正随 Fiber 回收）；
  - 门禁：`ruff check` 全部通过；`ruff format --check libs/xuan-plugin tools/xs libs/xuan-ffi
    libs/xuan-build` → 82 文件已格式化；`mypy libs/xuan-plugin/src libs/xuan-ffi/src
    libs/xuan-build/src tools/xs/src` → `Success: no issues found in 41 source files`；
    `check_pyproject_style.py`（18 个 pyproject）、`check_dependency_direction.py`（347 文件）、
    `check_spec_integrity.py`（本轮后 **38** 个任务编号）通过；`check_declaration_consistency.py`
    既有 1 项失败（清单快照过期，与本轮无关）；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - **V1 双重包装怎么办**：旧写法 `ctx.effect(lambda: ctx.provide(...))` 仍能工作——提供撤销
    与 effect 句柄都会被登记，逆序执行时第二次是幂等空操作；但登记数会多一条，故本轮把仓库内
    11 处测试与 3 处生产代码一并改写，并在 `provide` 文档串写明「勿再包 effect」。
  - **V2 为何跨 Fiber 摘除旧登记**：键的归属本来就遵循「最后一次写入」，注销责任跟随同一归属
    才有唯一答案；摘除只移除旧 `DisposableList` 中的一条元素，而 `DisposableList.clear()` 先
    快照后清空，卸载途中的摘除不会跳项。
  - **V3 根层登记的增量**：根层 `provide` 现在也占一条 `_root_disposables`；`dispose` 会逐键
    撤销并 `notify`（`flush=False` 丢弃），规模与键数同阶，未见性能风险。
  - **V4 为什么提供 `label` 而不是丢掉标签**：`xuan_build` / `xuan_ffi` / `xs commands` 原本用
    `effect(label="provide:<key>")` 让供应显形于效应观测树；`provide(label=...)` 让标签跟随
    动作本身，`get_effects()` 内容与改动前逐项一致。
  - **V5 未纳入本轮的相邻项**：`build_graph`（诊断图）仍按 `(-priority, id)` 取首个提供者，
    与 ADR-05「同优先级必须显式选择」的口径尚未收口；`okf` 插件中为挂标签而存在的空效应
    （`ctx.effect(lambda: [lambda: None], label=...)`）亦未改动。

## Task P2-13-1: 诊断图与求解器冲突同口径（`build_graph` 不再静默择一）

- **Status**: `completed`
- **Priority**: medium
- **Depends On**: P2-12-1
- **Description**:
  - 缺口：`manifest.build_graph` 只把「全 `single` 多提供者」记为冲突，对「含 `multi`
    但 `priority` 并列」仍按 `(-priority, id)` 取首个提供者作为依赖边终点；而
    `solver._find_conflicts` 对后一类判 `CONFLICT`。结果是同一份清单，`xs plugin graph`
    显示 `inject K → a`，`xs plugin apply` 却报冲突——诊断图给出了运行期不存在的裁决。
  - 改动：
    - `manifest.py` 新增 `find_conflicts(index)`（ADR-05 口径）：全 `single` 多提供者，
      或含 `multi` 但 `priority` 不两两不同，均判冲突；结果按键排序、提供者保持
      `(-priority, id)` 定序。`solver._find_conflicts` 改为委托它，两个入口不再各写一份。
    - `build_graph`：`conflicts = find_conflicts(index)`；冲突键的依赖边 `provider` 置
      `None`（不指向任何提供者）；`unresolved` 改为「该键在候选集合内没有任何提供者」，
      冲突键不再被误报为缺失。
    - `to_dot`：`provider is None` 时区分 `AMBIGUOUS`（冲突）与 `MISSING`（缺失）。
    - `plugin_cmd.graph`：文本输出为冲突键单独打印「→ 冲突（无唯一胜出者）」，并修正
      冲突小标题的口径（不再写「未声明 multi」）。
- **Acceptance Criteria Addressed**: 诊断图与运行期裁决对同一清单给出同一组冲突键；
  冲突键不得被渲染为「已解析到某提供者」，也不得被计为「缺失」
- **Test Requirements**:
  - `rule` TR-P2.13-1: `multi` 同优先级并列进 `conflicts`，依赖边 `provider is None`，
    `unresolved` 为空，DOT 含 `AMBIGUOUS`；证据：
    `test_build_graph_multi_tie_is_conflict_without_winner`
  - `rule` TR-P2.13-2: 全 `single` 多提供者的依赖边同样不指向任何提供者；证据：
    `test_build_graph_single_conflict_edge_has_no_provider`
  - `rule` TR-P2.13-3: 同一清单下 `solve` 抛 `CONFLICT` 且 `build_graph.conflicts` 非空，
    两组冲突键相同；证据：`test_build_graph_and_solve_agree_on_conflicts`
  - `rule` TR-P2.13-4: `multi` 但 `priority` 不同时不判冲突（免冲突条件收窄为「可裁决」）；
    证据：`test_build_graph_single_conflict_and_multi_with_distinct_priorities`
  - `rule` TR-P2.13-5: `xs plugin graph` 文本对冲突键输出「→ 冲突」，且不输出「→ 未解析」；
    证据：`test_plugin_graph_text_marks_ambiguous_key`
- **Completion Evidence**（2026-11-23 实测）:
  - `libs/xuan-plugin/tests` → **126 passed**（P2-12 基线 123，新增 3）；
  - 根 `tests/` → **45 passed**（基线 44，新增 1）；
  - 第一批 → **247 passed**（45 + 126 + 7 + 34 + 35）；
  - `tools/okf` → **259 passed**（无回归）；
  - 门禁：`ruff check` 全部通过；`ruff format --check libs/xuan-plugin tools/xs libs/xuan-ffi
    libs/xuan-build tests` → 89 文件已格式化；`mypy` → 41 源文件无问题；
    `check_pyproject_style.py`（18）、`check_dependency_direction.py`（347）、
    `check_spec_integrity.py`（本轮后 **39** 个任务编号）通过；`check_declaration_consistency.py`
    既有 1 项失败（清单快照过期，与本轮无关）；
  - 实机：`xs plugin graph` → 「插件 3 个，能力键 20 个，依赖边 1 条，未解析 0，冲突 0」；
    `xs plugin apply` → 3 节点 ACTIVE、16 条命令、20 键对账一致。
- **Notes**:
  - **V1 为什么不在 `build_graph` 抛错**：`graph` 是诊断命令，抛错会让排障者连图都看不到；
    冲突已是 `PluginGraph` 的一等数据（`conflicts`），与 `apply` 的 `kind=conflict` 同义。
    本轮选择「如实标注」，而非「拒绝渲染」。
  - **V2 `unresolved` 语义变更的影响面**：全仓只有 `plugin_cmd.graph`（文本与 JSON）消费它；
    冲突键被计为缺失本就是错误事实，故改为「键无任何提供者」后才算缺失，并新增用例锁定。
  - **V3 共享函数为何放在 `manifest.py`**：冲突规则由「声明形态」决定，与拓扑算法无关；
    `build_graph` 与 `solver` 都要用，放在两者共同的下层可避免循环导入；未加入
    `xuan_plugin.__init__` 导出，避免扩大公开 API。
  - **V4 免冲突条件被收窄**：旧文档口径是「显式声明 multi 即免冲突」，但它与 §5.4 的
    「priority 两两不同」相冲突；本轮以运行期实现为准，multi 仅在同优先级可裁决时免冲突，
    并在测试名中显式写出 `distinct_priorities`。
  - **V5 未纳入本轮的相邻项**：`provider="<id>"` 显式选择尚未出现在清单的 inject 声明中，
    因此诊断图目前只需与 priority 裁决对齐；若后续引入声明级显式选择，`build_graph` 与
    `solve` 必须同步扩展，届时以本轮的「同口径」用例为回归基线。

---

> 复盘报告（P2-4～P2-11 全链，R → I → E → C，G1–G4 通过）：
> 已原子化并入 [libs/xuan-plugin/doc/README.md（第二部）](../../../../libs/xuan-plugin/doc/README.md)，原独立文件已删除。

