> 分卷来源：`tasks.md` 原文 L1112–L1779（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：06-p2-b-lifecycle.md · 下一卷：（无）
> 主题：P2-C：不变量收口、互锁、ADR-05、同名装配、供应即撤销、诊断图同口径
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
  - **后续（2026-11-23）**：`okf` 6 个插件中的 7 处空效应已删除，标签改由
    `ctx.provide(..., label=...)` 承载；`get_effects()` 内容与改动前一致，`tools/okf`
    仍为 259 passed。

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

## Task P2-14-1: 通知排空失败保留待处理集合并可重试

- **Status**: `completed`
- **Priority**: medium
- **Depends On**: P2-12-1
- **Description**:
  - 缺口：`_flush_notifications` 在排空 100 轮仍未收敛时于 `_deferred.clear()` **之前**抛
    `RuntimeError`，残留待处理键被保留；但事务外 `notify` 只刷新「本次新键」，完全忽略残留
    `_deferred`。若后续通知的键与任何 Fiber 都无关（如向无消费者的能力键 `provide`），
    `_refresh_dependents` 不会命中任何 Fiber，也就不会触发嵌套排空——残留键被永久搁置，
    依赖方停在过期视图上，且没有任何观测口提示「还有未处理的通知」。
  - 改动：
    - `Context.notify`：事务外先判 `_deferred` 是否非空。非空时把本次键并入 `_deferred` 并
      调用 `_flush_notifications()` 重试；为空时保留原 `_refresh_dependents` 快路径。批次内
      分支与快路径语义均不变。若残留仍未收敛则继续抛错，使病态环持续可见，而非静默吞掉。
    - `_flush_notifications`：docstring 写明「超限时不修改 `_deferred`、由下一次 `notify`/
      `batch` 重试、`dispose` 以 `flush=False` 显式丢弃、`_batch_depth` 由 `finally` 复位」；
      错误信息补上「上限 N 轮」。
- **Acceptance Criteria Addressed**: 排空失败后 `_deferred` 保留可见；事务深度复位；后续任一
  通知都能把残留键重新排空；失败后的上下文可安全 `dispose`
- **Test Requirements**:
  - `rule` TR-P2.14-1: 构造双向振荡（A 供 x 依赖 y、B 供 y 依赖 x，且每轮供应递增值）触发
    超限错误后，`_deferred` 非空且 `_batch_depth == 0`；停止振荡后一次「与任何 Fiber 无关」
    的 `provide` 能把残留排空，且 A/B 均 ACTIVE、x/y 均注册；证据：
    `test_flush_overflow_keeps_pending_and_next_notify_drains`
  - `rule` TR-P2.14-2: 排空失败后 `dispose()` 不抛错、`registered_keys()` 为空、`_deferred`
    清空、`_batch_depth == 0`；证据：`test_dispose_after_flush_overflow_is_safe`
- **Completion Evidence**（2026-11-23 实测）:
  - `libs/xuan-plugin/tests` → **128 passed**（P2-13 基线 126，新增 2）；
  - 根 `tests/` → **45 passed**；第一批 → **249 passed**（45 + 128 + 7 + 34 + 35）；
  - `tools/okf` → **259 passed**（无回归）；
  - 门禁：`ruff check` 全部通过；`ruff format --check` → 518 文件已格式化（全仓无参扫描）；
    `mypy` → 41 源文件无问题；`check_pyproject_style.py`（18）、
    `check_dependency_direction.py`（347）、`check_spec_integrity.py`（本轮后 **40** 个任务编号）
    通过；`check_declaration_consistency.py` 既有 1 项失败（清单快照过期，与本轮无关）；
  - 实机：`xs plugin graph` → 「插件 3 个，能力键 20 个，依赖边 1 条，未解析 0，冲突 0」；
    `xs plugin apply` → 3 节点 ACTIVE、16 条命令、20 键对账一致。
- **Notes**:
  - **V1 为什么不在失败时直接丢弃残留**：丢弃会制造「静默过期视图」，违反 §5.4 第 9 步「依赖
    全可解却 INACTIVE 不可接受」的同一判据。保留 + 重试是本任务的第一性目标；病态未消除时再次
    抛错是期望的「持续可见」，一旦消除（如振荡停止）残留即被正常排空。
  - **V2 是否改变事务外同步语义**：仅在「上一次排空异常退出」时 `_deferred` 才非空；正常
    batch 退出总会清空它。因此无残留时的快路径与原实现逐字节等价，既有 126 用例回归验证。
  - **V3 为什么留在 `notify` 而不新开公开方法**：残留重试是 `notify` 的既有契约（「通知必须
    被处理」）的自然补齐，不需要调用方额外感知；不新增 API，避免扩大公开表面。
  - **V4 轮次上限对合法深链的误伤（未纳入本轮）**：100 轮对「101 级合法依赖深链」同样会触发
    错误。该上限是 P2-6 的既有设计，本任务不改变其语义，仅改变失败后的残留处理；若未来出现
    真实深链需求，再评估改为「宽度上限 + 深度上限」或可配置。
  - **V5 `plugin()` 内 flush 抛错留下半注册 Fiber（未纳入本轮）**：`_fibers` 已写入但
    `_refresh` 未完成，属装配原子性缺陷，与残留状态正交。本轮以
    `test_dispose_after_flush_overflow_is_safe` 保证「可安全回收」，半注册问题留待下一任务，
    避免单提交混多职责。
  - **V6 振荡用例是否依赖 `_digest` 的实现细节**：用例依赖「同一整数键每次供应不同值 → epoch
    变化 → 继续重载」。若 `_digest` 改为按类型取摘要，用例会因不再振荡而在
    `pytest.raises(..., match="未收敛")` 处失败，不会静默「通过」；当前 `_digest` 对基础类型
    走 `f"{type(value).__name__}:{value!r}"`，契约明确。

## Task P2-15-1: 排空轮次上界改为按 Fiber 数推导，消除对合法深链的误伤

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-14-1（`5b9a900`）；处置 P2-14 Notes V4 遗留
- **Description**:
  - 缺口：`_MAX_FLUSH_ROUNDS = 100` 是固定常量，而合法依赖链的排空轮次随链深线性
    增长——101 级以上的合法级联会在第 101 轮被误判为病态环，抛「通知排空 … 轮仍未
    收敛」，把正常装配变成不可用。
  - 判据：合法图每一轮至少把变更推进一层（本轮处理的键必然由上一轮某次装配产生），
    DAG 的路径长度不超过 Fiber 数，故合法 DAG 的轮次不超过 Fiber 数；收敛的环状图
    （上游停止振荡后的恢复路径）至多再绕一圈。据此把上界改为 `2 × Fiber 数 + 2`，
    并与 `_FLUSH_ROUNDS_CEILING = 10_000`（资源护栏，非语义上限）取小。
  - 改动：
    - `context.py`：删除 `_MAX_FLUSH_ROUNDS`，新增 `_flush_round_limit()`（每轮按当前
      Fiber 数重算，动态装配时上界自动抬高）与护栏常量；超限错误信息改为
      「轮次上界 N 轮，当前 Fiber 数 M」；
    - `test_notification_batch.py`：新增 120 级合法深链收敛用例、病态环按 Fiber 数
      上界失败的用例。
- **Acceptance Criteria Addressed**: 合法深链不得被轮次上限误伤；病态环仍在有限轮次内
  转成可诊断错误；P2-14 的「残留保留 + 下次通知重试」语义不变
- **Test Requirements**:
  - `rule` TR-P2.15-1: 120 级链式 inject/provide 在单次通知后全部 ACTIVE、`_deferred`
    清空、注册键数 = 121；证据：`test_deep_legal_chain_needs_one_round_per_level`
  - `rule` TR-P2.15-2: 2 个 Fiber 的振荡环在「上界由 Fiber 数推导」处抛错且残留键保留；
    证据：`test_pathological_cycle_hits_fiber_count_bound`
  - `rule` TR-P2.15-3: 既有 P2-14 残留重试与拆除安全用例不回归；证据：
    `test_flush_overflow_keeps_pending_and_next_notify_drains`、
    `test_dispose_after_flush_overflow_is_safe`
- **Completion Evidence**（2026-11-23 实测）:
  - `libs/xuan-plugin/tests` → **130 passed**（P2-14 基线 128，新增 2）；
    第一批 → **216 passed**（45 + 130 + 7 + 34）；
  - `tools/okf` → **259 passed**（无回归）；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 89 文件已格式化；
    `mypy` → 41 源文件无问题；`check_pyproject_style.py`（18）、
    `check_dependency_direction.py`（347）、`check_spec_integrity.py`
    （本轮后 **41** 个任务编号）通过；`check_declaration_consistency.py` 既有 1 项
    失败（清单快照过期，与本轮无关）；
  - 实机：`xs plugin apply` → 3 节点 ACTIVE、16 条命令、20 键对账一致；
    `xs plugin graph` → 3 插件 / 20 键 / 未解析 0 / 冲突 0。
- **Notes**:
  - **V1 为什么是 2 × Fiber 数 + 2**：DAG 上界为 Fiber 数；P2-14 的恢复路径实测需要
    「链深 + 环上再绕一圈」，故留一倍余量并加一轮收尾。
  - **V2 被否方案一：直接调大常量**（1000/10000）——仍会误伤更深合法链，且病态环的
    失败时延线性上升，治标不治本。
  - **V3 被否方案二：同态重复检测**——P2-14 的振荡环每轮键集恒为 `{x, y}`，但实现摘要
    每轮都变，按「键集是否重复」判定会漏检；合法深链每轮键集也不同，该判据无理论边界。
  - **V4 被否方案三：每 Fiber 重载次数上限**——合法 DAG 中同一 Fiber 可因多条上游路径
    多次重载（菱形依赖），次数上限难取，且需额外状态，复杂度高于轮次上界。
  - **V5 被否方案四：一次性拓扑排序求解**——会改变「一次变更一次传播」的观测语义
    （apply/undo 顺序、静态插件抑制计数），破坏既有 11 条传播用例。
  - **V6 异步 apply 的复核点**：上界依赖「每轮推进一层」；若将来 apply 支持 await
    交错，需重新论证。`_flush_round_limit()` 每轮按当前 Fiber 数重算，动态装配场景
    已自动适配。

## Task P2-16-1: 装配失败回滚半注册 Fiber，允许同名重装

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-15-1（`3016e09`）；P2-15 收口时建议单独处理的装配原子性缺口
- **Description**:
  - 缺口：`Context.plugin()` 先把新 Fiber 写入 `_fibers`，再在 `with self.batch()`
    中执行 `fiber._refresh()`；若事务退出时的 `_flush_notifications()` 抛错
    （如病态环超出轮次上界），Fiber 已登记且已产生供应与效应，却既未返回也未
    撤销——成为「半注册」。后续同名 `plugin()` 因 `existing.state is not
    DISPOSED` 抛 `PluginNameConflictError`，该名称被永久占用而无法重装。
  - 改动：把登记与启动加载包进 `try`；异常时先撤销 `_fibers` 登记（装配前无旧
    实例则删除该键，旧实例为 `DISPOSED` 则恢复其引用），再在
    `with self.batch(flush=False)` 内 `fiber.dispose()` 撤销本次已登记的效应。
    `flush=False` 与 `dispose` 同口径：排空已失败，清理期的通知不得再次排空同一
    病态图，避免二次抛错掩盖原始异常；遗留的待处理键随拆除丢弃。
  - docstring 明确「装配是原子的」契约。
- **Acceptance Criteria Addressed**: 装配失败不得留下半注册 Fiber；同名插件在失败后
  仍可重装；成功装配路径语义不变
- **Test Requirements**:
  - `rule` TR-P2.16-1: 排空阶段失败的装配不得留下 `_fibers` 占位；证据：
    `test_plugin_raises_without_leaving_half_registered_fiber`
  - `rule` TR-P2.16-2: 装配失败后同名插件可重新装配并达到 ACTIVE；证据：
    `test_failed_plugin_does_not_block_same_name_registration`
  - `rule` TR-P2.16-3: 既有装配、传播与拆除用例不回归；证据：
    `libs/xuan-plugin/tests` 132 passed
- **Completion Evidence**（2026-11-23 实测）:
  - 修复前：两条新用例均失败（C 残留为 ACTIVE；同名重装抛
    `PluginNameConflictError`）；
  - 修复后：`libs/xuan-plugin/tests` → **132 passed**（P2-15 基线 130，新增 2）；
    根与三库 → **223 passed**；`tools/okf` → **259 passed**；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 38 文件已格式化；
    `mypy` → 41 源文件无问题；`check_pyproject_style.py`（18）、
    `check_dependency_direction.py`（347）、`check_spec_integrity.py`
    （本轮后 **42** 个任务编号）通过；
  - 实机：`xs plugin apply` → 3 节点 ACTIVE、16 条命令、20 键对账一致；
    `xs plugin graph` → 3 插件 / 20 键。
- **Notes**:
  - **V1 清理为何用 `flush=False`**：失败发生在排空内部；若清理期
    `fiber.dispose()` 撤销供应时触发的 `notify` 再次排空，会在同一振荡图上二次
    抛错并掩盖原始异常。`batch(flush=False)` 让通知只积压、退出时丢弃，与
    `Context.dispose()` 同口径。
  - **V2 被否方案一：先 dispose 再回滚登记**——若 dispose 抛错，登记将残留；
    先撤销登记可保证即使拆除失败也不留半注册占位。
  - **V3 被否方案二：只删除 `_fibers` 键、不撤销效应**——供应与效应会成为孤儿，
    正是 P2-11 禁止的形态；必须同时 `dispose` 本次 Fiber。
  - **V4 被否方案三：把 `plugin()` 拆成两阶段（先构造后提交）**——会改变公开 API
    与全部调用点的语义，超出本缺口范围；当前 try/except 已满足原子性。
  - **V5 回滚边界**：只撤销本次装配的 Fiber 自身；振荡中被激活的既有 Fiber 保持
    其自洽快照（输入快照与已供应键一致），不回溯到装配前——Python 进程内无法
    回滚外部副作用（ADR-03），强制回溯会破坏既有热替换语义。
  - **V6 `except BaseException` 的必要性**：`Ctrl-C`/`SystemExit` 同样应走回滚，
    否则中断后留下半注册；清理完成后以裸 `raise` 原样重抛，不吞异常。
  - **V7 测试对私有字段的依赖**：`_fibers` 是「是否残留登记」的唯一可观测口径
    （公开 API 不暴露）；对外契约由第二用例（同名重装成功、状态 ACTIVE）覆盖。

## Task P2-17-1: `xs plugin graph` 冲突与必需未解析以非零退出码收口

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-15-1（`3016e09`）
- **Description**:
  - 缺口：`xs plugin graph` 的文本/JSON/DOT 三种输出在存在「能力键冲突」或「必需依赖
    未解析」时仍以退出码 0 结束——诊断只对人可见，CI 与脚本无法据此拦截；同一份清单下
    `xs plugin apply` 会以退出码 1 报不可解，两个命令的口径不一致。
  - 判据：退出码表达「装配是否可解」。冲突键无唯一胜出者、必需依赖无任何提供者，均为
    不可解；可选依赖缺失不影响可解性。故仅当前两者非空时退出 1，其余维持 0。
  - 改动：
    - `plugin_cmd.py`：新增 `_graph_diagnostics()` 统一生成诊断摘要（冲突在前、未解析在
      后）；文本输出「诊断摘要（N 项，退出码 1）」；JSON 增加 `status`（`ok`/`error`）与
      `diagnostics`，既有 nodes/edges/unresolved/conflicts 不变；DOT 在图后用 `//` 注释
      追加同一摘要；三种模式结束后统一 `Exit(1)`。
    - `test_plugin_cmd.py`：新增 5 条用例（冲突退出 1、必需未解析退出 1、可选缺失仍 0、
      JSON 字段、DOT 注释、三模式同源摘要），并同步既有成功路径断言。
    - `doc/cli/index.md`：补充退出码与共享摘要说明。
- **Acceptance Criteria Addressed**: 冲突/必需未解析 → 非零退出码；文本/JSON/DOT 共用
  一份错误摘要；诊断图仍完整可查看；可选依赖缺失不误判为失败
- **Test Requirements**:
  - `rule` TR-P2.17-1: 冲突或必需未解析时三种输出退出码均为 1，摘要逐行一致；证据：
    `test_plugin_graph_text_marks_ambiguous_key`、
    `test_plugin_graph_unresolved_required_exits_nonzero`、
    `test_plugin_graph_json_carries_status_and_diagnostics`、
    `test_plugin_graph_dot_appends_diagnostics_as_comments`、
    `test_plugin_graph_three_modes_share_diagnostics`
  - `rule` TR-P2.17-2: 可选依赖未解析与正常清单维持退出码 0；证据：
    `test_plugin_graph_optional_unresolved_keeps_zero`、`test_plugin_graph_json`、
    `test_plugin_graph_dot`
  - `rule` TR-P2.17-3: 含诊断的 DOT 仍可被 Graphviz 渲染；证据：实机 `dot -Tsvg` 输出成功
- **Completion Evidence**（2026-11-23 实测）:
  - `tests/test_plugin_cmd.py` → **23 passed**（原 18，新增 5）；根与三库 →
    **223 passed**（45 + 132 + 7 + 34；其中 132 含并行任务 P2-16-1 新增的 2 条）；
    `tools/okf` → **259 passed**；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 518 文件已格式化；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src libs/xuan-build/src tools/xs/src`
    → 41 源文件无问题；`check_pyproject_style.py`（18）、
    `check_dependency_direction.py`（347）、`check_spec_integrity.py`（本轮后 43 个
    任务编号）通过；`check_declaration_consistency.py` 既有 1 项失败（清单快照过期，
    与本轮无关）；
  - 实机：`python -m xs.cli plugin graph` → 退出码 0（未解析 0 / 冲突 0）；
    `--json` → `status=ok`；`plugin apply` → 3 节点 ACTIVE、16 条命令、20 键对账一致。
- **Notes**:
  - **V1 被否方案一：新增 `--strict` 开关**——本轮目标就是默认收口；开关会让默认路径
    继续静默通过，需调用方额外记得打开，反而扩大配置面。
  - **V2 被否方案二：冲突/未解析分别用退出码 2/3**——仓库约定 2 表示脚本自身错误
    （三个治理脚本均如此），工具命令沿用 0/1；区分信息交由 `diagnostics` 与 JSON 字段。
  - **V3 被否方案三：摘要只写 stderr**——JSON 消费方会混入非 JSON 文本，DOT 消费方
    以重渲染为主而不读 stderr；三种输出各带同一份摘要才满足「共用」。
  - **V4 被否方案四：文本保留原两段标题、仅 JSON/DOT 新增摘要**——同源摘要才能保证
    三模式逐字一致（TR-P2.17-1），否则文案漂移无门禁可查。
  - **V5 口径说明**：可选依赖缺失不计入错误，与 `build_graph` 的 `unresolved` 定义及
    solver 可解性一致；若将来需要「可选也必须存在」的严格模式，应作为独立选项评估。
  - **V6 并行任务协调**：本轮开工时工作区另有装配原子性任务在改 `context.py` 与
    `test_notification_batch.py`，其先占用 P2-16-1，本任务顺延为 P2-17-1；上述文件不属
    本任务，未纳入本次变更与提交。

## Task P2-18-1: `xs plugin why <id>` 输出失活因果链（设计 V4 / 度量 M5）

- **Depends On**: P2-17-1（`85a4c31`）
- **Description**:
  - 缺口：设计文档 V4 要求 INACTIVE 携带「差哪个键、谁可能提供」，M5 要求定位根因
    小于 1 分钟；此前只能靠 `graph` 与 `apply --dry-run` 人工串联，缺少单插件的
    因果解释入口。
  - 语义：`xs plugin why PLUGIN_ID` 区分五种状态——已生效（ACTIVE）、未生效
    （INACTIVE）、已禁用（disabled）、未知 id（unknown）、装配计划无解（blocked）。
  - 信息项：当前状态与装配位次；每条 inject 的提供者与状态；缺失键的全部候选提供者
    （标注已生效/未生效）、被 `disabled` 排除的来源；提供者自身失活时逐层下钻到根因，
    每层给出可执行的修复建议。失活原因只在下钻段展示一次，不与「依赖」段重复。
  - 退出码：0 仅当被查询插件已生效；INACTIVE / disabled / unknown / blocked 均为 1。
    这是「该插件能否生效」的问答语义，与 `graph`/`apply` 的「装配计划是否可解」区分——
    INACTIVE 在求解层面是合法结果（`apply --dry-run` 仍返回 0），已在 CLI 文档写明。
  - 改动：
    - `plugin_cmd.py`：新增 `why` 子命令与 `_key_owners`、`MissingKey`、`_missing_key`、
      `_provider_text`、`_render_chain`、`_render_missing`；候选提供者沿用求解器的
      `(-priority, id)` 定序，等优先级并列已由 `solve` 以冲突拦截，不会出现双胜出者。
    - `test_plugin_cmd.py`：新增 7 条用例（ACTIVE 位次、未知 id、禁用来源、因果链、
      JSON 形态、可选降级、无解 blocked），本文件 23 → 30 条。
    - `doc/cli/index.md`：补充 `why` 用法、状态集、退出码语义与 `PLUGIN_ID` 参数。
- **Acceptance Criteria Addressed**: V4「INACTIVE 必须携带 reason 与差哪个键、谁可能
  提供」；M5「`xs plugin why` 小于 1 分钟定位根因」
- **Test Requirements**:
  - `rule` TR-P2.18-1: 已生效插件给出装配位次与已解析依赖，退出码 0；证据：
    `test_plugin_why_active_reports_position`、`test_plugin_why_optional_missing_is_degraded`
  - `rule` TR-P2.18-2: INACTIVE 逐键给出候选提供者、禁用来源与下钻因果链；证据：
    `test_plugin_why_inactive_names_disabled_source`、`test_plugin_why_chain_reaches_root_cause`、
    `test_plugin_why_json_payload`
  - `rule` TR-P2.18-3: 未知 id 与无解计划显式收口为 1，不伪装成可解释；证据：
    `test_plugin_why_unknown_id_exits_nonzero`、`test_plugin_why_reports_unsolvable_plan`
- **Completion Evidence**（2026-11-23 实测）:
  - `tests/test_plugin_cmd.py` → **30 passed**（原 23，新增 7）；根与三库 →
    **230 passed**（`tests` 57 + `xuan-plugin` 132 + `xuan-build` 7 + `xuan-ffi` 34）；
    `tools/okf` → **259 passed**；
  - 门禁：`ruff check` 全部通过；`ruff format --check` 518 文件已格式化；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src libs/xuan-build/src tools/xs/src`
    → 41 源文件无问题；`check_pyproject_style.py`（18）、
    `check_dependency_direction.py`（347）、`check_spec_integrity.py`（本轮后 44 个
    任务编号）、`check_declaration_consistency.py`（G1-G3 全部通过）；
  - 实机：`python -m xs.cli plugin why xs.commands` → 退出码 0，输出
    「状态：已生效（ACTIVE）装配顺序第 2 / 3 位」与 `xuan:build/backend#dispatch
    ← xuan.build`；`--json` → `status=active`、`position=2`、`order` 三节点；
    `why nope` → 退出码 1 并列出 `xs.commands, xuan.build, xuan.ffi`；
  - 失活场景（构造清单）：`a → b → 缺 xuan:demo/x#y` 输出「因果：b 缺少
    xuan:demo/x#y → 无提供者」，并建议先修 b；`disabled` 插件输出「已禁用（disabled）」。
- **Notes**:
  - **V1 被否方案一：只做 `why --explain` 布尔开关**——默认路径仍需人读 `graph`；
    诊断命令本身应当默认给全，开关只会让新人多查一次帮助。
  - **V2 被否方案二：复用 `apply --dry-run` 的退出码（INACTIVE 返回 0）**——那会
    把「询问某插件为何没生效」的答案伪装成成功；`why` 的退出码表达被查询插件状态，
    与计划可解性是两个问题，已在文档与 docstring 双处写明。
  - **V3 被否方案三：未知 id 时用编辑距离猜相似 id**——引入阈值调参与误猜风险；
    直接列出全部插件 id（当前 3 个）已足够。
  - **V4 被否方案四：blocked 时仍强行给出单插件链**——冲突/环下 `owner` 无定义，
    任何链都会是编造；改为输出 `kind`/`details` 并提示修复后重试。
  - **V5 边界：JSON 无 schema 版本字段**——只增字段、不改既有语义；若将来需要
    稳定契约，另立任务引入显式版本。
  - **V6 口径说明**：`_missing_key` 的深度上限 8 是纯防御；`_inactive_nodes` 的标记
    顺序（提供者先于消费者）保证失活链无环，正常清单不会触达上限。
  - **V7 并行任务协调**：本轮开工时 HEAD 为 `f43939d`（并行会话刷新声明快照），
    工作树干净；本任务只改 `plugin_cmd.py`、`test_plugin_cmd.py`、`doc/cli/index.md`
    与本台账，未纳入他人改动。

---

> 复盘报告（P2-4～P2-11 全链，R → I → E → C，G1–G4 通过）：
> 已原子化并入 [libs/xuan-plugin/doc/README.md（第二部）](../../../../libs/xuan-plugin/doc/README.md)，原独立文件已删除。

