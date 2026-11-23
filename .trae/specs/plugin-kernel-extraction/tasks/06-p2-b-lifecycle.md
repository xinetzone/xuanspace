> 分卷来源：`tasks.md` 原文 L841–L1111（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：05-p2-a-build-ffi.md · 下一卷：07-p2-c-invariants.md
> 主题：P2-B：生命周期、热替换、传播、摘要
## Task P2-4-1: 插件生命周期与状态机（§5.5 落地）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P0-2-1（epoch 结构化摘要）、P0-3-1（卸载异常聚合）、P2-3-1（`ea2fc0e`）
- **Description**:
  - `libs/xuan-plugin/src/xuan_plugin/plugin.py`：
    - `FiberState` 增加 `INACTIVE = 6`（追加在末尾，既有六态数值不漂移）；
    - 新增 `FiberTransitionError` 与 `_TRANSITIONS` 迁移表，表外迁移快速失败而非静默降级；
    - `_refresh` 在依赖不可解时统一进入 `INACTIVE`（初次刷新与原因集合变化均覆盖），
      `Fiber.reason` 返回缺失注入；`FAILED` 遇依赖消失时归一到 `INACTIVE`；
    - 新增 `Fiber.retry()`：仅 `FAILED` 可调用，依赖仍缺则退回 `INACTIVE`；
    - `DISPOSED` 为终态：`_refresh` / `retry` 抛 `FiberTransitionError`，二次 `dispose` 幂等；
    - `apply` / 撤销期间 `provide → notify` 的重入被短路，避免 `LOADING → LOADING` 表外迁移；
    - `Plugin.lifecycle`（`reloadable` | `static`）：静态插件 ACTIVE 后抑制热换，
      `Fiber.suppressed_reloads` 计数可观测；`dispose` 仍执行逻辑退役（ADR-03）。
  - `assembly.py`：未激活节点的失败信息携带 `state` 与缺失注入；
  - `libs/xuan-ffi`：`PLUGIN` 声明 `lifecycle="static"`；
  - `tools/okf` 测试与 `doc/okf/architecture.md` 同步七态契约（`PENDING` → `INACTIVE`）。
- **Acceptance Criteria Addressed**: 设计文档 §5.5 状态表与不变量；D7 / ADR-03
  （原生插件不承诺 `.so` 卸载，只承诺能力句柄可撤销）
- **Test Requirements**:
  - `rule` TR-P2.4-1: 依赖缺失时进入 `INACTIVE` 且 `reason` 给出缺失键，无残留效应；
    证据：`test_missing_dependency_enters_inactive_with_reason`、
    `test_reason_updates_when_partial_dependency_appears`
  - `rule` TR-P2.4-2: 依赖出现自动激活、依赖消失退回 `INACTIVE`；
    证据：`test_dependency_appearance_and_loss_transition`
  - `rule` TR-P2.4-3: `FAILED` 可显式 `retry()`，非 `FAILED` 调用被拒绝；`DISPOSED`
    终态且二次 `dispose` 幂等；证据：`test_failed_plugin_can_retry_explicitly`、
    `test_retry_is_only_legal_from_failed`、`test_disposed_is_terminal_and_idempotent`
  - `rule` TR-P2.4-4: 静态插件 ACTIVE 后不热换、不重复 `apply`，`dispose` 仍撤销句柄；
    证据：`test_static_plugin_suppresses_hot_reload`、
    `test_static_plugin_still_retires_handles_on_dispose`
  - `rule` TR-P2.4-5: `apply` 内自依赖写入不重入状态机；证据：
    `test_refresh_during_apply_does_not_reenter_state_machine`
- **Completion Evidence**（2026-11-21 实测）:
  - 联跑 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests`
    → **168 passed**（基线 157，新增 11，无删减）；
  - `tools/okf` 独立运行 → **259 passed**（六态断言同步为七态）；
  - 门禁：`ruff check` 与 `ruff format --check`（46 文件）通过；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src tools/xs/src`
    → `Success: no issues found in 40 source files`；
    `check_pyproject_style.py`（16 个 pyproject）、`check_dependency_direction.py`
    （335 文件）、`check_spec_integrity.py`（任务编号 30）、
    `check_declaration_consistency.py` 通过；
  - 实机：`xs plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；对账：一致 20 键，
    缺 0，多 0」。
- **Notes**:
  - `INACTIVE` 追加为枚举值 6 而非插入中间，避免日志/文档按数值比对时整体漂移（V2）；
  - 依赖 epoch 变化仍自动重载（P0-2 反应式语义，既有测试依赖）；`retry()` 只覆盖
    「epoch 未变但 `apply` 瞬时失败」的场景，两者语义互补（V4）；
  - `FAILED → INACTIVE` 与 `FAILED → UNLOADING` 是对 §5.5 主路径的补充边，分别用于
    「依赖消失」与「dispose 回收失败节点」，均满足「无效应、原因可解释」不变量（V6）；
  - 状态机保持单线程契约（与 `Context` 一致）；并发隔离留给 P2-3 的子进程模式（V7）。

## Task P2-5-1: 插件热更新事务与效应归属（§5.4 第 6–8 步）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-4-1（`f6fd5dd`，七态状态机与静态抑制）
- **Description**:
  - `libs/xuan-plugin/src/xuan_plugin/context.py`：
    - 新增 `Context._apply_scope(fiber)`（栈式上下文管理器）：`apply` 期间经
      `Context.effect` 登记的效应归属当前 Fiber，作用域外仍归属根 `DisposableList`；
    - `Context.effect` 的逆函数先从所属句柄清单摘除再执行，避免「手动撤销 +
      卸载重放」导致同一逆操作执行两次。
  - `libs/xuan-plugin/src/xuan_plugin/plugin.py`：
    - 新增 `Fiber._snapshot_inputs` / `_epoch_of` / `_describe_change`：依赖快照
      `(服务名, 实现摘要, 配置摘要)` 同时用于 epoch 计算与逐字段对账；
    - 新增 `Fiber._replace`：依赖 epoch 变化时执行「先撤旧效应、再应用新实现」的
      热替换事务；新实现失败时其部分效应由 `_reload` 回滚为零残留，状态 `FAILED`，
      `retry()` 可恢复；
    - 新增只读观测：`reloads`、`rollbacks`、`last_reload_error`、`reload_reason`
      （`dep:impl` / `dep:config` / `dep:added` / `dep:lost`）、`pending_reload`；
    - `_unload` 成功后清空输入快照；`dispose` 清理待重试目标；静态插件抑制仍先于
      替换判定（P2-4 语义不变）。
  - `libs/xuan-plugin/tests/test_assembly.py`：回滚期撤销失败的标签由 `root:` 改为
    `fiber:`——apply 期效应归属变化后的预期更新；
  - `doc/okf/architecture.md`：同步热替换与效应归属语义。
- **Acceptance Criteria Addressed**: 设计文档 §5.4 第 6–8 步（装/卸/换、逐字段对账、
  事务回滚）、§5.5 状态表；F11（热替换以备份—恢复实现事务性）
- **Test Requirements**:
  - `rule` TR-P2.5-1: `apply` 期效应归属 Fiber，卸载后归零且供应键撤销；
    证据：`test_apply_effects_belong_to_fiber`
  - `rule` TR-P2.5-2: 依赖实现变化完成热替换，旧供应撤销、新供应生效，
    `reloads` 与 `reload_reason` 可观测；证据：`test_hot_reload_undoes_old_and_applies_new`
  - `rule` TR-P2.5-3: 新实现失败时部分效应回滚为零残留，`FAILED` + `rollbacks` +
    `last_reload_error`；证据：`test_failed_reload_rolls_back_partial_effects`
  - `rule` TR-P2.5-4: 仅注入配置变化时重载理由为 `config` 而非 `impl`；
    证据：`test_config_change_reports_config_field`
  - `rule` TR-P2.5-5: 依赖消失时效应归零并进入 `INACTIVE`，原因可解释；
    证据：`test_dependency_loss_retires_fiber_effects`
  - `rule` TR-P2.5-6: `FAILED` 后显式 `retry()` 恢复并计入 `reloads`；
    证据：`test_retry_after_failed_reload_recovers`
  - `rule` TR-P2.5-7: 依赖方热替换后其下游键指向新实现；
    证据：`test_consumer_hot_swap_preserves_downstream_key`
- **Completion Evidence**（2026-11-21 实测）:
  - 联跑 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests`
    → **175 passed**（基线 168，新增 7，无删减）；
  - `tools/okf` 独立运行 → **259 passed**；
  - 门禁：`ruff check` 与 `ruff format --check`（47 文件）通过；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src tools/xs/src`
    → `Success: no issues found in 40 source files`；
    `check_pyproject_style.py`（16 个 pyproject）、`check_dependency_direction.py`
    （336 文件）、`check_spec_integrity.py`（任务编号 31）、
    `check_declaration_consistency.py` 通过；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」；`toolchain check` 无新增告警。
- **Notes**:
  - **V1 旧视图不可还原**：依赖实现被 `provide` 覆盖后，Python 进程内无法在原地
    恢复旧实现与外部副作用（ADR-03）。事务边界因此收敛为「不留半成品」：失败即
    撤销本次已登记效应并以 `FAILED` + 原始异常暴露，是否重试交给 `retry()`，
    不伪造「旧实现继续服务」；
  - **V2 根效应语义**：改道只发生在 `_apply_scope` 内；作用域外 `Context.effect`
    仍归根，`test_context.py` / `test_context_extra.py` 的根效应断言保持通过；
  - **V3 嵌套装配**：用栈而非单值，覆盖「apply 内 provide → notify → 依赖方就地
    装配」，内层 Fiber 独立归属并在退出时 LIFO 还原；
  - **V4 逆操作重放**：`effect` 返回的 undo 先摘除句柄再执行，卸载不会重放已手动
    撤销的效应；
  - **V5 重试风暴**：失败只进入 `FAILED`，同一 epoch 的 notify 不自动重试，
    `pending_reload` 记录目标 epoch，恢复须显式 `retry()`；
  - **V6 下游传播**：旧供应撤销时依赖方进入 `INACTIVE`，新供应登记后重新 `ACTIVE`，
    最终一致；`test_consumer_hot_swap_preserves_downstream_key` 覆盖该链路；
  - **V7 静态插件**：抑制判定仍在替换之前，`suppressed_reloads` 语义与 P2-4 一致，
    原生插件不因本次改动获得 `.so` 卸载能力。

## Task P2-6-1: 下游传播批处理（事务内汇总 notify，事务结束统一刷新）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-5-1（`8d3b173`，热替换事务与效应归属）
- **Description**:
  - 现状缺口（探针实测）：多键提供方卸载时逐键 `notify`，依赖方在中间视图被
    完全卸下；若重供的实现摘要不变（同类实现、撤销后重供同一对象），依赖方会
    **永久停在 `INACTIVE`**（`_epoch` 相等且 `_refresh` 直接返回），出现「依赖
    全可解却未激活」的静默缺口；两位提供方在同一次逻辑变更中先后变更时，下游
    被拆成两轮装/卸。
  - `libs/xuan-plugin/src/xuan_plugin/context.py`：
    - 新增 `Context.batch(flush=True)`（可重入、深度计数）：事务内 `notify` 只
      并入 `_deferred` 去重集合，最外层退出时统一排空；异常退出同样排空（`store`
      已变更时下游不能停留在过期视图）；
    - 新增 `_flush_notifications()`：排空期间回填事务态，依赖方装配产生的新通知
      由下一轮处理；轮次上限 `_MAX_FLUSH_ROUNDS=100` 把病态环转为可诊断错误；
    - `_refresh_dependents()` 以元组快照迭代 Fiber，容忍 apply 期间新增 Fiber；
    - `plugin()` 包在事务内；`dispose()` 使用 `flush=False`，终态拆除丢弃通知。
  - `libs/xuan-plugin/src/xuan_plugin/plugin.py`：
    - `_refresh()`、`retry()` 整体包在 `ctx.batch()` 内，使「卸载旧键 + 供应新键」
    合并为一次下游传播；
    - `_refresh()` 增加收敛分支：`new_epoch == self._epoch` 且状态为 `INACTIVE`
    时重新装配（撤销后重供同一实现的场景）；
    - `_reload()` 成功后以装配后的快照重算 `_epoch`：apply 自身改写注入键（同 id
    环/自提供）时，避免 epoch 永久滞后导致的反复重载与排空不收敛。
  - `libs/xuan-plugin/tests/test_notification_batch.py`（新增 9 个测试）；
    `doc/okf/architecture.md` 同步传播事务与收敛语义。
- **Acceptance Criteria Addressed**: 设计文档 §5.4 第 5–9 步（按依赖集传播、逐字段
  对账、静默断言）、度量 M4（变更半径）
- **Test Requirements**:
  - `rule` TR-P2.6-1: 事务内只登记、最外层退出才刷新，且可重入；
    证据：`test_batch_defers_propagation_until_outermost_exit`、
    `test_batch_is_reentrant_and_flushes_once`
  - `rule` TR-P2.6-2: 一侧多键热替换只产生一轮下游装/卸，批内无中间态；
    证据：`test_multi_key_provider_reload_is_single_cycle`
  - `rule` TR-P2.6-3: 同一批内的多位提供方变更只触发一次依赖方替换（M4）；
    证据：`test_two_providers_in_one_batch_single_cycle`
  - `rule` TR-P2.6-4: 依赖恢复且身份未变时 `INACTIVE` 收敛回 `ACTIVE`；
    证据：`test_resolvable_inactive_fiber_reactivates`
  - `rule` TR-P2.6-5: 排空期间产生的新通知在下一轮处理（级联收敛）；
    证据：`test_cascade_flushes_new_notifications_in_next_round`
  - `rule` TR-P2.6-6: 静态插件抑制计数按逻辑变更计一次；
    证据：`test_static_plugin_suppression_counts_once_per_batch`
  - `rule` TR-P2.6-7: `dispose` 丢弃待处理传播，终态无残留；
    证据：`test_dispose_discards_pending_propagation`
  - `rule` TR-P2.6-8: 事务体抛异常时仍排空，下游不留下过期视图；
    证据：`test_batch_flushes_even_if_body_raises`
- **Completion Evidence**（2026-11-21 实测）:
  - 联跑 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests`
    → **184 passed**（基线 175，新增 9，无删减）；
  - `tools/okf` 独立运行 → **259 passed**；
  - 门禁：`ruff check` 与 `ruff format --check`（58 文件）通过；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src tools/xs/src`
    → `Success: no issues found in 40 source files`；
    `check_pyproject_style.py`（16 个 pyproject）、`check_dependency_direction.py`
    （337 文件）、`check_spec_integrity.py`（任务编号 32）通过；
    `check_declaration_consistency.py` 仍有 2 项既有失败（`libs/mystx` 子模块指针
    漂移与清单快照过期，本次改动前 `git status` 即显示 `M libs/mystx`，与本条无关）；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - **V1 同步语义**：事务外 `provide` 仍逐次同步生效，`test_context.py` /
    `test_context_extra.py` 的既有断言不变；只有显式 `batch()` 或内核内部边界
    （`_refresh` / `retry` / `plugin`）才延迟；
  - **V2 异常传播**：事务体抛异常时仍排空——`store` 已变更，丢弃传播会永久留下
    「依赖全可解却 INACTIVE」；排空自身仅可能抛非法迁移（编程错误），与现状同口径；
  - **V3 非收敛保护**：互提供且摘要每轮变化的病态环会触发轮次上限错误，而不是
    挂死；`_reload` 成功后的 epoch 重算消除了「自提供」这一最常见的振荡源；
  - **V4 终态拆除**：`dispose` 用 `flush=False` 丢弃通知，避免刷新正在销毁的
    Fiber；`dispose` 的错误聚合语义不变；
  - **V5 计数口径**：`reloads` 仍是「成功热替换次数」，批处理后一次逻辑变更只计
    一次；`suppressed_reloads` 同理从「按键」变为「按变更」；
  - **V6 遗留风险（不在本条范围）**：`_digest` 对无源码对象回退到 `repr`，其中
    含内存地址，存在地址复用导致「新实现摘要与旧实现相等」的概率；本条通过
    「依赖可解即必须 ACTIVE」的收敛分支使其不再产生静默缺口，但摘要身份本身
    仍不满足 M2 的跨进程可复现要求，留待身份改造任务处理（已由 Task P2-7-1 处置）。

## Task P2-7-1: 身份摘要去地址化（无源码对象跨进程可复现）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-6-1（`beb5fbb`，批处理传播；本条处置其 V6 遗留风险）
- **Description**:
  - 现状缺口：`_digest` 对源码不可得的对象回退 `sha256(repr(value))`，而默认
    `repr` 内嵌堆地址（`<object object at 0x…>`），由此产生两个后果——同一装配
    在两次独立进程中算出不同 epoch（不满足设计 §8 M2）；旧对象回收后新对象
    复用同一地址时，新实现被误判成旧实现而跳过重载（地址复用伪相等）。
  - `libs/xuan-plugin/src/xuan_plugin/plugin.py`：
    - 新增 `_stable_repr()`：把 `repr` 中形如 `0x` 加至少 8 位十六进制的堆地址
      折叠为固定占位符 `0xADDR`；位数下限取 8，避免把 24 位色值等短字面量
      误判为地址，同时覆盖 32/64 位指针；
    - `_digest` 的无源码分支改用 `_stable_repr`；`_config_digest` 的
      `json.dumps(default=…)` 由 `repr` 改为 `_stable_repr`，配置内嵌的不可
      序列化对象同样不再把地址带入摘要；
    - 身份契约明确为「实现契约而非某次分配」：同类型、同内容的实例摘要相同，
      与有源码实现的实例判等方式一致。
  - 测试口径同步：`test_digest.py` 以新契约替换旧的「同类型实例必须可区分」
    断言并新增 6 个用例；`test_hot_reload.py`、`test_lifecycle.py`、
    `tools/okf/tests/test_plugin_extra.py` 中以 `object()` 充当「另一版实现」
    的写法改为源码可见的 `ImplV1`/`ImplV2` 类，保持「实现切换触发重载」的
    原始验证意图；`doc/okf/architecture.md` 同步摘要构成与 M2 口径。
- **Acceptance Criteria Addressed**: 设计 §7 P0-2「摘要函数同输入同输出、
  跨进程一致」；§8 M2「两次独立进程装配摘要相等」
- **Test Requirements**:
  - `rule` TR-P2.7-1: 无源码对象摘要跨进程相等；证据：
    `test_digest_source_less_reproducible_across_processes`
  - `rule` TR-P2.7-2: 摘要不含堆地址且折叠可观测；证据：
    `test_stable_repr_collapses_address`、`test_digest_contains_no_memory_address`
  - `rule` TR-P2.7-3: 短十六进制字面量不被误折叠；证据：
    `test_stable_repr_keeps_short_hex_literal`
  - `rule` TR-P2.7-4: 无源码但类型不同仍可区分；证据：
    `test_digest_source_less_types_are_distinguished`
  - `rule` TR-P2.7-5: 配置内嵌对象不泄漏地址；证据：
    `test_config_digest_normalizes_addresses`
  - `rule` TR-P2.7-6: 实现切换仍触发热替换；证据：
    `test_hot_reload_undoes_old_and_applies_new`、
    `test_static_plugin_suppresses_hot_reload`、`test_epoch_change_unload_then_reload`
- **Completion Evidence**（2026-11-21 实测）:
  - 联跑 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests`
    → **189 passed**（基线 184，净增 5：新增 6 项、按新契约删除 1 项）；
  - `tools/okf` 独立运行 → **259 passed**（与基线一致）；
  - 门禁：`ruff check` 与 `ruff format --check`（58 文件）通过；
    `mypy libs/xuan-plugin/src libs/xuan-ffi/src tools/xs/src`
    → `Success: no issues found in 40 source files`；
    `check_pyproject_style.py`（16 个 pyproject）、`check_dependency_direction.py`
    （337 文件）、`check_spec_integrity.py`（任务编号 32）通过；
    `check_declaration_consistency.py` 仍为既有 2 项失败（`libs/mystx` 指针漂移
    与清单快照过期，改动前即存在）；
  - 实机：`python -m xs.cli plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；
    对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - **V1 身份语义变化**：同类型、无源码、无状态的两个实例（如两个 `object()`）
    不再被区分。这是刻意取舍——地址不是语义身份；且既有实现对「有源码类」
    的实例本就按类身份判等，本条只是把兜底路径拉齐；
  - **V2 折叠粒度**：仅折叠「`0x` + ≥8 位十六进制」，32/64 位指针全覆盖，
    24 位色值不受影响；若出现十进制形式的自引用 id，需另行扩展（当前
    可达路径中 `Mock` 一类对象因类源码可得而不走兜底分支）；
  - **V3 自定义 repr 边界**：若作者在 `__repr__` 中打印非 `0x` 形式的地址，
    摘要仍不可跨进程复现；此类对象应提供源码可见的类型，或在
    `InjectSpec.config` 中显式声明版本；
  - **V4 性能**：无源码分支本就依赖 `repr`，折叠只增加一次线性正则扫描，
    相对 `inspect.getsource` 与 sha256 可忽略；
  - **V5 截断长度**：`_DIGEST_LEN=16`（64 bit）未变，碰撞概率与既有口径一致。

