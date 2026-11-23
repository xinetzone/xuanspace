> 分卷来源：`tasks.md` 原文 L466–L659（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：03-p0-3-disposal.md · 下一卷：05-p2-a-build-ffi.md
> 主题：增量三 P1：清单、求解器、CLI 插件化
# 增量三：P1-1（清单解析与装配图）— 执行记录

> 2026-11-21 追加。自设计文档 §7 P1-1 派生，原文判据为「`[tool.xuan.plugin]` schema +
> `entry_points("xuan.plugins")` 合并 → 产出 `xs plugin list/graph`」，验收要求为
> 「对内置命令生成等价图」。与 P0 系列不同，本增量新增公开命令与运行时依赖，故同样独立成节。

## Task P1-1-1: K2 清单解析器与 `xs plugin list/graph`

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P3-1（`2dadf4f`）
- **Description**:
  - 新库新增 `manifest.py`（282 行、纯标准库）：`ProvideDecl` / `InjectDecl` / `PluginManifest`
    三个不可变数据类；`parse_manifest` 做 K2 字段校验（必填 `id`/`version`、未知字段拦截、
    `mode` 与 `lifecycle` 枚举、能力键语法 `xuan:<domain>/<owner>#<api>`、provide/inject 键去重、
    TOML 布尔不得冒充 `priority` 整数）；`manifest_from_pyproject` 定位 `[tool.xuan.plugin]`；
  - `merge_manifests`：同 id 且契约相同则去重（保留先到者），契约不同直接报错，杜绝双真相；
    `source` 为诊断标记，不参与契约比较（`contract()` 显式剔除）；
  - `build_graph`：展开节点（跳过 `disabled`）、依赖边、未解析的**必需**依赖与同键冲突；
    多提供者按 `(-priority, id)` 定序，同键多提供者仅在全部为 `single` 时计冲突（`multi` 免冲突），
    与 ADR-05 一致；另提供 `PluginGraph.to_dot()`；
  - `tools/xs` 新增 `plugin_cmd.py`（183 行）：`xs plugin list` 与 `xs plugin graph`，支持
    `--json` 与 `--dot`；数据来源＝工作区各 `pyproject.toml` 静态声明 ∪ `xuan.plugins` 入口点，
    入口点加载失败或缺少清单对象时降级（告警跳过 / 按元数据合成最小清单）；
  - `tools/xs/pyproject.toml` 新增 `[tool.xuan.plugin] id = "xs.commands"`，逐条声明 16 个能力键，
    并登记运行时依赖 `xuan-plugin>=0.1.0`；`cli.py` 注册 `plugin` 子命令组；
  - CI 四处 `Install xs-cli` 之前各插入一步内核安装（本地包不在 PyPI，否则依赖解析失败）；
  - 治理同步：`README.md` 子命令数 15 → 16，`doc/intro.md` 与 `doc/cli/index.md` 补 `xs plugin`，
    `declaration-snapshot.json` 重新生成。
- **Acceptance Criteria Addressed**: 对内置命令生成等价图；清单违约可诊断；同 id 双真相被拒
- **Test Requirements**:
  - `rule` TR-P1.1: 最小清单解析出既定默认值；证据：`test_parse_minimal_defaults`
  - `rule` TR-P1.2: 违约清单逐类被拒（缺必填、键语法、`mode`、`lifecycle`、未知字段、
    重复 provide 键、`priority=true`）；证据：8 个 `ManifestError` 用例
  - `rule` TR-P1.3: 同 id 同契约去重、不同契约报错；证据：`test_merge_manifests_*` 两项
  - `rule` TR-P1.4: 依赖边正确，必需依赖未满足时记入 `unresolved`，`optional` 不计入
  - `rule` TR-P1.5: 多提供者按 priority 定序；同键 `single` 记冲突、`multi` 免冲突
  - `rule` TR-P1.6: `disabled` 插件不进入节点集与边集
  - `rule` TR-P1.7: `to_dot()` 含 `digraph` 头与带键标签的边
  - `rule` TR-P1.8: `tools/xs` 声明的能力键集合与 `cli.py` 注册表**逐项相等**；证据：
    `test_declared_commands_match_cli_registry`（差异时报出两侧差集）
  - `rule` TR-P1.9: 入口点清单被合并；加载失败的入口点被跳过且不影响静态声明
  - `rule` TR-P1.10: 80 列终端下清单表格不出现省略号，ID、`reloadable`、来源完整可读
  - `rule` TR-P1.11: `tools/okf` 与根 `tests/` 通过数不回归
  - `rule` TR-P1.12: 声明一致性、依赖方向、pyproject 风格三项治理脚本全通过
- **Completion Evidence**（2026-11-21 实测）:
  - 提交：`b06aa1e`（完整 `b06aa1e0d95ad758727e3e8c055952aa825a939f`），
    信息 `feat(plugin): 新增 K2 清单解析器与 xs plugin list/graph`，12 文件 +985 / −4；
    后续 `e736bae`（完整 `e736bae6603b5f98bc0310e28296bbf4cc19e968`），
    信息 `fix(xs): 插件清单表格窄终端截断与折行`，2 文件 +20 / −10。
  - 新增文件：`libs/xuan-plugin/src/xuan_plugin/manifest.py`（282 行）、
    `libs/xuan-plugin/tests/test_manifest.py`（154 行、21 项）、
    `tools/xs/src/xs/commands/plugin_cmd.py`（183 行）、
    `tests/test_plugin_cmd.py`（89 行、7 项）。
  - TR-P1.1～P1.7 通过：`cd libs/xuan-plugin && python -m pytest -q` → **39 passed**（18 + 21）。
  - TR-P1.8～P1.10 通过：根 `tests/` → **42 passed**（35 + 7）。
  - TR-P1.11 通过：`cd tools/okf && python -m pytest -q` → **259 passed**（无回归）。
  - TR-P1.12 通过：`check_declaration_consistency.py`、`check_dependency_direction.py`、
    `check_pyproject_style.py` 均无输出（通过）。
  - 门禁：`ruff check` All checks passed；`ruff format --check` 30 files already formatted；
    `mypy tools/xs libs/xuan-plugin` → Success: no issues found in 30 source files。
  - 实机输出：`xs plugin graph` → 「插件 1 个，能力键 16 个，依赖边 0 条，未解析 0，冲突 0」；
    `xs plugin list` 表格宽 67 列，边框闭合、无省略号。
- **Notes**:
  - **计数偏差**：设计文档写「14 个内置命令」，Task 1 实测为 15 个；加上本阶段新增的 `plugin`
    子命令组，清单共 **16** 条，故 TR-P1.8 以「两侧集合相等」而非固定数字为断言核心。
  - **环境订正**：本机 `xs` 的 editable 安装原指向另一份拷贝 `D:\AI\projects\xuanspace`，
    已按用户指示在仓库内重装（`pip install -e tools/xs --no-deps --no-build-isolation`），
    `import xs` 现解析至本仓源码。
  - **已知上限**：入口点若无 `MANIFEST`/`manifest` 属性，只能按其元数据合成「零能力」节点，
    真实能力仍以清单对象为准；求解与回滚（拓扑定序、隔离域、逐字段对账）留待 P1-2。

## Task P1-2-1: 装配求解器、事务回滚与 `xs plugin apply --dry-run`

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P1-1（`c193ab1`）
- **Description**:
  - 新库新增 `solver.py`（325 行、纯标准库）：`SolveError(kind ∈ cycle/missing/conflict)`、
    `Solution(order, domains, inactive, provides)`、`solve(manifests, *, allow_inactive=True)`；
    就绪集按 `(-priority, id)` 出队的 Kahn 定序，输入置换不改变结果；缺失必需依赖按设计 §5.4
    第 3 步标 INACTIVE 并沿依赖闭包传播（`allow_inactive=False` 转为 `MISSING` 异常）；
    跨 id 环报 `a -> b -> a` 明细，同键 `single` 多提供者或 `multi` 同优先级判冲突；隔离域
    按共享 `isolate` 标签的连通簇划分，无标签者合并为默认域；
  - `reconcile(solution, registered)`：声明 provide 与实际注册键逐项对账，缺/多分别列出；
  - 新库新增 `assembly.py`（115 行）：按解序装配，节点非 `ACTIVE` 即判失败（防止「供应者未
    真正登记键、消费者停在 PENDING」被计成假成功）；失败时逆序回滚并聚合撤销异常，静默断言
    回滚后效应数 = 0 且已装配提供键全部从 store 清除；同一异常被 `dispose` 重复上报时按对象去重；
  - `Context` 新增 `registered_keys()`；`Context.plugin()` 改为返回 `Fiber`，供装配器查证状态；
  - `xs plugin apply --dry-run`（支持 `--json`）：输出定序、隔离域、失活原因与选定提供者；
    未加 `--dry-run` 时显式拒绝并指向 P1-3，避免把空操作伪装成装配成功；
  - 文档同步：`doc/cli/index.md`、`doc/intro.md`、`libs/xuan-plugin/README.md`。
- **Acceptance Criteria Addressed**: 拓扑 + 隔离 + 对账 + 回滚 + 静默断言；对刻意构造的冲突/环/缺依赖样例给出确定性输出
- **Test Requirements**:
  - `rule` TR-P1.2-1: 供应者先于消费者，就绪集按 `(-priority, id)` 定序，输入置换不改变结果；
    证据：`test_provider_precedes_consumer`、`test_ready_set_sorted_by_priority_then_id`、
    `test_order_is_independent_of_input_order`
  - `rule` TR-P1.2-2: 自提供键按 peer 语义不构成环；跨 id 环报 `a -> b -> a` 且两次求解文本一致；
    证据：`test_self_dependency_is_peer_semantics`、`test_cross_cycle_is_rejected_with_deterministic_path`
  - `rule` TR-P1.2-3: 同键 `single` 多提供者、`multi` 同优先级均判冲突，`multi` 异优先级按
    priority 取胜出者；证据：`test_single_key_conflict_is_rejected`、`test_multi_providers_require_distinct_priority`
  - `rule` TR-P1.2-4: 缺必需依赖标 INACTIVE 并沿闭包传播，`optional` 不阻塞，`disabled` 不入解；
    证据：`test_missing_required_dependency_marks_inactive`、`test_inactive_propagates_to_dependents`、
    `test_optional_dependency_never_blocks`、`test_strict_mode_rejects_missing_dependencies`、
    `test_disabled_manifest_is_excluded`
  - `rule` TR-P1.2-5: 隔离域按共享标签聚类、标签链传递合并、无标签归默认域；
    证据：`test_domains_cluster_by_shared_isolate_label`、`test_domains_merge_transitively_across_labels`
  - `rule` TR-P1.2-6: 对账缺/多分别列出，两侧一致时 `ok`；证据：`test_reconcile_reports_missing_and_unexpected`
  - `rule` TR-P1.2-7: 装配失败逆序回滚，残留效应数与注册键均为 0，撤销异常被聚合；
    证据：`test_failure_rolls_back_in_reverse_and_leaves_no_residue`、
    `test_dispose_errors_are_aggregated_but_residue_still_zero`
  - `rule` TR-P1.2-8: 供应者未真正登记键时消费者不被计为成功；
    证据：`test_unregistered_provide_leaves_consumer_inactive_and_fails`
  - `rule` TR-P1.2-9: 两次 `apply --dry-run --json` 输出逐字节一致；冲突/环/缺依赖三类反例给出
    确定性输出；证据：`test_apply_dry_run_json_is_byte_identical`、`test_apply_conflict_is_deterministic`、
    `test_apply_cycle_reports_path`、`test_apply_missing_dependency_is_reported_as_inactive`
  - `rule` TR-P1.2-10: 未加 `--dry-run` 显式拒绝；证据：`test_apply_refuses_when_not_dry_run`
  - `rule` TR-P1.2-11: `okf` 与既有根测试不回归；三套治理脚本全通过
- **Completion Evidence**（2026-11-21 实测）:
  - 提交：`a8be022`（完整 `a8be022b1bb4cacc1f7a39ad8412aa5b69017e4c`），
    信息 `feat(plugin): 求解器与装配回滚及 apply --dry-run`，11 文件 +985 / −12。
  - 新增文件：`libs/xuan-plugin/src/xuan_plugin/solver.py`（325 行）、
    `libs/xuan-plugin/src/xuan_plugin/assembly.py`（115 行）、
    `libs/xuan-plugin/tests/test_solver.py`（199 行、16 项）、
    `libs/xuan-plugin/tests/test_assembly.py`（126 行、6 项）。
  - 修改文件：`tools/xs/src/xs/commands/plugin_cmd.py` 183 → 293 行（新增 `apply` 命令）；
    `tests/test_plugin_cmd.py` 89 → 208 行（新增 6 项）；`context.py`（`registered_keys()`、
    `plugin()` 返回 `Fiber`）；`__init__.py` 导出新符号。
  - `cd libs/xuan-plugin && python -m pytest -q` → **61 passed**（39 + 22）。
  - 根 `tests/` → **48 passed**（42 + 6）。
  - `cd tools/okf && python -m pytest -q` → **259 passed**（无回归）。
  - 门禁：`ruff check` All checks passed；`ruff format --check` 451 files already formatted；
    `mypy tools/xs libs/xuan-plugin` → Success: no issues found in 34 source files。
  - 治理脚本：声明一致性、依赖方向、pyproject 风格三项均通过。
  - 实机输出：`xs plugin apply --dry-run` → 「装配顺序 1 个节点 / 1. xs.commands priority=100
    provide=16 / 隔离域（默认域）：xs.commands / 可解性：全部必需依赖可满足」；
    两次 `--json` 输出 `-ceq` 比较为 `True`。
- **Notes**:
  - 设计文档 §5.4 第 3 步要求缺失必需依赖「标 INACTIVE 而不失败」，故 `solve` 默认不抛错；
    `allow_inactive=False` 供需要「全量可激活」的调用方（如发布前检查）显式收紧。
  - 真实装配（`apply` 不加 `--dry-run`）依赖 P1-3 的运行时插件实现，当前显式拒绝而非空转。
  - `Fiber` 卸载成功后仍保留上一次的 `_error`，导致 `Context.dispose` 重复上报同一异常；
    本轮在装配器按异常对象去重，未改动 `Fiber` 语义（留待后续按需修正，已由 Task P2-8-1 处置）。

## Task P1-3-1: 命令注册切换为插件装配并接通真实 `apply`

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P1-2（`a8be022`）
- **Description**:
  - 新增 `tools/xs/src/xs/plugins/`：`registry.py`（`CommandRegistry`，`register` 返回撤销函数，
    撤销后回到装配前空态）、`commands.py`（`COMMAND_SPECS` 16 条 + `MANIFEST` + `apply`，
    每条命令与能力键均经 `ctx.effect` 登记，可整批回收）、`__init__.py`；
  - `tools/xs/src/xs/cli.py`：不再逐个 import 子命令，改为构造 `Context` →
    `ctx.plugin(COMMANDS_PLUGIN, {"registry": ...})` → 从 `CommandRegistry.entries()` 构建
    Typer 应用；新增/移除内置命令只需改声明表；
  - `tools/xs/src/xs/commands/affected_cmd.py`：把 `affected` 从 `cli.py` 抽出为独立命令模块；
  - `tools/xs/pyproject.toml`：新增 `[project.entry-points."xuan.plugins"]`，静态清单与入口点
    两条通道共用同一份声明；
  - `plugin_cmd.py`：取消「未加 `--dry-run` 即拒绝」，默认执行真实装配（`assemble` +
    `CommandRegistry`），节点非 `ACTIVE` 即判失败并逆序回滚；输出已装配节点、命令清单与
    「声明 provide / 实际注册键」对账（missing/unexpected/matched）；缺运行时实现显式列名退出；
  - `scripts/check_declaration_consistency.py`：CLI 命令的事实来源由 `cli.py` 正则改为
    `plugins/commands.py` 的 `CommandSpec(...)` 登记表；
  - 文档同步：`doc/cli/index.md`、`doc/intro.md` 改述真实装配、ACTIVE 判据与对账语义。
- **Acceptance Criteria Addressed**: 装配后 `xs --help` 与改造前逐字一致；`apply` 真实装配可逆、
  对账闭合；命令声明与实现单源，无双真相漂移
- **Test Requirements**:
  - `rule` TR-P1.3-1: 16 个命令由插件登记表声明，与 `pyproject.toml` 静态清单契约一致；
    证据：`test_declared_commands_match_cli_registry`、`test_builtin_manifest_matches_static_declaration`
  - `rule` TR-P1.3-2: 命令登记逐项可撤销，撤销后无残留；证据：`test_command_plugin_registration_is_reversible`
  - `rule` TR-P1.3-3: 改为装配后 `xs --help` 与改造前快照逐字相同；证据：`test_help_output_matches_snapshot`
  - `rule` TR-P1.3-4: `apply` 不加 `--dry-run` 时真实装配，16 条命令登记且对账一致；
    证据：`test_apply_activates_builtin_plugin`
  - `rule` TR-P1.3-5: 缺运行时实现显式报错退出，不把空操作伪装成成功；
    证据：`test_apply_reports_missing_implementation`
  - `rule` TR-P1.3-6: `--dry-run` 输出与 P1-2 逐字节一致（改造未影响求解面）；
    证据：`test_apply_dry_run_json_is_byte_identical`
  - `rule` TR-P1.3-7: 声明一致性门禁改读命令登记表后仍全通过
- **Completion Evidence**（2026-11-21 实测）:
  - 根 `tests/` → **52 passed**；`cd libs/xuan-plugin && python -m pytest -q` → **61 passed**；
  - 门禁：`ruff check` All checks passed；`ruff format --check` 455 files already formatted；
    `mypy tools/xs libs/xuan-plugin` → Success: no issues found in 38 source files；
  - `python scripts/check_declaration_consistency.py` → 六类声明 + G 类子模块指针对账全部通过；
  - 实机：`xs plugin apply` → 「装配完成：1 个节点 ACTIVE，16 条命令已登记 /
    对账：一致 16 键，缺 0，多 0」；`xs plugin apply --dry-run` → 定序、隔离域、可解性输出不变；
  - `tests/snapshots/xs_help.txt` 由与测试同源的 `CliRunner` 生成（rich 15.0.0，圆角面板），
    与 `--help` 输出逐字相同。
- **Notes**:
  - 快照为逐字比对，rich 升级会改变面板字符与宽度；升级后须按同一 harness 重建快照，
    不可手工编辑。
  - `Fiber` 卸载后仍保留 `_error` 的问题未在 P1-3 处理，仍由装配器按异常对象去重兜底
    （承接 P1-2 的 A4 待办，已由 Task P2-8-1 处置）。

