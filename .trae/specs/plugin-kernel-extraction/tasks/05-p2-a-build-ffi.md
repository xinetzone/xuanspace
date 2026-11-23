> 分卷来源：`tasks.md` 原文 L660–L840（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：04-p1-kernel-cli.md · 下一卷：06-p2-b-lifecycle.md
> 主题：P2-A：构建后端、compose、FFI 试点
## Task P2-1-1: 构建后端多提供者（`xuan:build/backend#dispatch`）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P1-3（`29fd63e`）
- **Description**:
  - 新增独立能力库 `libs/xuan-build`（包名 `xuan-build`，模块 `xuan_build`）：
    `backends.py` 定义 `Reporter` / `BuildBackend` 协议与 Python/Native/Static/Other 四种实现，
    `dispatch.py` 提供 `BuildDispatcher`（未知类型落 `other`）与 `DEFAULT_DISPATCHER`，
    `__init__.py` 导出 `DISPATCH_KEY = "xuan:build/backend#dispatch"`、`MANIFEST`、`apply`、`PLUGIN`；
  - `tools/xs/src/xs/commands/build_cmd.py`：删除 `_build_python_project` /
    `_build_native_project` 与 `subprocess` 直调，`_build_single_project` 改为
    `backend = dispatch(project_type)` → `skip_message` 命中即打印跳过 → 否则在 Progress 内
    `backend.build(project_path, console)`；
  - `tools/xs/pyproject.toml`：新增 `xuan-build>=0.1.0` 依赖、`[[tool.xuan.plugin.inject]]`
    该键，并把入口点名改为带引号的 `"xs.commands"`；`xs/plugins/commands.py` 的 `MANIFEST`
    同步声明 inject；
  - `tools/xs/src/xs/commands/plugin_cmd.py`：内置插件集加入 `xuan.build`，装配顺序由
    priority 决定（`xuan.build` 50 → `xs.commands` 100）；
  - `pyproject.toml`：mypy `exclude` 中 `build/`、`dist/` 改为路径段锚定 `(^|/)build/`、
    `(^|/)dist/`——原写法是 `re.search` 语义，会误命中 `libs/xuan-build/src/`，使新库被整体跳过；
  - 治理同步：CI 三个 job 均安装 `libs/xuan-build`，quality 新增其测试步骤，mypy 命令加入该库；
    `scripts/check_declaration_consistency.py` 的 `LIB_PROJECT_META` 登记新库并刷新快照；
    根 `README.md` 项目索引补 `xuan-build` 行。
- **Acceptance Criteria Addressed**: 设计文档 §7 P2-1——构建实现由单一能力键分派，
  `xs build` 对 python/native/static 三类项目的行为与文案不变
- **Test Requirements**:
  - `rule` TR-P2.1-1: 四种项目类型分派到对应后端，未知类型落 `other`；
    证据：`test_dispatch_covers_all_project_types`
  - `rule` TR-P2.1-2: 跳过文案逐字保留（static / other）；
    证据：`test_skip_messages_are_stable`
  - `rule` TR-P2.1-3: Python 后端成功/失败路径与输出保留；
    证据：`test_python_backend_success`、`test_python_backend_failure_reports_both_streams`
  - `rule` TR-P2.1-4: Native 后端 `pip install -e` 失败后回退 `python -m build --wheel`；
    证据：`test_native_backend_falls_back_to_wheel_build`
  - `rule` TR-P2.1-5: 代码清单与 `pyproject.toml` 静态清单同契约，插件可逆；
    证据：`test_manifest_matches_pyproject`、`test_plugin_registration_is_reversible`
  - `rule` TR-P2.1-6: `xs build` 经能力键委派，跳过与失败文案不变；
    证据：`tests/test_build_dispatch.py` 的三条用例
- **Completion Evidence**（2026-11-21 实测）:
  - 根 `tests` + `libs/xuan-build/tests` + `libs/xuan-plugin/tests` → **123 passed**；
  - 门禁：`ruff check .` All checks passed；`ruff format --check .` 462 files already formatted；
    `mypy tools/xs libs/xuan-core libs/xuan-ext-demo libs/xuan-build libs/xuan-plugin` →
    Success: no issues found in 45 source files；
  - `check_pyproject_style.py`（15 个 pyproject）、`check_dependency_direction.py`（318 文件）、
    `check_spec_integrity.py`（3 个台账、26 个任务编号）、`check_declaration_consistency.py`
    全部通过；`xs toolchain check` 无新增异常；
  - 实机：`xs plugin apply --json` → `applied = ["xuan.build", "xs.commands"]`，
    `matched` 17 键（`xuan:build/backend#dispatch` + 16 条命令），missing/unexpected 均为空。
- **Notes**:
  - 设计文档中的阶段号 P2-1 与本台账既有的 `Task P2-1`（epoch 摘要身份）重号，故本条以
    任务编号 `P2-1-1` 登记，避免台账门禁判定重复。
  - 未改动内核的多值查找语义：multi 语义仍停留在 solver/manifest 层，本轮只新增单键单值的
    分派门面。
  - `xs build --type` 的取值仍限定 `python/native/all`，未新增 static 批量入口——保持行为不变。

## Task P2-2-1: `xuan-compose` 指令登记化（处理器表 → 登记 + Disposable）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-1-1（`8e6b427`）
- **Description**:
  - 新增 `libs/xuan-compose/src/xuan_compose/commands/registry.py`：
    `CommandSpec`（name / handler / parsers / help / description）与
    `CommandRegistry`。`register()` 返回 `xuan_plugin.Disposable`，撤销即摘除该条；
    `register_all()` 用 `DisposableList` 栈式逆序回滚；`bind()` 把 handler 灌入
    `compose.commands` 并在撤销时逐项恢复绑定前取值（原本不存在的键被删除）；
    重复命令名视为契约冲突，显式 `ValueError` 而非静默覆盖；
  - 新增 `libs/xuan-compose/src/xuan_compose/cli/command_specs.py`：
    24 条指令的**单一登记点** `COMMAND_SPECS`，同时携带 handler、parser 挂载序、
    help/description；`COMMAND_HANDLERS` / `COMMAND_PARSERS` / `COMMAND_HELP`
    降级为只读派生视图（`MappingProxyType`），仅供断言与快照对拍；
  - `commands/__init__.py`：删除 `COMMAND_HANDLERS` 静态表，仅保留
    `install_handlers(compose, registry=None) -> Disposable` 绑定入口；
  - `cli/parser.py`：删除 `COMMAND_PARSERS` / `COMMAND_HELP` 两张静态表，
    `build_parser()` / `parse_args()` 增加 `registry` 参数，subparser 由登记序派生；
    systemd 的 help/description 仍由 handler docstring 派生（`SYSTEMD_HELP`/`SYSTEMD_DESC`）；
  - `cli/main.py`：装配期只建一次注册表，parser 与引擎查表共用同一实例；
  - `libs/xuan-compose/pyproject.toml`：新增 `xuan-plugin>=0.1.0` 运行时依赖；
  - `tests/`：三个文件的表引用改指 `command_specs`，并新增 4 条撤销语义用例；
  - `README.md`：架构图的 CLI/commands 节点同步为新结构。
- **Acceptance Criteria Addressed**: 设计文档 §7 P2-2——处理器表 → 登记 + Disposable；
  现有 compose 测试全绿；新增指令的声明改动集中在单一文件（`command_specs.py`）
- **Test Requirements**:
  - `rule` TR-P2.2-1: 24 条指令的 handler/parser/help 三视图键集合与上游 `@cmd_run`
    一致，登记序不变；证据：`test_registry_matches_upstream_cmd_run_set`、
    `test_command_set_and_order_equal_to_snapshot`
  - `rule` TR-P2.2-2: 登记返回的 Disposable 只摘除本条登记；证据：
    `test_register_returns_undo_that_removes_only_that_entry`
  - `rule` TR-P2.2-3: 重复登记报错、批量登记失败整体回滚（无半装配状态）；证据：
    `test_duplicate_registration_is_contract_conflict`、`test_register_all_rolls_back_on_conflict`
  - `rule` TR-P2.2-4: `bind` 的逆函数恢复 `compose.commands` 绑定前状态；证据：
    `test_bind_disposable_restores_previous_lookup`
  - `rule` TR-P2.2-5: 顶层 help 文本逐字不变（4688 字符，
    sha256 `d27cf8b0…b3742729`，改造前后同值）
- **Completion Evidence**（2026-11-21 实测）:
  - `libs/xuan-compose` 全量：**1015 passed, 79 subtests passed**（改造前基线
    1011 passed / 79 subtests，新增 4 条撤销语义用例）；
  - 门禁：`ruff check` / `ruff format --check`（129 文件）通过；
    `mypy libs/xuan-compose/src` → Success: no issues found in 41 source files；
    `check_pyproject_style.py`（15 个 pyproject）、`check_dependency_direction.py`
    （320 文件，libs 未依赖 apps/scripts/attic/vendor）通过；
  - 仓库级：`pytest -q tests libs/xuan-build/tests libs/xuan-plugin/tests` → 123 passed；
    `check_spec_integrity.py`（3 个台账、28 个任务编号）、
    `check_declaration_consistency.py`、`xs toolchain check` 全部通过。
- **Notes**:
  - 设计文档 §7 的 P2-2 与本台账既有的 epoch 任务 `Task P2-2` 同号风险：本轮以
    `P2-2-1` 登记；本台账中 epoch 摘要身份实际登记为 `Task P2-1`，未出现 `P2-2`，
    但沿用「设计阶段号 + 序号」的既有约定以避免后续重号。
  - 「新增指令只改一个文件」的度量口径：清单是唯一声明点，新增一条指令只需在
    `command_specs.py` 追加一条 `CommandSpec`；若 handler 尚不存在，需另加其实现
    模块（2 个文件），与 P1-3 的「1–2 个文件」口径一致。
  - 未把 `xuan-compose` 纳入 CI 的 mypy 命令范围（CI 现状不含该库），本轮仅在本地
    以 `mypy libs/xuan-compose/src` 验证零告警；扩大 CI 范围属独立治理项。

## Task P2-3-1: FFI 能力键与子进程隔离模式（试点 C）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-1-1（`8e6b427`）、P2-2-1（`46f0e47`）
- **Description**:
  - 新增 `libs/xuan-ffi`（发行名 `xuan-ffi`，纯 Python；运行时仅依赖 `xuan-plugin`）：
    - `keys.py`：能力键语法 `xuan:ffi/<name>#api`、`ffi_key()` / `ffi_name()`；名称正则
      与 `xuan_plugin.manifest` 的 K2 键校验同一口径；
    - `api.py`：`FFIAPI` 协议（`describe` / `invoke` / `close`）、`FFIInfo` 快照、
      `FFIError` / `FFIUnavailableError`；同一条 `invoke` 面同时适配进程内与子进程；
    - `inprocess.py`：`InProcessFFI` 惰性 import、点分属性解析；`describe()` 以
      `available=False + error` 表达不可用而不抛错；`close()` 只丢弃引用；
    - `proxy.py`：`SubprocessFFI` 以 `python -m xuan_ffi.worker` + JSON Lines over stdio
      调用；请求/响应按 id 匹配；`start()` / `close()` 可反复循环，`close()` 幂等并按
      wait → terminate → kill 收敛；支持 `extra_paths` 与 `PYTHONIOENCODING/PYTHONUTF8`；
    - `worker.py`：子进程侧请求循环，复用同一 `resolve()` 语义；加载失败与不可序列化
      结果均转成结构化错误回传；
    - `providers.py`：`BUILTIN_FFI = {caffe: caffe_ffi, demo: demo_ffi, npu: npu_ffi}`
      与 `create_ffi(name, module=..., mode=...)`；
    - `__init__.py`：`MANIFEST`（`lifecycle="static"`、priority 40、三条 provide）、
      `apply()`（按 `mode` 登记三条能力键，撤销时先关闭提供者再移除键）、`PLUGIN`；
  - `libs/xuan-ffi/pyproject.toml` 同步 `[tool.xuan.plugin]` 与
    `[project.entry-points."xuan.plugins"]`；
  - `tools/xs`：`plugin_cmd` 的内置插件表加入 `xuan.ffi`，`pyproject.toml` 增加
    `xuan-ffi>=0.1.0` 运行时依赖；
  - `.github/workflows/ci.yml`：三处安装块补 `xuan-ffi`，质量门新增
    `pytest libs/xuan-ffi/tests` 与 mypy 路径；
  - `scripts/check_declaration_consistency.py` 与根 `README.md` 项目索引补登记，
    `libs/README.md` 生成块与 `scripts/declaration-snapshot.json` 由 `--generate` 刷新。
- **Acceptance Criteria Addressed**: 设计文档 §7 P2 试点 C——切换后端不改调用方代码；
  子进程模式可重复装载；能力句柄可撤销；不承诺 `.so` 卸载（ADR-03）
- **Test Requirements**:
  - `rule` TR-P2.3-1: `xuan:ffi/<name>#api` 键通过 `xuan_plugin.manifest` 校验，非法
    名称被拒绝；证据：`test_keys.py`（6 条参数化拒绝用例 + K2 兼容用例）
  - `rule` TR-P2.3-2: 同一消费者函数在进程内与子进程两种传输下结果一致；证据：
    `test_same_consumer_works_across_transports`
  - `rule` TR-P2.3-3: 子进程模式连续三轮 `start → invoke → close` 得到三个不同 pid，
    `close()` 后 `pid is None`；证据：`test_repeated_load_uses_fresh_processes`
  - `rule` TR-P2.3-4: 装配—调用—撤销两轮后能力键清空、effect 清空、worker 结束；证据：
    `test_apply_registers_and_disposes`、`test_subprocess_mode_is_repeatable`
  - `rule` TR-P2.3-5: 静态清单与 `MANIFEST` 契约逐字段一致，`xs plugin apply` 对账
    20 键；证据：`test_manifest_matches_pyproject_declaration` 与 apply 输出
- **Completion Evidence**（2026-11-21 实测）:
  - `libs/xuan-ffi`：**34 passed**；
  - 联跑 `pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests`
    → **157 passed**；
  - 门禁：`ruff check` 与 `ruff format --check`（69 文件）通过；
    `mypy libs/xuan-plugin/src libs/xuan-build/src libs/xuan-ffi/src tools/xs/src`
    → `Success: no issues found in 43 source files`；
    `check_pyproject_style.py`（16 个 pyproject）、`check_dependency_direction.py`
    （334 文件）、`check_spec_integrity.py`、`check_declaration_consistency.py` 通过；
    `xs toolchain check` 无新增异常（仅提示可选 pdm 未安装）；
  - 实机：`xs plugin list` → 3 个插件 / 20 个能力键（`xuan.ffi` 3 键、static、prio 40）；
    `xs plugin apply` → 「3 个节点 ACTIVE，16 条命令已登记；对账：一致 20 键，缺 0，多 0」。
- **Notes**:
  - 设计文档 §7 的 P2-3 与本台账既有编号无冲突，仍沿用「设计阶段号 + 序号」登记为
    `P2-3-1`；
  - 未实现「同一键在两种传输间的运行期在线迁移」：切换发生在装配配置层（`mode`），
    这与 `lifecycle="static"` 的声明一致，也避免把 `.so` 卸载误当成可撤销；
  - `apply` 默认 `in-process`，`xs plugin apply` 不预启动 worker；子进程在首次
    `invoke` 时按需拉起，避免 CLI 装配引入后台进程；
  - 「切换后端不改调用方代码」在本试点中的口径是：同一能力键的提供者可在
    `in-process` 与 `subprocess` 间替换，调用方只依赖 `FFIAPI`；跨名称替换
    （如 caffe → npu）仍须换键，因为契约以名称隔离能力域。

