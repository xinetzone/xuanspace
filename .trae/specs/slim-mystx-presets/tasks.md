# Tasks — slim-mystx-presets

> 每项任务 = 一个原子提交，可独立回滚。全部任务在 `libs/mystx` 仓库内执行。
> 验证命令统一在仓库根运行：`$env:PYTHONPATH='src'; python -m pytest tests/unit -q`、`python -m ruff check src/mystx/presets`。

- [x] Task 1: 声明表对齐上游 `_Opt` 形状并删除包装器（`src/mystx/presets/mystx_site.py`）
  - [x] SubTask 1.1: `ConfigSpec` 的默认值字段改为「值或工厂」两态，装配时 `callable` 判断求值；docstring 写明「可变值必须用工厂」
  - [x] SubTask 1.2: 删除 `_constant()`；4 个静态值条目改写为字面值直传
  - [x] SubTask 1.3: `SITE_CONFIG_SPEC` 19 条改为单行声明，**键序与改动前逐字一致**
  - [x] SubTask 1.4: `_assert_spec_contract()` 调用点由 `build_config()` 移至模块导入期；删除 `apply_config()` 中不可达的 `undeclared` 分支
  - [x] SubTask 1.5: `_BuildContext.gettext_compact` 类型改为 `bool | str`，与 SPEC 的 `types` 声明对齐
  - [x] 验证：`pytest tests/unit/test_presets.py` 全绿；`ruff` 零告警；SPEC 声明块 ≤35 行
  - [x] 提交：`refactor(presets): 声明表对齐上游 Config 形状并删除冗余校验`

- [x] Task 2: 极简预设删除三写一（`src/mystx/presets/minimal_myst.py`）
  - [x] SubTask 2.1: 删除 `build_project_meta(params)` 之后对 `extensions`/`html_theme`/`html_theme_options` 的重复赋值
  - [x] SubTask 2.2: `override = override or {}` + `dict(override)` 合并为一次构造
  - [x] 验证：`pytest tests/unit/test_presets.py -k minimal` 全绿；返回字典键集与取值不变
  - [x] 提交：`refactor(presets): 极简预设删除重复赋值`

- [x] Task 3: 站点画像隔离策略收敛（`src/mystx/presets/mystx_docs.py`、`src/mystx/presets/_shared.py`）
  - [x] SubTask 3.1: 取值表整体构造后**一次** `deepcopy` 注入，替换 3 处逐键 `deepcopy`
  - [x] SubTask 3.2: `build_project_meta` 的默认值合并收敛为一次字典合并
  - [x] 验证：`pytest tests/unit/test_conf_py.py tests/unit/test_presets.py` 全绿；`configure` 注入键值逐项不变
  - [x] 提交：`refactor(presets): 站点画像改为整体深拷贝注入`

- [x] Task 4: 新增 3 条防回归测试（`tests/unit/test_presets.py`）
  - [x] SubTask 4.1: 断言连续两次 `build_config()` 的 `locale_dirs`/`inheritance_graph_attrs`/`nitpick_ignore`/`suppress_warnings` 互不为同一对象
  - [x] SubTask 4.2: 断言 `SITE_CONFIG_SPEC` 键序等于显式键列表（顺序漂移即失败）
  - [x] SubTask 4.3: 断言键集契约在缺键/多键时抛出含差异键名的 `RuntimeError`（改为直接测试纯函数 `_check_spec_contract`，避免 reload 干扰）
  - [x] SubTask 4.4: 补第 4 条用例，断言静态默认值按值共享且为不可变类型（抵消删除守卫用例后的总数缺口）
  - [x] 验证：`pytest tests/unit/test_presets.py` 全绿，用例数 19 → 22；`pytest tests/unit` 134 passed / 3 skipped
  - [x] 提交：`test(presets): 补齐默认值隔离与键序契约`

- [x] Task 5: 全量验证与文档同步
  - [x] SubTask 5.1: `pytest tests/unit -q` ≥134 passed / 3 skipped；`ruff check src/mystx/presets` 零告警
  - [x] SubTask 5.2: 对比 19 键取值与键序快照，逐项相等（基线取自 `git archive bd53ebf`）
  - [x] SubTask 5.3: `sphinx-build -b html doc <临时目录>` 退出码 0，告警数不高于基线
  - [x] SubTask 5.4: 更新 `presets` 各模块 docstring 中关于元数据消费状态的表述；核对 `doc/dev/sphinx-conf.md` 的「缺口三项」描述与新实现一致
  - [x] 提交：`docs(presets): 同步声明层元数据与校验时机说明`

# Task Dependencies

- Task 2 与 Task 3 相互独立，可在 Task 1 之后并行。
- Task 4 依赖 Task 1（键序与导入期校验）与 Task 3（隔离语义）。
- Task 5 依赖 Task 1–4 全部完成。
- 快照基线（19 键取值）必须在 Task 1 动手前生成并留存。
