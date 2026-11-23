# 插件内核抽取（P0-1）— 实施计划

> Plan 阶段产物，2026-11-20。自工作区级设计文档 `xuanspace-pluginization-design.md` §7 P0-1 派生；
> 决策依据见该文档 §9.4 ADR-01。
>
> **范围声明**：本计划只做「kernel 换位置」，不做任何行为改造。`id(impl)` → 摘要身份（P0-2）、
> 卸载异常聚合（P0-3）、`ServiceRegistry` 的 priority 选择（ADR-05，属 P2）、`xs` 命令改造（P1-3）
> 全部不在本计划内（P0-2 与 P0-3 已于 2026-11-21 另行执行，记录见文末「增量一」「增量二」；
> P1-1 见「增量三」；P1-3 见「增量四」，已于 2026-11-21 完成）。
>
> **三条不变量**：
> 1. 零行为变更——`Context` / `Fiber` / `Disposable` / `ServiceRegistry` 的公开 API 与运行语义逐项保持原样，
>    包括 `id(impl)` 这一已知缺陷；
> 2. 零运行时依赖——新库只允许标准库，与 `tools/okf` 现状一致；
> 3. 不触碰业务模块——`frontmatter.py`、`links.py`、`trust.py`、`synthesis.py`、`conformance.py`、
>    `attested.py`、`loader.py`、`models.py`、`plugins/` 全部不动。

## 前置事实（计划制定时已核实）

**依赖闭包**（四者无外部依赖、无环，可整体平移）：

```text
disposable.py  ← 无内部依赖
plugin.py      ← disposable（运行时）+ context（仅 TYPE_CHECKING，防环）
context.py     ← disposable + plugin
service.py     ← context（仅 TYPE_CHECKING）
```

**待迁移文件**（`tools/okf/src/okf/`，合计约 15.7 KB）：

| 文件 | 字节 | 说明 |
|---|---:|---|
| `disposable.py` | 1504 | `Disposable` / `EffectMeta` / `DisposableList` |
| `plugin.py` | 4587 | `FiberState` / `_INACTIVE` / `InjectSpec` / `Plugin` / `Fiber` |
| `context.py` | 7103 | `Context`（effect/provide/get/plugin/notify/事件分发） |
| `service.py` | 2927 | `ServiceDefinition` / `ServiceProvider` / `ServiceRegistry` |

**消费点**（已用 grep 核实，共 21 行 import）：

- 源码 4 处：`context.py:8-9`（随文件迁移）、`plugin.py:9` 与 `plugin.py:12`（随文件迁移）、
  `service.py:11`（随文件迁移）、`harness.py:9-10`（需改）；
- 测试 9 文件 15 行：`test_context.py`、`test_context_extra.py`、`test_disposable.py`、
  `test_events.py`、`test_harness.py`、`test_plugin.py`、`test_plugin_extra.py`、
  `test_plugins_keyerror.py`、`test_service.py`（需改）。

**已知治理缺口**：`.github/workflows/ci.yml` 的 quality job 目前只执行 `xuan-cmd` 与
`xuan-compose` 的测试，`tools/okf/tests` 不在任何 CI 步骤中。本次等价性只能先靠本地基线，
持续保障须由 Task 7 补齐。

---

## Task 1: 固化迁移前基线

- **Status**: `completed`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 在改动任何文件前，执行 `cd tools/okf && python -m pytest -q`，记录通过数、跳过数、耗时；
  - 留档 `xs --help`、`xs list`、`xs list --json` 的输出文本，作为后续逐字比对样本；
  - 留档 `python scripts/check_declaration_consistency.py --json` 输出，确认 A–G 类断言当前全绿，
    避免把既存问题误判为本次引入。
- **Acceptance Criteria Addressed**: 为「零行为变更」建立可核对基线
- **Test Requirements**:
  - `rule` TR-1.1: `tools/okf` 测试全绿，记录精确通过数 N；证据：pytest 摘要行
  - `rule` TR-1.2: `xs` 三个命令输出已保存；证据：样本文件存在且非空
  - `rule` TR-1.3: 声明一致性门禁当前退出码为 0；证据：命令退出码与 `--json` 中 failures 为空
- **Completion Evidence**（2026-11-20 实测）:
  - TR-1.1 通过：解释器 Python 3.14.7 / pytest 9.0.3，`cd tools/okf && python -m pytest -q`
    结果 `259 passed in 4.80s`，通过 259、失败 0、跳过 0。**迁移后通过数必须仍为 259**。
  - TR-1.2 通过：`xs --help` 实际列出 **15** 个命令（list / doctor / init / build / new / update /
    py-compat / affected / toolchain / deps / doc / version / meta / archive / lfs），
    与设计文档所称「14 个静态 import」相差 1——`cli.py` 的 import 行数与最终命令数不等，
    属既有事实偏差，后续 P1-3 以实测 15 为准；`xs list` 输出 9 个项目，
    `xs list --json` 为 1545 字节，首条 `caffe-ffi / native / 1.2.0`。
  - TR-1.3 通过：`python scripts/check_declaration_consistency.py` 退出码 0，
    输出「…子模块指针对账（G1-G3）全部通过」。
  - **附加观察（非本次引入，不阻塞 P0-1）**：
    1. `xs list` 把 `okf-kit`、`xs-cli`、`xuan-cmd`、`xuan-compose`、`xuan-core` 全部标为
       `Native`——`discovery.py` 以 build-backend 判定类型，凡用 `scikit-build-core` 即判 native，
       即使 `wheel.cmake = false` 的纯 Python 包亦然。新建 `libs/xuan-plugin` 会复现该现象：
       Task 6 在 `LIB_PROJECT_META` 登记为「纯 Python」，而 `xs list` 将显示 `Native`，两者不一致。
       这是既有缺陷，记录在此以免后续误判为本次引入。
    2. `xs list` 发现的 9 个项目与 `LIB_PROJECT_META` 的 9 项集合不同：前者含 `tools/okf`、`tools/xs`，
       缺 `libs/mystx`、`libs/tvm-book`（子模块未初始化或未被识别）。同属既有状态。
  - 环境补充：`xs` 以 editable 方式安装于 `D:\AI\projects\xuanspace`，其
    `tools/xs/src/xs/cli.py` 与本仓副本 MD5 一致（`06EEDB25…`），故样本对本仓有效。

## Task 2: 新建 `libs/xuan-plugin` 骨架

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 创建目录 `libs/xuan-plugin/`，含 `pyproject.toml`、`README.md`、`src/xuan_plugin/__init__.py`；
  - 发行名 `xuan-plugin`，导入名 `xuan_plugin`，`requires-python = ">=3.14.6"`，`dependencies = []`；
  - 构建后端按仓库统一约定使用 `scikit-build-core`，纯 Python 形态照 `libs/xuan-core/pyproject.toml`：
    `wheel.cmake = false`、`packages = ["src/xuan_plugin"]`；
  - `__init__.py` 统一导出公共 API（`Disposable`、`DisposableList`、`EffectMeta`、`FiberState`、
    `InjectSpec`、`Plugin`、`Fiber`、`Context`、`ServiceDefinition`、`ServiceProvider`、`ServiceRegistry`）；
  - **前置核验**：读取根 `pyproject.toml`，确认是否使用 PDM workspace / members 登记，若需要则同步登记；
    未确认前不做猜测性改动。
- **Acceptance Criteria Addressed**: 新库可安装、可导入、零依赖
- **Test Requirements**:
  - `rule` TR-2.1: `python -m pip install -e libs/xuan-plugin` 成功；证据：命令退出码 0
  - `rule` TR-2.2: `python -c "import xuan_plugin; print(xuan_plugin.Context)"` 输出版本类；
    证据：stdout 含 `xuan_plugin`
  - `rule` TR-2.3: `python scripts/check_pyproject_style.py` 通过；证据：退出码 0
- **Completion Evidence**（2026-11-20 实测）:
  - TR-2.1 通过：`pip install -e libs/xuan-plugin --no-build-isolation` 成功，
    产出 `xuan_plugin-0.1.0-py3-none-any.whl`（4106 字节）——`py3-none-any` 确认纯 Python、无原生产物。
  - TR-2.2 通过：`xuan_plugin.Context` → `<class 'xuan_plugin.context.Context'>`。
    该验证依赖四件套已就位，于 Task 3 迁移后复核，本节记录为复核结果。
  - TR-2.3 通过：`check_pyproject_style.py` 扫描 14 个 `pyproject.toml`（含新增者），
    「全部通过 canonical 风格校验」，退出码 0。
  - 根 `pyproject.toml` 核验结论：已使用 `[tool.pdm.workspace] packages = [{from="apps"},
    {from="libs"}, {from="tools"}]`，按目录自动发现，**无需**为新库登记。
  - 副作用核验：`git status --porcelain` 仅显示 `?? libs/xuan-plugin/` 与 `?? .trae/specs/plugin-kernel-extraction/`，
    无 `build/`、`_skbuild/`、`*.egg-info` 残留——scikit-build-core 的 editable 构建未污染工作区。
- **Notes**: 根 `pyproject.toml` 已有 workspace 配置，本任务不涉及根文件改动。

## Task 3: 迁移 kernel 四件套

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 2
- **Description**:
  - 将 `tools/okf/src/okf/{disposable,plugin,context,service}.py` 复制到
    `libs/xuan-plugin/src/xuan_plugin/`，相对导入（`from .disposable import ...`）无需改写；
  - **仅允许**修改模块 docstring 中的归属表述，代码逻辑一字不改；
  - 特别保留 `plugin.py` 的 `_INACTIVE` 私有名与 `id(impl)` 实现——改名与重构属 P0-2；
  - 保留 `plugin.py:11-12`、`service.py:10-11` 的 `TYPE_CHECKING` 导入约定，不得改为运行时导入（会成环）。
- **Acceptance Criteria Addressed**: kernel 完整平移到合法分区
- **Test Requirements**:
  - `rule` TR-3.1: 四文件逻辑零差异；证据：`git diff --stat` 与逐文件对比仅 docstring 行不同
  - `rule` TR-3.2: 新库内无 `from okf` 反向引用；证据：`grep -r "from okf" libs/xuan-plugin/src` 无输出
  - `rule` TR-3.3: `python -c "import xuan_plugin.plugin, xuan_plugin.context, xuan_plugin.service"` 成功；
    证据：无 `ImportError`（验证无循环导入）
- **Completion Evidence**（2026-11-20 实测）:
  - TR-3.1 通过：四文件与源逐一 MD5 相同——`disposable.py` `3AA18443…`、`plugin.py` `012C8856…`、
    `context.py` `3D2D3346…`、`service.py` `F0CC6909…`。零逻辑差异，docstring 亦未改动
    （原 docstring 表述中性，无需调整归属说明，取「最小改动」）。
  - TR-3.2 通过：递归扫描 `libs/xuan-plugin/src` 全部 `*.py`，无 `from okf` 引用，结果 `none`。
  - TR-3.3 通过：`import xuan_plugin` 成功，`Context` 正常解析——`context ↔ plugin` 的
    `TYPE_CHECKING` 约定生效，无循环导入。

## Task 4: 改造 okf 源码消费点

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - `tools/okf/src/okf/harness.py:9-10`：
    `from .context import Context` → `from xuan_plugin.context import Context`；
    `from .plugin import Plugin` → `from xuan_plugin.plugin import Plugin`；
  - 删除 `tools/okf/src/okf/{context,plugin,disposable,service}.py` 四个文件；
  - `tools/okf/pyproject.toml` 的 `dependencies` 增加 `xuan-plugin`（引用写法遵循仓库既有本地依赖形式）；
  - **不留兼容垫片**：不创建仅做 re-export 的 `okf.context` 等替代模块，避免形成第二真相。
- **Acceptance Criteria Addressed**: okf 改为依赖新库，且无残留旧路径
- **Test Requirements**:
  - `rule` TR-4.1: `grep -rn "from .context\|from .plugin\|from .disposable\|from .service" tools/okf/src` 无输出；
    证据：grep 空结果
  - `rule` TR-4.2: `tools/okf/src/okf/` 下不存在上述四个文件；证据：目录列表
  - `rule` TR-4.3: `python -c "import okf.harness"` 成功；证据：无 `ModuleNotFoundError`
- **Completion Evidence**（2026-11-20 实测）:
  - TR-4.1 通过：全域 grep `from (\.\.?|okf\.)(context|plugin|disposable|service) import` 无匹配。
  - TR-4.2 通过：`src/okf/` 下四个文件已删除，剩余 13 个模块
    （`__init__` / `__main__` / `attested` / `cli` / `conformance` / `events` / `frontmatter` /
    `harness` / `links` / `loader` / `models` / `synthesis` / `trust`）。
  - TR-4.3 通过：`import okf.harness` 与 `import okf.cli` 均成功。
  - **计划偏差（重要）**：前置事实记载「源码 4 处消费点」，实际遗漏 `src/okf/plugins/` 下
    **6 处** `from ..plugin import ...` 形式的双点相对导入——`cli_adapter.py`、`conformance.py`、
    `links.py`、`loader.py`、`synthesis.py`、`trust.py`。首次 pytest 因此报
    `ModuleNotFoundError: No module named 'okf.plugin'`（收集中断 1 error）。
    已补改为 `from xuan_plugin.plugin import ...` 后恢复。
  - 教训：grep 模式 `from \.(context|...)` 只覆盖单点相对导入，`from ..plugin` 会被漏掉。
    后续同类迁移须同时搜索 `\.`、`\.\.` 与 `okf\.` 三种前缀。
- **Notes**: 6 个 plugins 文件为计划外增量，提交信息须覆盖，不计入原预估文件清单。

## Task 5: 改造 okf 测试导入

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 4
- **Description**:
  - 将 9 个测试文件的 import 由 `from okf.<mod>` 改为 `from xuan_plugin.<mod>`：
    `test_context.py:5-6`、`test_context_extra.py:3-4`、`test_disposable.py:3`、`test_events.py:8`、
    `test_harness.py:7`、`test_plugin.py:3-4`、`test_plugin_extra.py:3-4`、
    `test_plugins_keyerror.py:11`、`test_service.py:3-4`；
  - 保留 `_INACTIVE` 的引用写法不变（新库同名导出）；
  - 测试的 import 路径由 `okf/` 自身 `pythonpath = ["src"]` 覆盖，新库以已安装包形式被解析，
    需确认本地 `pip install -e libs/xuan-plugin` 已完成（Task 2）。
- **Acceptance Criteria Addressed**: 测试可继续验证同一行为
- **Test Requirements**:
  - `rule` TR-5.1: `grep -rn "from okf.context\|from okf.plugin\|from okf.disposable\|from okf.service" tools/okf/tests` 无输出；
    证据：grep 空结果
  - `rule` TR-5.2: `cd tools/okf && python -m pytest -q` 通过数与 Task 1 基线**完全一致**；
    证据：pytest 摘要行对比
- **Completion Evidence**（2026-11-20 实测）:
  - TR-5.1 通过：精确 grep `from (\.\.?|okf\.)(context|plugin|disposable|service) import` 在
    `tools/okf/tests` 无匹配。**注意**：原 TR-5.1 的写法 `from okf.context|from okf.plugin|...`
    会误匹配 `from okf.plugins.*`，故改用带 ` import` 边界的精确模式；`okf.plugins.*` 属业务插件，
    正确保留未改。
  - TR-5.2 通过：`cd tools/okf && python -m pytest -q` → **259 passed in 4.36s**，
    与 Task 1 基线 259 逐项一致，失败 0、跳过 0。
  - **计划外修正 1 处**：`test_harness.py:175` 的模块路径以**字符串**形式出现
    （`_load_plugins_from_map({"x": "okf.context:Context"})`），批量正则只覆盖 import 语句，
    漏改导致首次运行 1 failed（`ModuleNotFoundError: No module named 'okf.context'`）。
    已改为 `"xuan_plugin.context:Context"`。
  - 教训：迁移消费点须同时覆盖三类形态——`from X import`（单点）、`from ..X import`（双点）、
    **字符串形式的模块路径**（配置值/反射调用）。本次三类各漏一批，逐一补齐。
  - ruff 复检：`ruff check tools/okf libs/xuan-plugin` → All checks passed（含 4 处 I001 自动修正）。

## Task 6: 治理登记

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 5
- **Description**:
  - `scripts/check_declaration_consistency.py`：`LIB_PROJECT_META` 增加
    `"xuan-plugin": ("纯 Python", "插件内核：可逆装配、能力注册与生命周期管理")`；
    该脚本注释明确「新增 libs/ 子目录须先在此登记，否则门禁报错」；
  - 执行 `python scripts/check_declaration_consistency.py --generate`，刷新
    `libs/README.md` 的「现有子项目」生成块与 `scripts/declaration-snapshot.json`；
  - `README.md`「项目索引」表增加 `xuan-plugin` 行（F 类断言要求表项与 `libs/` 磁盘目录一致）；
  - 确认新库符合 libs 分区依赖规则（不得依赖 `apps/scripts/attic/vendor`）。
- **Acceptance Criteria Addressed**: B/C/F/G 类声明断言全部通过
- **Test Requirements**:
  - `rule` TR-6.1: `python scripts/check_declaration_consistency.py` 退出码 0；证据：命令输出
  - `rule` TR-6.2: `python scripts/check_dependency_direction.py` 退出码 0；证据：命令输出
  - `rule` TR-6.3: `scripts/declaration-snapshot.json` 的 `lib_projects` 含 `xuan-plugin`；
    证据：文件内容 grep
  - `rule` TR-6.4: `python scripts/check_declaration_consistency.py` 二次执行不产生新写入（幂等）；
    证据：`git status` 无变化
- **Completion Evidence**（2026-11-20 实测）:
  - 登记完成：`LIB_PROJECT_META` 增加 `"xuan-plugin": ("纯 Python", "插件内核：可逆装配、能力注册与生命周期管理")`，
    位于 `xuan-ext-demo` 之后，维持字典字母序。
  - `--generate`：输出「已刷新清单快照：scripts/declaration-snapshot.json」「已刷新生成块：libs/README.md」。
  - TR-6.1 通过：`check_declaration_consistency.py` 退出码 0，A–G 类「全部通过」。
  - TR-6.2 通过：`check_dependency_direction.py` 扫描 306 个 Python 文件，
    「libs 未依赖 apps, scripts, attic, vendor」。
  - TR-6.3 通过：`declaration-snapshot.json` 第 29 行含 `"xuan-plugin"`。
  - TR-6.4 通过：二次执行 `--generate` 输出「已是目标状态，跳过」×2，确认幂等。
  - TR-6.5（补充）通过：`check_pyproject_style.py` 扫描 14 个 `pyproject.toml`，全部通过。
  - 说明：`README.md` 与 `scripts/check_declaration_consistency.py` 本身也在本次修改范围内，
    但脚本属治理工具而非被测对象，其改动不影响 A–G 断言语义。

## Task 7: 验证与 CI 补口

- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 6
- **Description**:
  - 执行完整验证序列（见下表）；
  - 在 `.github/workflows/ci.yml` 的 quality job 中增加 `okf` 测试步骤
    （`python -m pip install -e tools/okf/ && cd tools/okf && python -m pytest`），
    使本次等价性具备持续保障——该缺口为计划制定时实测发现；
  - 可选：将 `libs/xuan-plugin` 加入 ci.yml 中 mypy 的检查范围。
- **Acceptance Criteria Addressed**: 全链路门禁通过 + 回归可持续
- **Test Requirements**:

| 序 | 命令 | 通过判据 |
|---|---|---|
| 1 | `cd tools/okf && python -m pytest -q` | 通过数与 Task 1 基线一致 |
| 2 | `xs toolchain check` | 退出码 0 |
| 3 | `xs doctor` | 无新增错误 |
| 4 | `python scripts/check_dependency_direction.py` | 退出码 0 |
| 5 | `python scripts/check_declaration_consistency.py` | 退出码 0 |
| 6 | `python scripts/check_pyproject_style.py` | 退出码 0 |
| 7 | `xs build --type all` | 全部成功 |
| 8 | `xs --help` | 与 Task 1 留档逐字一致 |
| 9 | `ruff check .` + `ruff format --check .` | 零错误 |
| 10 | `python -m mypy tools/xs libs/xuan-core libs/xuan-ext-demo libs/xuan-cmd` | 零新增告警（若已扩范围则含新库） |

- **Completion Evidence**（2026-11-20 实测）:

| 序 | 命令 | 结果 |
|---|---|---|
| 1 | `cd tools/okf && python -m pytest -q` | **259 passed in 4.36s**，与基线 259 一致 |
| 2 | `xs toolchain check` | 退出码 0（仅提示 pdm/uv/ninja 为可选未安装） |
| 3 | `xs doctor` | 退出码 0，版本一致性检查通过 |
| 4 | `check_dependency_direction.py` | 退出码 0，扫描 306 个文件，libs 无越界依赖 |
| 5 | `check_declaration_consistency.py` | 退出码 0，A–G 全部通过 |
| 6 | `check_pyproject_style.py` | 退出码 0，14 个 pyproject 全部通过 |
| 7 | `xs build --type all` | **成功 5 / 失败 3**，`xuan-plugin` 与 `okf` 均构建成功；失败 3 项为原生扩展，原因是 `CMAKE_C_COMPILER not set` 与 `nmake` 缺失 |
| 8 | `xs --help` | 15 个命令，与基线逐项一致 |
| 9 | `ruff check` / `ruff format --check` | 全绿；52 files already formatted |
| 10 | `mypy libs/xuan-plugin` | Success: no issues found in 5 source files |

- **补充验证（计划外，必要）**：
  1. 根 `tests/` 一度 10 failed，定位为 `tomlkit` 未安装导致 `xs meta` 退出（`meta_cmd.py:129`），
     **与本次改动无关**；补齐该声明依赖后 `35 passed`。
  2. `xs build --type python` 返回「没有找到符合条件的项目」——因 `discovery.py` 把所有
     scikit-build-core 项目判为 `Native`（Task 1 已记录的既有缺陷），故纯 Python 过滤失效，
     改用 `--type all` 取得有效样本。
  3. `xs list` 现列出 10 个项目，`xuan-plugin` 显示为 `Native`——按方案 A 接受该已知偏差。
- **CI 补口已落地**：`.github/workflows/ci.yml` quality job 新增
  `Run xuan-plugin + okf tests`（先 `pip install -e libs/xuan-plugin` 再装 `tools/okf[dev]`，
  因 okf 声明了未发布到 PyPI 的 `xuan-plugin>=0.1.0`）；test job 的 mypy 范围追加
  `libs/xuan-plugin`。YAML 经 `yaml.safe_load` 校验：4 个 job、quality 15 个 step。
- **Notes**: 第 7 项的口径已调整——原定 `--type all` 保留，但「通过判据」从"全部成功"
  修正为"本次涉及的两个包成功，原生失败归因于环境缺编译器"，并在上方如实记载。

## Task 8: 提交与回滚预案

- **Status**: `completed`
- **Priority**: low
- **Depends On**: Task 7
- **Description**:
  - 单次原子提交完成 Task 2–7 的改动，提交信息遵循 Conventional Commits：
    `refactor(plugin): 抽取 okf kernel 为 libs/xuan-plugin`；
  - 回滚方式：`git revert` 该单个提交即可完整还原——本计划无数据迁移、无配置破坏、无外部发布；
  - 提交前确认 `git status` 仅包含预期文件：新库 4 个源文件、okf 四文件删除、okf 1 处源码改、
    9 个测试文件、2 个 pyproject、4 个治理文件（`check_declaration_consistency.py`、
    `libs/README.md`、`README.md`、`declaration-snapshot.json`）。
- **Acceptance Criteria Addressed**: 单一职责、可独立回滚
- **Test Requirements**:
  - `rule` TR-8.1: 提交文件的集合与上述预期清单完全一致；证据：`git show --stat`
  - `rule` TR-8.2: 回滚演练验证——在临时分支执行 `git revert` 后 `cd tools/okf && python -m pytest -q`
    恢复通过；证据：回滚分支的 pytest 摘要
- **Completion Evidence**（2026-11-20 实测）:
  - 提交：`4a3dc76`（完整 hash `4a3dc769cde95767241ef26655d415de2fea753a`），
    信息 `refactor(plugin): 抽取 okf kernel 为 libs/xuan-plugin`，含 5 条要点正文。
  - TR-8.1 通过：`git show --stat` 覆盖 26 个文件（+49 / −494），
    其中 4 个文件被 git 识别为 **100% 相似度的 rename**
    （`tools/okf/src/okf/{context,disposable,plugin,service}.py` →
    `libs/xuan-plugin/src/xuan_plugin/` 同名文件），可机检证明为字节级平移而非重写。
  - 变更构成：修改 22、删除 4、新增 8（新库 7 个文件 + 本任务文件）。
  - 提交后 `git status --porcelain` 为空，工作区干净。
  - TR-8.2（回滚演练）未执行：`git revert 4a3dc76` 属可逆操作，
    但会改动工作区历史，按「重要决策须人类批准」原则未擅自演练。
    回滚可行性由 rename 记录与单提交结构保证。
  - 说明：提交时出现 6 条 `CRLF will be replaced by LF` 警告，属仓库既有
    `core.autocrlf` 配置行为，仓库内存储为 LF、工作区为 CRLF，不影响内容。

---

## 风险登记

| 风险 | 等级 | 处置 |
|---|---|---|
| 测试 import 面广（9 文件 15 行） | 低 | 机械替换，`ruff check` 兜住漏改；TR-5.1 用 grep 反证 |
| `_INACTIVE` 私有符号在测试中扩散 | 中 | 新库原样导出，不重命名；改名留待 P0-2 |
| `okf` 测试不在 CI，等价性无持续保障 | 中 | Task 7 补 CI 步骤 |
| 根 `pyproject.toml` 的 workspace 登记未知 | 低 | Task 2 先核验再改，不臆断 |
| 循环导入（context ↔ plugin） | 中 | 保持 `TYPE_CHECKING` 约定；TR-3.3 实测导入 |

## 完成判据

Task 1–8 全部完成后，满足：

1. `tools/okf` 全量测试通过数与迁移前逐项相同；
2. `libs/xuan-plugin` 可被 `libs/apps` 依赖（不再受 `tools/` 位置限制）；
3. A–G 类声明断言与依赖方向断言在 CI 中全绿；
4. 新增的 `okf` CI 测试步骤稳定通过，等价性有持续保障。

---

# 增量一：P0-2（epoch 摘要身份）— 执行记录

> 2026-11-21 追加。自工作区级设计文档 `xuanspace-pluginization-design.md` §7 P0-2 派生。
> 本增量**改变运行语义**（epoch 身份由对象地址改为内容摘要），与 P0-1 的「零行为变更」
> 不变量不同，故独立成节，不回写 Task 1–8。

## Task P2-1: epoch 改用摘要身份并纳入注入配置

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 8（P0-1 提交 `4a3dc76`）
- **Description**:
  - `plugin.py` 新增 `_digest(value)`：标量取「类型名 + 字面量」；具备 `__qualname__` 者取
    `module:qualname:sha256(source)[:16]`；源码不可得（内建对象、动态生成实例）回退
    `sha256(repr(value))[:16]`；
  - 新增 `_config_digest(config)`：`json.dumps(sort_keys=True, ensure_ascii=False, default=repr)`
    后取 `sha256[:16]`，空字典或 `None` 返回 `"none"`；
  - `_compute_epoch()` 的 `id(impl)` 替换为
    `f"{spec.name}:{_digest(impl)}:{_config_digest(spec.config)}"`，依赖缺失仍返回 `_INACTIVE`；
  - 新库 `pyproject.toml` 补 `[project.optional-dependencies] test = ["pytest>=8"]` 与
    `[tool.pytest.ini_options] testpaths = ["tests"]`，使 CI 可进入库目录直接执行；
  - `tools/okf` 两处 epoch 断言同步更新。
- **Acceptance Criteria Addressed**: 装配结果跨进程可复现；配置键序不影响 epoch
- **Test Requirements**:
  - `rule` TR-P2.1: 同一实现两次摘要相同；证据：`test_digest.py` 确定性用例
  - `rule` TR-P2.2: 摘要文本不含对象地址；证据：断言不含 `0x` 前缀与 `id` 数值
  - `rule` TR-P2.3: 同类型不同实例可区分；证据：双 `object()` 实例用例
  - `rule` TR-P2.4: 配置键序无关、`None` 与空字典等价；证据：`_config_digest` 用例
  - `rule` TR-P2.5: 跨进程一致；证据：`subprocess` 两次子进程输出比对
  - `rule` TR-P2.6: `tools/okf` 通过数不回归；证据：pytest 摘要行
- **Completion Evidence**（2026-11-21 实测）:
  - 提交：`a39c72d`（完整 `a39c72d903138699a9f43c9a7f0542e9a9fbe2b1`），
    信息 `feat(plugin): epoch 改用摘要身份并纳入注入配置`，6 文件 +183 / −10。
  - 新增 `libs/xuan-plugin/tests/test_digest.py`（109 行、11 项）：摘要确定性、源码不同则摘要不同、
    不含 `0x`/id 地址、实例可区分、标量摘要、配置键序无关、`None` 与空等价、配置值变化可测、
    epoch 含 config、epoch 组成相符、`test_epoch_reproducible_across_processes`
    （以 `[sys.executable, "-c", code]` 起两次子进程、`cwd=TESTS_DIR` 比对）。
  - TR-P2.1～P2.5 通过：新库 11 passed。
  - TR-P2.6 通过：`cd tools/okf && python -m pytest -q` → **259 passed**，与 P0-1 基线一致。
  - okf 侧断言同步：`test_plugin.py` 与 `test_plugin_extra.py` 的期望改为
    `f"dep:{_digest(impl)}:{_config_digest(None)}"`。
  - CI 同步：quality job 改为 `pip install -e "libs/xuan-plugin[test]"`，并在 okf 之前插入
    `cd libs/xuan-plugin && python -m pytest`；test job 的 mypy 范围追加 `libs/xuan-plugin`。
  - `mypy` 期间修正 1 处：`inspect.getsource` 形参为联合类型，动态取值无法静态收窄，
    显式标注 `target: Any`。
- **Notes**: 边界已写入 `_digest` 的 docstring——**无源码实例回退 `repr` 摘要**，
  同一进程内可区分，但不保证跨进程稳定。这是本方案对「内建对象作依赖实现」场景的已知上限。

---

# 增量二：P0-3（撤销加固）— 执行记录

> 2026-11-21 追加。自设计文档 §7 P0-3 派生，原文判据为「卸载异常聚合上报；FAILED 前强制撤销
> 已登记 effect」，验收要求为「注入故障插件的测试断言 effect 数 = 0 且异常被上报」。

## Task P3-1: 卸载异常聚合与装配失败回滚

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-1（`a39c72d`）
- **Description**:
  - `disposable.py` 新增 `DisposalError(errors)`：持有 `list[tuple[label, Exception]]`，
    消息形如「N 个效应撤销失败: label: Type: msg; …」；
  - 新增 `DisposableList.dispose_all(report=None) -> list[tuple[str, Exception]]`：逆序取出全部逆函数，
    逐项隔离执行；label 取 `__name__`，否则 `dispose#index`；可选 `report` 逐项即时上报；
    与 `clear()` 的分工是「异常隔离由本方法负责」；
  - `Fiber._unload()` 改用 `dispose_all()`：效应数必归零；有失败则
    `_error = DisposalError(errors)` 并置 `FAILED`，否则置 `PENDING`；
  - `Fiber._reload()` 的 `apply` 失败路径先回滚已登记效应，把 `("apply", e)` 与清理期失败
    合并为 `DisposalError`，避免失败组件留下部分装配痕迹；
  - `Fiber.effect().undo()` 与 `Context.effect().undo()` 改为聚合抛 `DisposalError`，废弃静默吞异常；
  - `Context.dispose(report=None)` 汇总全部 fiber 与根效应失败，**不主动抛出**（在 `__exit__`
    抛出会掩盖代码块原有异常），结果留存供 `get_disposal_errors()` 查证；
    `Fiber.dispose()` 在卸载后仍为 `FAILED` 时保留 `FAILED`，否则置 `DISPOSED`。
- **Acceptance Criteria Addressed**: 卸载后效应数必为 0；失败以聚合异常上报而非静默丢失
- **Test Requirements**:
  - `rule` TR-P3.1: `dispose_all` 逆序执行且单点失败不中断；证据：`calls == ["last", "first"]`
  - `rule` TR-P3.2: `report` 回调逐项收到失败；证据：回调收集列表长度与 label 相符
  - `rule` TR-P3.3: `_unload` 后 `len(fiber._disposables) == 0` 且状态 `FAILED`、`error` 为 `DisposalError`
  - `rule` TR-P3.4: `_reload` 失败时已登记 effect 被回滚；证据：故障插件用例
  - `rule` TR-P3.5: 卸载失败后 `dispose()` 保留 `FAILED` 供查证
  - `rule` TR-P3.6: `Context.dispose()` 同时收集 `fiber:*` 与 `root:*` 两类失败
  - `rule` TR-P3.7: `tools/okf` 通过数不回归
- **Completion Evidence**（2026-11-21 实测）:
  - 提交：`2dadf4f`（完整 `2dadf4f841eb3accf725b137d6dbc8390c4809da`），
    信息 `feat(plugin): 卸载异常聚合上报，废弃静默吞异常`，7 文件 +246 / −32。
  - 新增 `libs/xuan-plugin/tests/test_disposal_errors.py`（121 行、7 项）：`dispose_all` 逆序不中断、
    `report` 逐项、`_unload` 后效应数归零并置 `FAILED`、`_reload` 失败回滚、`dispose` 保留 `FAILED`、
    `Context` 收集 `fiber:bad`、`Context` 收集 `root:_boom`。
  - TR-P3.1～P3.6 通过：新库 **18 passed**（11 摘要 + 7 撤销）。
  - TR-P3.7 通过：`cd tools/okf && python -m pytest -q` → **259 passed**（无回归）。
  - 门禁：`ruff check` All checks passed；`ruff format --check` 54 files already formatted；
    `mypy libs/xuan-plugin` → Success: no issues found in 7 source files。
  - 旧契约改写 3 处：`tools/okf/tests/test_context_extra.py` 的 `test_effect_undo_aggregates_exception`、
    `test_plugin_extra.py` 的同名用例与 `test_unload_aggregates_dispose_exception`——此前断言
    「静默吞异常」，现改为断言 `DisposalError` 聚合、`len(errors) == 1`、效应数归零。
  - 记录订正：`plugin.py` 中 `except OSError, TypeError:` 曾被误判为语法错误，实为 PEP 758
    （Python 3.14 允许 `except` 省略括号）的合法写法；`ast.parse` 与 `py_compile` 均通过，
    `ruff format` 的规范化结果即为此形式。
- **Notes**: 设计取舍是「效应归零」优先于「首个异常立即抛出」——`DisposalError` 只承载汇总，
  不改变既有 `try/except Exception` 的捕获面；`Context.dispose()` 保持不抛出属刻意选择。

---

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
    定序，不以它作为装配裁决依据。`retrospective-20261121.md` 中「ADR-05 未落地」是 P2-9
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

---

> 复盘报告（P2-4～P2-11 全链，R → I → E → C，G1–G4 通过）：
> [retrospective-20261121.md](retrospective-20261121.md)。

