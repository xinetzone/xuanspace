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
    本轮在装配器按异常对象去重，未改动 `Fiber` 语义（留待后续按需修正）。

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
    （承接 P1-2 的 A4 待办）。

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

