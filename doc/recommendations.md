# Xuanspace（玄境）优化建议书

> **配套文档**：[项目蓝图](blueprint.md)
> **Session**：`sc-20261027-xuanspace-boshu-blueprint`
> **基线**：`HEAD = 8db593d5fc1ee999c8311f83951e40f15024e2d9`，工作树干净
>
> **分级依据**：P0＝影响信任与可验证性，必须先做；P1＝一致性缺口，应做；P2＝精确化，可选。
> **本条清单不含「正确的废话」**：每条建议都指向一个具体的文件位置与一个可判定的验收动作；凡无法给出验证方式的条目，已从清单中剔除。

---

## 汇总表

| 编号 | 级别 | 一句话问题 | 影响面 |
|---|---|---|---|
| P0-1 | P0 | 根文档目录名 `docs/` 与 `doc/` 在多个文件内共存 | 7 处引用（跨 6 个文件） |
| P0-2 | P0 | `apps/` 为空，但文档宣称其含「竹简悟道」「xs-cli」 | 3 处声明 |
| P0-3 | P0 | 声明「3 个上游子模块」，实际 4 个 | 1 处声明 |
| P0-4 | P0 | 无任何机制校验「声明与磁盘事实」一致 | 全套门禁 |
| P1-1 | P1 | 命令清单三处口径不一（12 vs 15） | 3 处声明 |
| P1-2 | P1 | 依赖声明示例与实际 `pyproject.toml` 不一致 | 1 处示例 |
| P1-3 | P1 | 构建后端版本下限声明不一（`>=0.5` vs `>=0.11`） | 1 处声明 |
| P1-4 | P1 | 三份「现有子项目」清单与磁盘不符 | 4 处清单 |
| P1-5 | P1 | `doc/README.md` 对目录结构与主题的描述与实现不符（已随「文档入口合并」消解） | 2 处段落 |
| P1-6 | P1 | 两份元数据规范互相矛盾且无可判定触发条件 | 2 份规范 |
| P2-1 | P2 | 命名溯源引文未标注版本 | 1 处引文 |
| P2-2 | P2 | `doc/architecture.md` 存在上下文漂移 | 1 处段落 |
| P2-3 | P2 | mypy 检查范围口径不一 | 2 处表述 |
| P2-4 | P2 | Mermaid 图中存在重名节点 | 1 处图 |

---

## P0 · 影响信任与可验证性

### P0-1　统一根文档目录命名口径为 `doc/`

- **判据**：仓库根及根级资产中，凡指**根文档目录**处一律写 `doc/`，不再出现 `docs/`。判定范围为根 `AGENTS.md`／`README.md`／`CONTRIBUTING.md`／`CHANGELOG.md` 与 `.agents/`／`doc/`／`tools/`／`scripts/`／`tests/`／`apps/`／`attic/`／`.trae/`。
- **排除项（同类字串，但**不属**本条款，不纳入修正）**：
  - **语义不同**：[CONTRIBUTING.md](../CONTRIBUTING.md#L316) 第 316 行的**分支命名** `docs/<name>`。
  - **有意兼容实现**：[docs_cmd.py](../tools/xs/src/xs/commands/docs_cmd.py#L21) 第 21／24 行与 [doctor_cmd.py](../tools/xs/src/xs/commands/doctor_cmd.py#L200) 第 200／210 行按 `("doc", "docs")` 依次回退查找目录；[test_doctor_sphinx.py](../tests/test_doctor_sphinx.py#L19) 第 19 行注释「兼容通用 docs/ 命名」。这三处是正确实现，改动会收窄兼容面。
  - **历史记录与归档**：[CHANGELOG.md](../CHANGELOG.md#L12) 第 12 行的 **v0.1.0 发布记录**（子命令列写 `…version/docs/meta…`；该行为发布历史，其中 `docs` 子命令已改名为 `xs doc`，但**历史发布记录不改写**）；[attic/README.md](../attic/README.md#L72) 第 72 行（归档目录内的示例说明）；`.trae/specs/unify-scikit-build-core/` 下 7 处引用 `docs/build-system.md`（历史 spec 记录，该文件现为 [doc/build-system.md](build-system.md)）。
  - **子包与第三方自有命名空间**：`libs/caffe-ffi/docs/`、`libs/npu-ffi/docs/` 等子包自建目录；`vendor/` 与 `libs/mystx`、`libs/tvm-book` 三个 git submodule。
- **现状证据（7 处／6 个文件）**（原始清点为 11 处／8 个文件；其中 3 处位于原 `doc/README.md`、1 处为 `.agents/context-routing.md` 第 17 行的断链，均已在「文档入口合并」变更中一并消除，见[蓝图事实 #29](blueprint.md)，故不再列为待修正项）：
  1. [AGENTS.md](../AGENTS.md#L66)：目录树写 `├── docs/              # 文档目录（Sphinx+MyST）`；而同文件 [AGENTS.md](../AGENTS.md#L96) 又写 `配置见\`doc/conf.py\``。
  2. [.agents/ONBOARDING.md](../.agents/ONBOARDING.md#L103)：目录树写 `├── docs/       # Sphinx文档`。
  3. [CHANGELOG.md](../CHANGELOG.md#L13)：写 `Monorepo 目录结构（apps/libs/vendor/tools/docs/scripts/attic/.agents/.meta）`。
  4. [CONTRIBUTING.md](../CONTRIBUTING.md#L709)：写 `文档使用 **Sphinx + MyST Markdown** 编写，位于 \`docs/\` 目录。`
  5. [CONTRIBUTING.md](../CONTRIBUTING.md#L714)：第 12.1 节目录树首行写 `docs/`。
  6. [doc/architecture.md](architecture.md#L71)：目录树写 `├── docs/              # Sphinx 文档源文件`。
  7. [doc/dependency-report.md](dependency-report.md#L28)：目录树写 `├── docs/                           # Sphinx + MyST 文档`。
- **反证（正确一侧）**：[README.md](../README.md#L152) 写 `├── doc/           # Sphinx 文档源文件`；[doc/conf.py](conf.py#L101) 写 `"path_to_docs": "doc"`；磁盘目录名为 `doc/`。
- **期望态**：上述 7 处统一为 `doc/`，与磁盘实际、`conf.py` 的 `path_to_docs` 一致。
- **落地动作**：逐处将 `docs/` 改为 `doc/`；改后须复跑断链检查与 Sphinx 构建，确认无新增断链、无新增告警。（原 `doc/README.md` 与 `.agents/context-routing.md` 相关 4 处已在「文档入口合并」变更中完成，见[蓝图事实 #29](blueprint.md)。）
- **附带决策项（已裁定并落地）**：[init_cmd.py](../tools/xs/src/xs/commands/init_cmd.py#L296-L306) 的 `xs init` 脚手架原会创建 `docs/` 目录，其模板文本（第 35 行 `!docs/make.bat`、第 124 行 `` - `docs/` — 项目文档 ``）与之一致，但与根仓约定相反。该决策项已裁定为「脚手架改用 `doc/`」，并随 `xs docs` → `xs doc` 命令改名一并落地：脚手架现创建 `doc/`，模板文本同步更新为 `!doc/make.bat` 与 `` - `doc/` — 项目文档 ``。
- **验证方式**：
  1. 逐点复核上列 7 处的最终文本。
  2. 在仓库根执行
     `rg -n "docs/" --hidden --glob '!vendor/**' --glob '!libs/**' --glob '!.git/**' --glob '!.trae/**' --glob '!attic/**' --glob '!doc/blueprint.md' --glob '!doc/recommendations.md'`
     剩余命中应为 **7 行**，且**全部落在白名单**内：`docs_cmd.py:21/24`、`doctor_cmd.py:200/210`、`test_doctor_sphinx.py:19`、`CONTRIBUTING.md:316`、`CHANGELOG.md:12`；白名单之外应为零命中。
     **必须带 `--hidden`**：`rg` 默认不扫隐藏目录，而本条款 7 处中有 1 处位于 `.agents/`（`ONBOARDING.md`）。缺该参数时命令会**静默漏报该处并显示「修复已生效」**——这正是本条建议所针对的失效模式本身。
  3. 随后 `pdm run sphinx-build -b html doc/ doc/_build/html` 无新增告警。

### P0-2　修正 `apps/` 相关失实声明

- **判据**：文档对 `apps/` 的描述与 `apps/` 的磁盘内容一致（现为仅 `.gitkeep` 与 [README.md](../apps/README.md)）。
- **现状证据**：[doc/architecture.md](architecture.md#L59-L60) 的目录树写 `├── apps/  # 可执行应用和 CLI 工具`、`│   ├── culture/       #   文化类应用（子目录分组）`；[doc/architecture.md](architecture.md#L87) 表格写 `| apps/ | 可执行应用、CLI 工具、面向最终用户的程序 | 竹简悟道（HTML/JS 文化应用） |`；[apps/README.md](../apps/README.md#L87-L92) 的「现有子项目示例」表列出 `xs-cli/` 与 `zhujian-wudao/`；[tools/README.md](../tools/README.md#L32) 写「`apps/xs-cli/` 只是入口点包装，实际实现在 `tools/xs/` 中」，而 `apps/xs-cli/` 不存在，`xs` 的实际发行版名称为 `xs-cli`（[tools/xs/pyproject.toml](../tools/xs/pyproject.toml#L6)）。
- **期望态**：`apps/` 的描述改为「当前为空目录，预留应用位」；`xs-cli/`、`zhujian-wudao/` 若为规划内容，须显式标注「（示例／规划中）」；[tools/README.md](../tools/README.md#L32) 的句子改为与事实相符的表述（`xs` 的实现位于 `tools/xs/`，无可执行的 `apps/xs-cli/` 包装层）。
- **落地动作**：改写上述 3 处声明，并统一「示例」与「现有」的措辞边界——建议明确区分「现有」（磁盘存在）与「示例」（模板示意）两类。可参照 [libs/README.md](../libs/README.md#L102-L111) 的「示例」标题用法。
- **验证方式**：人工比对 `apps/` 实际内容（`.gitkeep` + `README.md`）与三处文档描述；断言检查纳入 P0-4 的门禁。

### P0-3　修正上游子模块数量声明

- **判据**：文档声明的子模块数量与条目列表，与 [.gitmodules](../.gitmodules) 一致。
- **现状证据**：[CONTRIBUTING.md](../CONTRIBUTING.md#L212) 写「Xuanspace 仓库包含 3 个上游子模块」，其下表格仅列 [CONTRIBUTING.md](../CONTRIBUTING.md#L216) `libs/tvm-book`、[#L217](../CONTRIBUTING.md#L217) `vendor/tvm-ffi`、[#L218](../CONTRIBUTING.md#L218) `vendor/caffe` 三行；而 [.gitmodules](../.gitmodules) 声明 4 个条目，多出 `libs/mystx`（`url = git@github.com:xinetzone/mystx.git`）。
- **期望态**：数字改为 4，表格补齐 `libs/mystx` 一行（含 URL 与用途说明），并同步 [CONTRIBUTING.md](../CONTRIBUTING.md#L223-L224) 的「初始化时机」表述。
- **落地动作**：替换数字、补表行、检查同节内其余「3」的计数表述。
- **验证方式**：`git config -f .gitmodules --list` 得到的条目数应为 4；`rg -n "3 个上游子模块" CONTRIBUTING.md` 应无命中。

### P0-4　建立「声明—实现一致性」门禁（关系位落地）

- **判据**：仓库中至少存在一条自动检查，能在「文档声明」与「磁盘事实」不一致时使 CI 失败。
- **现状证据**：[.github/workflows/ci.yml](../.github/workflows/ci.yml#L106-L151) 的全部 step 覆盖代码风格、类型、测试、文档构建与链接检查，无一条比对文档声明与磁盘内容；[scripts/README.md](../scripts/README.md) 目录仅含 [check_pyproject_style.py](../scripts/check_pyproject_style.py) 一个校验脚本；[README.md](../README.md#L101) 的「15 个子命令」与 [tools/xs/src/xs/cli.py](../tools/xs/src/xs/cli.py#L43-L88) 的实际注册数之间无断言。
- **期望态**：新增至少一条「声明一致性」检查 step，覆盖可机械提取的三类声明——子项目清单（对照 `xs list`）、CLI 命令清单（对照命令注册表）、目录名口径（对照磁盘）。
- **落地动作**：在 [scripts/README.md](../scripts/README.md) 下新增校验脚本（须先查阅 SpecWeave 侧 [.agents/scripts/lib/README.md](../../../.agents/scripts/lib/README.md) 共享库以避免重复实现），并在 [.github/workflows/ci.yml](../.github/workflows/ci.yml) 的 `quality` job（[#L156-L200](../.github/workflows/ci.yml#L156-L200)）中调用。该脚本应接受「生成模式」与「校验模式」两种运行方式：生成模式刷新可提取清单，校验模式在差异存在时退出码非 0。
- **验证方式**：故意把 [README.md](../README.md#L101) 的「15」改为「14」，本地或 CI 运行该检查应失败；改回后应通过。**这是本条的验收动作，不通过则本条视为未完成。**

---

## P1 · 一致性缺口

### P1-1　统一 CLI 命令清单口径

- **判据**：所有列出 `xs` 命令的位置，其条目集合与 [tools/xs/src/xs/cli.py](../tools/xs/src/xs/cli.py) 的实际注册一致。
- **现状证据**：[doc/intro.md](intro.md#L43-L58) 的「项目状态」表列 12 个命令（`list`/`new`/`build`/`doctor`/`init`/`deps`/`version`/`docs`/`meta`/`toolchain`/`py-compat`/`update`），缺 `archive`、`lfs`、`affected`；[CHANGELOG.md](../CHANGELOG.md#L12) 同样列 12 个；而 [tools/xs/src/xs/cli.py](../tools/xs/src/xs/cli.py#L43-L88) 注册了 7 个命令组（`toolchain`/`deps`/`docs`/`version`/`meta`/`archive`/`lfs`）与 8 个直接命令（`list`/`doctor`/`init`/`build`/`new`/`update`/`py-compat`/`affected`），共 15 个一级项；[README.md](../README.md#L101) 亦写「15 个子命令」。
- **期望态**：三处口径统一为 15 个，且由同一来源派生（手工补齐或生成）。
- **落地动作**：在 [doc/intro.md](intro.md#L43-L58) 的表中补 `xs archive`、`xs lfs`、`xs affected` 三行；核对 [CHANGELOG.md](../CHANGELOG.md#L12) 的表述（历史 changelog 可保留原样，但须确认其不构成现状声明）。
- **验证方式**：以 `xs --help` 的实际输出条目数为准，逐条比对两处清单。

### P1-2　同步依赖声明示例与实际配置

- **判据**：文档中的依赖声明示例与根 [pyproject.toml](../pyproject.toml) 的实际内容一致。
- **现状证据**：[doc/architecture.md](architecture.md#L100-L104) 的示例为 `docs = ["sphinx>=8.0", "myst-parser>=4.0"]`（2 项）与 `dev = ["pdm", "typer>=0.12", "xuanspace[docs,test,lint,build]"]`（引用式）；实际 [pyproject.toml](../pyproject.toml#L39-L88) 的 `docs` 含 6 项（`sphinx`、`myst-parser`、`sphinx-book-theme`、`sphinx-design`、`sphinx-copybutton`、`sphinxcontrib-mermaid`），`dev` 为显式展开而非 `xuanspace[docs,test,lint,build]` 引用式。
- **期望态**：示例与实体一致，或明确标注为「示意片段（非完整清单）」。
- **落地动作**：更新 [doc/architecture.md](architecture.md#L100-L104) 的示例，或在段首加「以下为示意，完整清单见 `pyproject.toml`」。
- **验证方式**：逐项比对示例与 [pyproject.toml](../pyproject.toml#L39-L88)。

### P1-3　统一 scikit-build-core 版本下限

- **判据**：全部文档中的 `scikit-build-core` 下限声明与全仓实际一致（`>=0.11`）。
- **现状证据**：[libs/README.md](../libs/README.md#L102) 的示例写 `requires = ["scikit-build-core>=0.5"]`；实际全仓（[pyproject.toml](../pyproject.toml#L1-L3)、[tools/xs/pyproject.toml](../tools/xs/pyproject.toml#L2)、[tools/okf/pyproject.toml](../tools/okf/pyproject.toml#L2)、[libs/xuan-core/pyproject.toml](../libs/xuan-core/pyproject.toml#L2)、[libs/xuan-ext-demo/pyproject.toml](../libs/xuan-ext-demo/pyproject.toml#L2)）统一为 `>=0.11`。
- **期望态**：示例改为 `>=0.11`。
- **落地动作**：替换该处版本号；核查 [libs/README.md](../libs/README.md) 同节是否另有 `minimum-version` 表述需同步。
- **验证方式**：`rg -n "scikit-build-core>=" libs/ tools/ pyproject.toml` 的返回值应一致。

### P1-4　把「现有子项目」清单改为生成物或断言对象

- **判据**：三份「现有子项目」清单的条目集合与磁盘目录一致。
- **现状证据**：[apps/README.md](../apps/README.md#L87-L92) 列 2 项（均不存在）；[libs/README.md](../libs/README.md#L106-L111) 列 2 项，而 `libs/` 实含 8 个子项目目录（`caffe-ffi`、`demo-ffi`、`mystx`、`npu-ffi`、`tvm-book`、`xuan-compose`、`xuan-core`、`xuan-ext-demo`）；[tools/README.md](../tools/README.md#L10-L17) 未列 `tools/templates/native-ffi/`；[README.md](../README.md#L99-L109) 的「项目索引」表列 9 行。
- **期望态**：清单由脚本从磁盘目录或 `xs list` 派生；若保留手写，则由 P0-4 的检查断言其完整性。
- **落地动作**：二选一——① 引入生成标记（如 `<!-- BEGIN GENERATED -->`）由脚本注入；② 在 P0-4 的脚本中增加「清单条目 ⊆ 磁盘目录」断言。
- **验证方式**：新增任一子项目目录后运行检查，清单应被提示缺失。

### P1-5　同步文档入口的结构与主题描述（已随「文档入口合并」消解）

- **判据**：文档入口对目录结构与主题的描述与 [doc/conf.py](conf.py) 及 `doc/` 实际一致。
- **现状证据（原始）**：原 `doc/README.md` 第 11 行称使用 `sphinx-autodoc2`（从 Python 源码自动生成 API 文档），第 12 行称主题「（待配置，推荐 furo 或 sphinx-rtd-theme）」；而实际 [doc/conf.py](conf.py#L27-L33) 启用的是 `sphinx.ext.autodoc`（非 autodoc2），第 67–72 行的主题回退链为 `mystx` → `sphinx_book_theme` → `alabaster`。其第 17–44 行的目录树（`getting-started/`、`user-guide/`、`api-reference/`、`developer-guide/`、`architecture/`）与 `doc/` 的扁平实际结构不符。
- **处置结果**：原 `doc/README.md` 已删除，其独有内容并入 [doc/index.md](index.md)，并在迁移时一并修正上述两处技术栈错误（`sphinx-autodoc2` → `sphinx.ext.autodoc`、主题写成实际回退链），失真的目录树改由 `toctree` 承担导航。本条款随之消解。
- **验证方式**：比对 [doc/index.md](index.md) 的「技术栈」节与 [doc/conf.py](conf.py#L27-L72)。

### P1-6　消解元数据规范冲突并定义「Task 20」触发条件

- **判据**：两份规范对同一动作不再给出相反指令；「当前阶段」有可判定的结束条件。
- **现状证据**：[.meta/README.md](../.meta/README.md#L23-L43) 规定 Markdown 文档须携带 YAML frontmatter（4 字段）并配 `.meta/toml/` 镜像，且给出 [.meta/README.md](../.meta/README.md#L45-L66) 的 `doc/quickstart.md ↔ .meta/toml/doc/quickstart.toml` 示例；[.agents/rules/frontmatter.md](../.agents/rules/frontmatter.md#L7) 明言「当前阶段（Task 20 之前）不要添加 YAML frontmatter」，[.agents/rules/frontmatter.md](../.agents/rules/frontmatter.md#L103) 亦写「❌ 不要手动添加 YAML frontmatter」。实际 `.meta/toml/` 仅含 5 个文件。
- **期望态**：以 [.agents/rules/frontmatter.md](../.agents/rules/frontmatter.md) 为当前阶段的唯一权威（规则层优先于说明层）；[.meta/README.md](../.meta/README.md#L23) 顶部加一段「当前阶段冻结」提示并指向规则文件；同时为「Task 20」定义可判定触发条件（例如「当 `xs meta` 具备生成 `.meta/toml/` 镜像的能力时」）。
- **落地动作**：① 在 [.meta/README.md](../.meta/README.md) 增补冻结说明；② 在 [.agents/rules/frontmatter.md](../.agents/rules/frontmatter.md#L7) 或任务清单中写入 Task 20 的判定条件；③ 明确本轮新增的两份文档（[blueprint.md](blueprint.md)、[recommendations.md](recommendations.md)）按冻结期处理——不加 frontmatter、不建 TOML 镜像。
- **验证方式**：两份文件对「当前是否添加 frontmatter」的回答一致；新增文档经检查确认无 frontmatter 且 `.meta/toml/` 未新增同名文件。

---

## P2 · 精确化（可选）

### P2-1　命名溯源引文补标注版本

- **判据**：引用《老子》原文处标注所据版本。
- **现状证据**：[doc/intro.md](intro.md#L5-L7) 引「道可道，非常道；名可名，非常名……玄之又玄，众妙之门」，采用传世本面貌，未标注版本。按版本纪律，马王堆甲本／乙本相应文句作「道可道也，非恒道也」「名可名也，非恒名也」（乙本此句不标残损；甲本此句残，所据「恒」字为帛书通释；「恒」改「常」为汉文帝刘恒讳）。
- **期望态**：标注为「传世本（王弼本系统）面貌」，或同时给出马王堆甲本／乙本面貌对照。
- **落地动作**：在 [doc/intro.md](intro.md#L7) 引文后加一行版本标注。
- **验证方式**：引文旁存在版本标注字样。

### P2-2　修正 `doc/architecture.md` 的上下文漂移

- **判据**：文档中的「本仓库」指代对象一致。
- **现状证据**：[doc/architecture.md](architecture.md#L144-L148) 的「自建库（projects/）」章节称「在 SpecWeave 主仓库中作为 Git 子模块」——该段落的语境是 Xuanspace 自身位于 `projects/xuanspace/` 之下，此处「主仓库」指代与其所在文件的语境不同。
- **期望态**：改为与当前语境一致的表述（如「作为上层工作区的 Git 子模块」）。
- **落地动作**：改写该段首句。
- **验证方式**：通读该节无指代歧义。

### P2-3　对齐 mypy 检查范围口径

- **判据**：CI 实际检查范围与文档声明的范围不冲突（或已注明差异）。
- **现状证据**：[.github/workflows/ci.yml](../.github/workflows/ci.yml#L113) 的 mypy 仅检查 `tools/xs libs/xuan-core libs/xuan-ext-demo` 三个目标；[AGENTS.md](../AGENTS.md#L111) 的验收基线声明「mypy 零新增告警」（未限定范围）。
- **期望态**：在 [.github/workflows/ci.yml](../.github/workflows/ci.yml#L113) 或 [AGENTS.md](../AGENTS.md#L111) 注明「范围以 CI 为准」。
- **落地动作**：二选一——扩大 mypy 范围，或在文档中注明范围。
- **验证方式**：两者表述不再相互隐含冲突。

### P2-4　消除 Mermaid 图中重名节点

- **判据**：同一 Mermaid 图内节点标识唯一。
- **现状证据**：[README.md](../README.md#L123-L124) 的 `法` 子图内出现两行同名节点 `Conventions[项目规范]` 与 `Conventions[AGENTS.md]`。
- **期望态**：节点标识唯一（如 `Conventions` 与 `AgentsMd`）。
- **落地动作**：重命名其一，并同步图中连线引用。
- **验证方式**：图渲染后两个节点分别可见，不互相覆盖。

---

*本建议书与配套的 [项目蓝图](blueprint.md) 同批次产出；两份文档均以 H1 开头，不携带 YAML frontmatter，不创建 `.meta/toml/` 镜像。本轮不执行任何建议项，仅提出清单。*
