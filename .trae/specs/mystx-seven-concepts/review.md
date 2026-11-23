# mystx 七概念复盘与优化方案

> 基于 R-I-F-A-V-C 链路（重构优化场景，叠加 R 作学习底座）对 `mystx` 0.3.x 的全量源码学习、复盘、洞察与优化编排。

> **归档说明**：2026-11-23 自 `doc/reports/mystx-seven-concepts-20261123/mystx-seven-concepts-review-and-plan-20261123.md` 移入 xuanspace 仓 `.trae/specs/mystx-seven-concepts/`，更名 `review.md`，正文未改。

## 文档元信息

| 项 | 内容 |
|---|---|
| 分析对象 | `mystx`（Sphinx 主题 + 配置编排 + invoke 任务 CLI + 指令扩展） |
| 源码位置 | 服务器 `/media/pc/data/ai/notebook/dev/libs/AI/projects/xuanspace/libs/mystx`（Windows 侧 `\\10.16.11.3\ai\notebook\dev\libs\AI\projects\xuanspace\libs\mystx`） |
| 对象版本 | 0.3.x（版本号由 `pyproject.toml` 的 `version_file` 指向 `src/mystx/_version.py`，无静态 `version` 字段） |
| 仓库 HEAD | `ffc3902`（2026-11-23，`build(doc): 字体配置改为可选守卫…`），提交总数 69 |
| 体量 | `src/mystx` Python 1854 行；`tests` 10 个文件 875 行 |
| 方法论 | 七概念：R 复盘（G1）→ I 洞察（G2）→ F 第一性原理（V 门强制）→ A 原子化（G4 前置）→ V 对抗审查（V 门）→ C 原子提交（G4）；链路 `I→F→A→V→C`，R 作为学习底座；E 萃取本次未启用 |
| 场景判定 | 重构优化（触发词「优化」「技术债」，强制 V 门） |
| 生成日期 | 2026-11-23 |
| 证据口径 | 关键断言均在服务器侧直接复核（`ssh ai@10.16.11.3`），以最后一次带输出的命令结果为准；SMB 侧元数据可能有缓存，不作为终判 |

---

## 一、执行摘要

`mystx` 的功能面是完整的：主题继承、MyST/Notebook 双解析器互斥注入、TOML 配置合成、`invoke doc.*` 三层 CLI（builder-mode 5 个 + `-M` 目标 3 个）、4 条 GitHub 只读指令，均已在文档站实际运行。问题不在功能，而在**契约层与边界层**：

1. **打包契约失真**：`pyproject.toml` 声明 `dependencies = []`，而包顶层直接导入 Sphinx；README 另行声明 4 项核心依赖。三者互不一致，`pip install mystx` 后 `import mystx` 的可用性不由声明保证。
2. **公共 API 无真实消费者**：仓库自带的 `doc/conf.py` 自行实现 `_has()`、三级主题回退与 sitemap 分支，绕开了包内同名能力。公共层因此缺少「被真实使用」的验证回路。
3. **破坏性操作缺边界**：`clean` 逐级 `os.rmdir` 上溯无 buildroot 边界；`pinned_repo` 拼接 HTML 未转义；`thebe` 先改配置后装扩展。
4. **隐式魔法多**：`"."`/`"_build"` 哨兵值、`Path.cwd()` 兜底、未知 TOML 键静默丢弃，失败点远离成因。
5. **版本与文档熵**：`SW_*` 兼容层承诺 `mystx>=0.7` 移除但无时间表；扩展 docstring 的指令数与模块名均错；README 示例含 `NameError` 与错误配置键。

**优先级建议**：先做零行为变更的元数据与边界修复（AC-4/AC-5/AC-6/AC-8），再处理需人类裁决的契约二选一（AC-1/AC-2），最后补测试闭环（AC-9）。9 个行动项均可独立验收。

---

## 二、学习地图

### 2.1 分层结构

| 层 | 模块 | 职责 |
|---|---|---|
| 装配层 | `__init__.py`、`pyproject.toml`、`theme/mystx/` | 包入口 `setup()`、entry point 注册、主题资产 |
| 配置层 | `config/defaults.py`、`config/runtime.py`、`presets/` | 静态默认与 `build_base_config`；`config-inited` 运行期注入；预设组合 |
| 主题层 | `themes.py`、`theme/mystx/theme.toml`、`layout.html` | 主题名单、选项解析、模板继承（`layout.html` 仅一条 `extends`） |
| 适配层 | `extensions.py`、`myst_compat.py`、`ext/github_readme_stats/` | 可选扩展装配、MyST 版本兼容别名、4 条只读指令 |
| 执行层 | `tasks/doc/`（`builder.py`、`_core.py`、`_namespace.py`） | `invoke doc.<name>` 任务实现与参数归一 |
| 工具层 | `utils.py`、`version_switcher.py` | 环境判定、版本切换清单 |

### 2.2 三条主调用链

- **启动链**：Sphinx 加载 `mystx` 扩展 → `setup()`（互斥注入 `myst_nb`/`myst_parser`，注册主题与配置）→ `config-inited` 处理器（`ConfigManager` 合成并注入）→ 主题与扩展就绪 → 构建。
- **配置合成链**：`defaults` 静态默认 → `presets` 叠加 → 项目 `_config.toml` → `runtime.deep_merge` → 写入 `app.config`。**该链中只有 `html_theme_options` 被实际消费**（`config/runtime.py:86`）。
- **CLI 链**：`invoke doc.*` → `_resolve`/`_anchor_root` 归一目录 → `run_cmd`（builder-mode）或 `_run_target`（`-M` 模式）→ 子进程 `sphinx-build`。

---

## 三、R 阶段：客观事实（G1，31 条）

事实条目仅陈述可验证状态，不含因果判断。

### 3.1 打包与入口（R-01 ~ R-06）

| 编号 | 事实 | 证据位置 |
|---|---|---|
| R-01 | 声明 `requires-python = ">=3.14"` | `pyproject.toml:12` |
| R-02 | 声明 `dependencies = []`；同时 `src/mystx/__init__.py` 在模块顶层导入 Sphinx 相关对象 | `pyproject.toml:37`；`src/mystx/__init__.py` |
| R-03 | 版本号由 `version_file = "src/mystx/_version.py"` 提供，文件内无静态 `version` 字段 | `pyproject.toml:196` |
| R-04 | 依赖清单在 `[dependency-groups]` 与 `[project.optional-dependencies]` 两处各写一份 | `pyproject.toml` |
| R-05 | 主题注册存在两条路径：`[project.entry-points."sphinx.html_themes"]` 与 `setup()` 内 `app.add_html_theme` | `pyproject.toml`；`src/mystx/__init__.py` |
| R-06 | `setup()` 中 `myst_nb` 与 `myst_parser` 按可用性互斥注入（约 55–81 行） | `src/mystx/__init__.py` |

### 3.2 配置运行期（R-07 ~ R-11）

| 编号 | 事实 | 证据位置 |
|---|---|---|
| R-07 | `deep_merge` 调用只把 `custom_config["html_theme_options"]` 作为消费对象（`remove_marker=None`） | `src/mystx/config/runtime.py:86` |
| R-08 | 先写入 `launch_buttons["thebe"] = True`（119–123 行），再调用 `app.setup_extension("sphinx_thebe")`（129 行） | `src/mystx/config/runtime.py` |
| R-09 | 同文件 147 行注释说明「动态加载应走 `setup_extension`」 | `src/mystx/config/runtime.py:147` |
| R-10 | 默认配置把 `mermaid` 登记为围栏到指令的映射：`myst_fence_as_directive: ["mermaid"]` | `src/mystx/config/defaults.py:23` |
| R-11 | 扩展装配表中 `"sphinxcontrib.mermaid"` 一项被注释 | `src/mystx/extensions.py:63` |

### 3.3 任务 CLI（R-12 ~ R-17）

| 编号 | 事实 | 证据位置 |
|---|---|---|
| R-12 | `clean` 用 `while parent ... os.rmdir(parent)` 逐级回收空父目录，循环条件不含 buildroot 判定 | `src/mystx/tasks/doc/builder.py:55-58` |
| R-13 | `tree` 在 walk 根层级打印目录名（113 行），又在父级分支重复打印一次（115–116 行） | `src/mystx/tasks/doc/builder.py:103-118` |
| R-14 | `doctest` 以字面量 `"sphinx-build"` 组装命令 | `src/mystx/tasks/doc/builder.py:132` |
| R-15 | `build` 仅在 `opts is None` 时注入 `-b html` 与环境分流并行 flag（161–167 行），第 167 行同样使用字面量 `"sphinx-build"` | `src/mystx/tasks/doc/builder.py` |
| R-16 | `_run_target` 通过 `os.environ.get("SPHINXBUILD", "sphinx-build")` 取命令，与 R-14/R-15 的硬编码路径不一致 | `src/mystx/tasks/doc/_core.py:80` |
| R-17 | `browse` 以 `connect_ex` 循环递增端口探测（81–88 行），`webbrowser.open(url)` 在 `subprocess.Popen` 之前调用 | `src/mystx/tasks/doc/builder.py` |

### 3.4 路径与预设（R-18 ~ R-21）

| 编号 | 事实 | 证据位置 |
|---|---|---|
| R-18 | `_anchor_root` 从 `Path(__file__).resolve().parents` 逐级上溯查找含 `conf.py` 的目录（33–42 行） | `src/mystx/tasks/doc/_namespace.py` |
| R-19 | 上溯未命中时返回 `Path.cwd()`（42 行） | `src/mystx/tasks/doc/_namespace.py:42` |
| R-20 | `minimal_myst` 预设的 `extensions_required` 默认值为 `("myst_parser",)` | `src/mystx/presets/minimal_myst.py:26` |
| R-21 | 同文件 15 行仍引用 `okf_docs` 预设 | `src/mystx/presets/minimal_myst.py:15` |

### 3.5 扩展与兼容层（R-22 ~ R-26）

| 编号 | 事实 | 证据位置 |
|---|---|---|
| R-22 | `create_image_node` 生成 `<img>` 时对 `url` 与 `alt` 调用 `escape` | `src/mystx/ext/github_readme_stats/base.py:51` |
| R-23 | `pinned_repo` 直接拼接 `'<a href="{link}"><img src="{url}" alt="Pinned Repo"></a>'`，未调用 `escape`；65 行回退到 `create_image_node` | `src/mystx/ext/github_readme_stats/pinned_repo.py:63,65` |
| R-24 | `github_readme_stats/__init__.py` 注册 4 条指令（实测 `DIRECTIVES=4`），其 docstring 声明模块名 `mystx.ext.github_stats` 与「三条指令」 | `src/mystx/ext/github_readme_stats/__init__.py` |
| R-25 | 提供 `SW_*` 兼容别名，并输出「removed in mystx>=0.7」日志，仓库内无对应移除时间表 | `src/mystx/myst_compat.py:39-52` |
| R-26 | `sections` 处理以 `len(first.children) <= 1` 为条件删除首个 section | `src/mystx/myst_compat.py:122-126` |

### 3.6 消费者与文档（R-27 ~ R-31）

| 编号 | 事实 | 证据位置 |
|---|---|---|
| R-27 | 文档站 `conf.py` 自行实现 `_has()`（`importlib.util.find_spec`），未调用包内探测能力 | `doc/conf.py:45-46` |
| R-28 | 同文件以三级回退选择主题（`mystx` → `sphinx_book_theme` → `alabaster`） | `doc/conf.py:84-88` |
| R-29 | 同文件自建 sitemap 环境分支（`GITHUB_ACTIONS` / `READTHEDOCS` / 本地三分支） | `doc/conf.py:142-151` |
| R-30 | README 使用 `static_path = ["_static"]`（101 行）与 `"home_page_in_toc": false`（131 行），二者与有效的 Sphinx 配置名 / Python 布尔字面量不一致 | `README.md:101,131` |
| R-31 | 实测度量：`pytest` 54 passed / 3 warnings / 4.27s；覆盖率 673 语句、193 未覆盖、合计 71%，其中 `config/runtime.py` 54%、`tasks/doc/builder.py` 41%、`ext/.../base.py` 53%、`stats.py` 62%、`pinned_repo.py` 33% | 附录复现命令 |

**G1 判定**：31 条 ≥ 20 条门槛；全部附路径与行号或可复现命令，无因果词。**通过。**

---

## 四、I 阶段：四元组洞察（G2，6 条）

### I-1 打包契约失真

- **现象**：`dependencies = []`、顶层直接 `import sphinx`、README 列出 4 项核心依赖，三处口径互不相同；`dependency-groups` 与 `optional-dependencies` 双份维护。
- **根因**：依赖声明被当作「开发者本地已装好」的备忘，而非安装契约；`[doc]`/`[dev]` 两套分组沿演进叠加，未做收敛。
- **影响**：`pip install mystx` 的可用性不由元数据保证，用户首次 `import` 即失败且错误信息指向第三方包；测试环境（预装全套）无法暴露该缺口，形成「CI 绿、安装红」的盲区。极端情况下可导致下游 CI 误判。
- **建议**：把 Sphinx 与主题基座提升为硬依赖（或明确降级为惰性导入 + 明确的 `ImportError` 提示），并让 `dependency-groups` 只保留「开发专用」增量。见 AC-1/AC-2。

### I-2 公共 API 缺少真实消费者

- **现象**：`doc/conf.py` 自行实现 `_has()`、主题三级回退、sitemap 分支，与包内能力同构但不复用。
- **根因**：`conf.py` 需要在「包不可导入」时仍能降级运行，于是复制了一份最小实现；包内公共 API 因缺少使用者而未被固化。
- **影响**：同一判定逻辑存在两份实现，修订时易漂移；公共 API 演进口径只能靠测试而非真实站点验证，重构风险被隐藏。
- **建议**：把 `_has`/主题回退/sitemap 判定收敛为包内单一函数，`conf.py` 在可导入时复用、不可导入时保留最小兜底，并在文档站构建中作为该 API 的常驻验收。见 AC-2 的配套项。

### I-3 测试结构偏斜

- **现象**：合计覆盖率 71%，但风险最高的几处偏低：`tasks/doc/builder.py` 41%、`pinned_repo.py` 33%、`config/runtime.py` 54%。
- **根因**：测试围绕「配置组装」这类纯函数易测点展开，对子进程、文件系统、HTML 拼接等需替身（mock/fixture）的路径覆盖成本高，被自然回避。
- **影响**：恰是破坏性最强、最容易回归的三块（删目录、拼 HTML、装扩展）缺少护栏；`clean` 上溯删除、未转义拼接这类问题无法被现有测试捕获。
- **建议**：按「破坏性优先」补测：`clean` 的 buildroot 边界、命令入口统一、HTML 转义、`tree` 输出结构。见 AC-9。

### I-4 隐式魔法取代显式契约

- **现象**：`"."` 与 `"_build"` 作为哨兵默认值；`_anchor_root` 未命中原样返回 `Path.cwd()`；`deep_merge` 只消费 `html_theme_options`，其余键静默丢弃。
- **根因**：为减少必填参数而采用「约定即默认」，且失败路径选择静默降级而非显式报错。
- **影响**：用户在错误目录执行 `invoke doc.build` 时，得到的是「在 cwd 里找不到 conf.py」之类的下游报错，而非「未能定位 source」的即时提示；TOML 键拼错无任何反馈，形成排查成本。
- **建议**：保留默认值语义，但在解析失败/未知键时输出明确 warning（可用 `--strict` 升级为错误）。见 WP3。

### I-5 破坏性操作缺边界与顺序约束

- **现象**：`clean` 逐级 `rmdir` 无 buildroot 边界；`pinned_repo` 拼接 HTML 未转义；`thebe` 先写 `launch_buttons` 后装扩展。
- **根因**：三类问题同源——「先写爽路径，再补异常路径」的实现顺序，使边界判定被推迟到不可见。
- **影响**：`clean` 在 `target` 指向浅层目录时可能回收用户目录；未转义的 `link`/`url` 构成 HTML 注入面（尽管数据源为 GitHub 只读接口，风险等级中低，但契约上不应存在）；`thebe` 配置先于扩展就绪，若扩展缺失则留下无效配置项。
- **建议**：`clean` 增加 buildroot 白名单边界判定；`pinned_repo` 统一走 `create_image_node`；`thebe` 改为「先确认可装载、再写配置」。见 AC-4/AC-6。

### I-6 版本与文档熵

- **现象**：`SW_*` 标注 `mystx>=0.7` 移除但无时间表；扩展 docstring 模块名与指令数均错；README 含 `NameError` 示例与错误配置键；CHANGELOG 记载 `theme.toml` 与 Python 侧曾对 8 个同名键取值分歧（后以测试锁死）。
- **根因**：文档与代码在同一提交中并行演进，缺少「文档可执行/可机检」的门禁；兼容承诺缺少版本发布日期锚点。
- **影响**：用户复制 README 示例即报错；维护者无法判断 `SW_*` 何时可删；`theme.toml` 类分歧只能靠事后补测试发现。
- **建议**：README 示例纳入可执行校验（至少语法层）；`SW_*` 移除绑定具体里程碑；docstring 与注册表以测试对齐。见 AC-6/AC-8/AC-9。

**G2 判定**：6 条 ≥ 3 条门槛；每条均含现象、根因、影响、建议四要素。**通过。**

---

## 五、F 阶段：第一性原理（V 门强制）

### 5.1 回归本质：一个 Sphinx 扩展包必须回答的四个问题

剥离实现细节，`mystx` 的本质量是「让一套 Sphinx 配置在最少人工干预下可复现地生效」。由此只剩四个不可再分的命题：

1. **装得上**：安装即得到可导入的包与完整运行依赖。
2. **配得对**：声明什么就生效什么，未声明则取唯一权威默认。
3. **跑得稳**：破坏性操作有边界，失败信息指向成因。
4. **说得清**：文档示例可执行，兼容承诺有期限。

### 5.2 四条设计原则

| 原则 | 表述 | 对应病灶 |
|---|---|---|
| P1 单一权威源 | 同一事实只允许一处定义，其余为派生 | 主题选项三处字面量、`_has`/sitemap 双实现、依赖双份 |
| P2 契约先行 | 元数据声明即安装契约，不得依赖环境预置 | `dependencies = []` 与顶层 `import sphinx` 并存 |
| P3 边界显式 | 破坏性操作必须声明「可作用域」，越界即拒 | `clean` 上溯删除、未转义 HTML |
| P4 可观测可回滚 | 静默降级让位于显式提示；承诺须带失效版本 | 哨兵默认、未知键丢弃、`SW_*` 无期 |

### 5.3 五层目标架构

```
L5 交付层    README / CHANGELOG / 文档站示例（可执行、可机检）
L4 执行层    invoke doc.* —— 统一命令入口、统一根目录解析、破坏性边界
L3 适配层    扩展装配 / 兼容别名 —— 声明式清单，无隐式副作用
L2 配置层    defaults → presets → _config.toml → runtime 注入（单一合并规则）
L1 装配层    pyproject 依赖与 entry point —— 安装即可用
```

成熟度判断：L2 结构最完整（预设、运行时注入、选项一致性测试齐备）；L5 最薄弱（示例不可执行、承诺无期）；L1 契约失真；L3 有一处顺序倒置与一处未转义；L4 有一处越界删除与入口分裂。

---

## 六、A 阶段：原子化工作包（G4 前置）

| 工作包 | 目标 | 覆盖事实 | 变更性质 |
|---|---|---|---|
| WP1 打包契约对齐 | 安装即得可用包；消除依赖三口径 | R-01~R-04 | 元数据 + 可能的行为变更 |
| WP2 装配去重 | 主题注册单路径；预设引用清理 | R-05、R-20、R-21 | 纯重构 |
| WP3 配置运行期治理 | 修正 `thebe` 顺序；未知键与合并范围显式化 | R-07~R-09 | 顺序修复 + 诊断增强 |
| WP4 任务 CLI 统一 | 命令入口收敛；`clean` 加边界；`tree` 去重；`browse` 顺序 | R-12~R-19 | 重构 + 安全修复 |
| WP5 扩展安全与元数据 | HTML 转义统一；docstring 校正 | R-22~R-24 | 安全修复 + 文档修正 |
| WP6 文档与兼容治理 | README 修正；`SW_*` 时间表；测试补齐 | R-25~R-31 | 文档 + 测试 |

工作包之间无循环依赖：WP2/WP3 可在 WP1 之前独立完成；WP4 与 WP5 互不干扰；WP6 依赖其余工作包定稿后收口。

---

## 七、V 阶段：四视角对抗审查（7 条，含 1 条自我证伪）

### 7.1 四条主要意见

**V-1（魔鬼视角）· 反对「补硬依赖」**：`dependencies = []` 可能是有意设计——让包在受限环境（RTD、无网 CI）里由外层显式控制版本，避免与宿主项目的 Sphinx 版本冲突。若草率改为硬依赖，反而可能触发下游解析冲突。
→ **裁决**：部分成立，降级为「二选一需人类决策」而非直接修复。方案 A：硬依赖 + 宽松下界（`sphinx>=8`）；方案 B：保留空依赖，但把顶层导入改为惰性，并在缺失时抛出带安装指引的 `ImportError`。二者都必须落地 README 同步。

**V-2（新人视角）· 「我不知道该改哪」**：R 事实 31 条平铺，缺少「先改哪一条」的排序依据；`clean` 边界与 `pinned_repo` 转义对新人而言都是可直接上手的修复，但风险等级不同。
→ **裁决**：成立。文档补执行批次与风险分级（见 §8.2），并把「零行为变更」项前置，降低新人首改的心理门槛。

**V-3（老板视角）· 「收益是否可量化」**：除覆盖率从 71% 提升到某个数字外，其余修复（转义、边界、文档）没有可对外的收益指标；若为纯内部洁癖，优先级应下调。
→ **裁决**：部分成立。补三条可量化验收：安装后可导入成功率（AC-1）、`clean` 越界用例拦截（AC-4 的回归测试）、README 示例语法通过率（AC-8）。

**V-4（未来视角）· 「修复即冻结」**：把 `clean` 的 buildroot 边界写死为 `_build`，未来用户自定义 builddir 时反而会误拒；`SW_*` 若绑定 0.7 而 0.7 迟迟不发，兼容层会长期滞留。
→ **裁决**：成立。`clean` 边界必须由解析出的 builddir 动态推导，不得硬编码；`SW_*` 的移除改为「绑定到具体 release 分支并写入 CHANGELOG」，不依赖口头版本号。

### 7.2 三条补充意见

**V-5（魔鬼视角）· `thebe` 顺序未必是真缺陷**：若 `sphinx_thebe` 缺失时 `setup_extension` 抛异常，配置写入项会随扩展加载失败一并失效，实际不可达。攻击的是「假想路径」。
→ **裁决**：部分成立。缺陷等级从「中」降为「低」，仍建议按「先装后配」顺序书写，因为可读性与失败语义更清晰；不作为独立行动项，并入 AC-5 的同批次重构。

**V-6（老板视角）· 6 个洞察是否过载**：I-1~I-6 中 I-3（测试偏斜）与 I-4（隐式魔法）在行动项上高度重叠（都落到测试补齐与诊断增强）。
→ **裁决**：成立。行动项层面合并：I-4 的诊断增强与 I-3 的测试补齐归入同一批次验收，文档保留两条洞察的分析差异，但不再拆成两条独立交付。

**V-7（自我证伪记录）· 一条曾被采纳后推翻的假设**：分析中期曾判定「`config-inited` 内调用 `add_config_value` 注册过晚，会导致 `conf.py` 中的赋值被丢弃」，并拟列为高优先级缺陷。经服务器侧核验 Sphinx 9.2.0 的 `Config.__getattr__` 实现（未命中注册表时回退 `_raw_config`），该路径可正常读到 `conf.py` 值。
→ **裁决**：假设**不成立**，已从问题清单删除，不作为行动项。保留此条以避免后续重复提出。

### 7.3 V 门判定

| 检查项 | 要求 | 实际 | 结论 |
|---|---|---|---|
| 四视角覆盖 | 魔鬼 / 新人 / 老板 / 未来 全覆盖 | 魔鬼（V-1、V-5）、新人（V-2）、老板（V-3、V-6）、未来（V-4） | 通过 |
| 意见条数 | ≥ 5 | 7 | 通过 |
| 是否有自我证伪 | 建议具备 | V-7 | 通过 |

---

## 八、C 阶段：原子行动项与执行顺序（G4）

### 8.1 行动项清单

| 编号 | 单一职责 | 具体变更 | 验收标准 | 依赖 | 风险 |
|---|---|---|---|---|---|
| AC-1 | 打包依赖对齐（方案 A） | 将 Sphinx、主题基座等运行期必需项写入 `dependencies`，`dependency-groups` 只留开发增量 | 干净虚拟环境 `pip install .` 后 `import mystx` 成功；`pip check` 无缺依赖 | 需人类裁决 A/B | 中 |
| AC-2 | 顶层导入策略（方案 B） | 若选 B：顶层导入改惰性，缺失时抛带安装指引的 `ImportError`；同时让 `doc/conf.py` 在可导入时复用包内 `_has`/主题回退/sitemap 判定 | 未装 Sphinx 环境下 `import mystx` 给出明确指引；文档站构建走包内实现 | AC-1 二选一 | 中 |
| AC-3 | 主题注册去重 | 保留 entry point 单一注册，删除 `setup()` 内重复的 `add_html_theme` | 主题仍可被 Sphinx 发现；无重复注册告警 | 无 | 低 |
| AC-4 | `clean` 增加作用域边界 | 边界由解析出的 builddir 动态推导；越界目录拒绝删除并提示 | 新增用例：`target` 指向 buildroot 之外时不删除任何目录 | 无 | 中 |
| AC-5 | 命令入口统一 | 抽出单一命令解析（`SPHINXBUILD` 环境变量 + 默认 `sphinx-build`），`build`/`doctest`/`_run_target` 共用；顺带把 `thebe` 调整为「先装后配」 | 三个入口在设置 `SPHINXBUILD` 后均使用同一命令；无字面量残留 | 无 | 低 |
| AC-6 | 扩展安全与元数据校正 | `pinned_repo` 统一走 `create_image_node`（转义）；修正 docstring 的模块名与指令数（4 条） | 含特殊字符的 `link`/`url` 输出被转义；docstring 与注册表一致 | 无 | 低 |
| AC-7 | mermaid 配套决策 | 恢复 `sphinxcontrib.mermaid` 装配，或删除 `myst_fence_as_directive` 中的 `mermaid` 映射（二选一） | 文档站 mermaid 围栏渲染正常或无残留死映射 | 需人类确认取向 | 低 |
| AC-8 | README 与兼容承诺 | 修正 `static_path` → `html_static_path`、`false` → `False`；`SW_*` 移除绑定具体 release 并写入 CHANGELOG | 示例代码 `ast.parse` 通过；CHANGELOG 有明确失效版本 | 无 | 低 |
| AC-9 | 测试补齐（汇总验收） | 覆盖 `clean` 边界、命令入口统一、HTML 转义、`tree` 输出结构；目标把 `builder.py`/`pinned_repo.py` 覆盖率提升到 ≥70% | 新增用例全部通过；总覆盖率不低于当前 71% 且上述模块达标 | AC-4/AC-5/AC-6 | 低 |
| AC-10（观察项） | `tree` 去重与 `browse` 顺序 | 修正 `tree` 重复打印；`browse` 改为先 `Popen` 后 `open` | `tree` 输出每个目录仅出现一次 | 无 | 低 |

> 说明：AC-10 由 V-2 引出（原 A 阶段未单列），风险最低、收益直观，可作为新人首个提交。

### 8.2 执行顺序（按风险与依赖分批）

**批次 1｜零行为变更（可并行）**
AC-6（扩展元数据与转义）→ AC-8（README 与承诺）→ AC-10（`tree`/`browse` 顺序）→ AC-3（注册去重）

**批次 2｜边界与入口**
AC-5（命令入口统一，含 `thebe` 顺序）→ AC-4（`clean` 边界）

**批次 3｜契约裁决（需人类决策）**
AC-1 与 AC-2（A/B 二选一，落地后同步 README）→ AC-7（mermaid 取向）

**批次 4｜闭环验收**
AC-9（测试补齐与覆盖率达标，作为前三个批次的汇总门）

### 8.3 G4 判定

| 检查项 | 要求 | 实际 | 结论 |
|---|---|---|---|
| 单一职责 | 每项只做一件事 | AC-1~AC-10 均单一 | 通过 |
| 可独立验证 | 有明确验收标准 | 每项均有可执行验收 | 通过 |
| 有 Owner | 需人类指派 | 建议：批次 1 由新人承担，批次 2 由模块维护者，批次 3 由项目负责人裁决 | 待指派 |
| 有验收标准 | 见上表 | 具备 | 通过 |

---

## 九、质量门记录

| 门 | 标准 | 实际 | 结论 |
|---|---|---|---|
| G1 事实门 | ≥20 条客观事实，可验证可追溯 | 31 条，均附路径行号或复现命令 | 通过 |
| G2 洞察门 | ≥3 条四元组 | 6 条，四要素齐备 | 通过 |
| V 门 | 四视角全覆盖，≥5 条意见 | 7 条，含 1 条自我证伪 | 通过 |
| G4 行动门 | 单一职责、可独立验证、有 Owner、有验收 | 10 项，Owner 待人类指派 | 条件通过 |

---

## 十、附录：复现命令与实测输出

### 10.1 元数据与仓库状态

```powershell
$r = '\\10.16.11.3\ai\notebook\dev\libs\AI\projects\xuanspace\libs\mystx'
Select-String -LiteralPath "$r\pyproject.toml" -Pattern '^version|^requires-python|^dependencies'
"COMMITS=$(git -C $r rev-list --count HEAD)"
git -C $r log -6 --oneline
```

输出要点：

```
12: requires-python = ">=3.14"
37: dependencies = []
196: version_file = "src/mystx/_version.py"
COMMITS=69
ffc3902 build(doc): 字体配置改为可选守卫…
```

主题选项与指令计数（脚本统计）：`THEMEOPTS=33`（含非 `options` 键，故正文不引用具体选项数）、`DIRECTIVES=4`。

### 10.2 测试与覆盖率

```bash
ssh ai@10.16.11.3 'cd /media/pc/data/ai/notebook/dev/libs/AI/projects/xuanspace/libs/mystx && \
  source /media/pc/data/lxw/envs/anaconda3a/bin/activate /media/pc/data/lxw/envs/anaconda3a/envs/py314 && \
  python -m pytest -q --cov=mystx'
```

输出要点：`54 passed, 3 warnings in 4.27s`；覆盖率合计 71%（673 语句 / 193 未覆盖），分模块：`config/runtime.py` 54%、`tasks/doc/builder.py` 41%、`ext/.../base.py` 53%、`stats.py` 62%、`pinned_repo.py` 33%。

### 10.3 事实核验纪律

- 磁盘实况以服务器直连为准；Windows SMB 侧元数据存在缓存，可能返回陈旧结果。
- 本报告涉及的核心事实均经过至少一次带输出的命令复核；对话中出现但未保留输出的推断（如 `sphinx.ext.intersphinx` 重复出现）已核验为误判并处剔除。
- 唯一一条被证伪并保留记录的技术假设见 §7.2 V-7。

---

## 十一、执行记录：批次 1（2026-11-23）

### 11.1 范围与结论

按 §8.2「批次 1｜零行为变更」执行 AC-6、AC-8、AC-10、AC-3。其中 AC-3 在执行前证伪，未改代码；其余三项已落地并通过验证。批次 2~4 及 AC-1/AC-2/AC-7 的人类裁决未启动。

| 行动项 | 状态 | 说明 |
|---|---|---|
| AC-6 | 已落地 | `pinned_repo` 改走 `create_image_node(..., link=...)`；`link`/`url`/`alt` 统一 `html.escape`；校正扩展 docstring 模块名与指令数（3→4），并补齐 `github-wakatime` |
| AC-8 | 已落地 | README 两处拼写修正；`SW_MYST_COMPAT_*` 移除版本绑定 `mystx>=0.7` 并写入 CHANGELOG；顺带修正文档站 `extension`→`extensions` 与指令数 |
| AC-10 | 已落地 | `tree` 仅由递归打印目录，消除重复；`dirs.sort()` 稳定顺序；`browse` 改为先 `Popen` 后 `webbrowser.open` |
| AC-3 | 证伪，未改动 | 见 §11.3 |

### 11.2 变更文件

- `src/mystx/ext/github_readme_stats/base.py`：`create_image_node` 新增 `link` 参数并转义。
- `src/mystx/ext/github_readme_stats/pinned_repo.py`：删除手工拼接分支与 `nodes` 导入，统一走基类方法。
- `src/mystx/ext/github_readme_stats/__init__.py`：模块名与指令数 docstring 校正。
- `src/mystx/tasks/doc/builder.py`：`browse` 启动顺序、`tree` 去重。
- `README.md`、`doc/ext/github-readme-stats/index.md`、`CHANGELOG.md`：文档修正与变更记录。

### 11.3 AC-3 证伪记录（对 R-05 的订正）

Sphinx 9.2.0 的 `sphinx.html_themes` 入口点并非独立注册路径：`ThemeLoader._load_entry_point_themes` 对未载入主题注册延迟闭包，闭包只调用 `app.setup_extension(entry_point.module)`。因此入口点 `mystx = "mystx"` 最终仍触发 `mystx.setup()` → `MySTX` → `themes.py:118` 的**唯一一次** `add_html_theme`；当扩展已在 `extensions` 中声明时，注册更早发生，入口点闭包被 `if entry_point.name in self._themes: continue` 跳过。两条路径共享同一注册点，不存在重复注册。若按原 AC-3 删除该调用，则两条路径都拿不到主题，`ThemeLoader.create()` 将抛 `ThemeError: no theme named 'mystx' found`。故 AC-3 的"删除 `setup()` 内重复 `add_html_theme`"前提不成立，本批次不做代码变更；建议将 R-05/AC-3 改写为"主题注册为单点（`themes.py:118`），入口点为延迟触发器"，并补一条 Sphinx 9 路径的回归用例。

### 11.4 验收输出

```bash
ssh ai@10.16.11.3 'cd .../libs/mystx && /media/pc/data/lxw/envs/anaconda3a/envs/py314/bin/python -m pytest tests/unit -q'
# 54 passed, 3 warnings
```

转义断言（服务器 py314 实跑）：

```text
NOLINK: <img src="http://x/?a=1&amp;c=&lt;d&gt;" alt="Pi&lt;ned">
LINK  : <a href="http://l/?x=&quot;y&quot;&amp;z=&lt;w&gt;"><img src="http://x/?q=1" alt="Pinned Repo"></a>
```

无 `link` 时输出结构与旧实现一致，仅新增转义；`compileall` 通过。改动已提交为 `bffec4d`（7 files changed, 28 insertions(+), 19 deletions(-)）。

### 11.5 待确认

- 批次 1 已按建议提交为 `bffec4d`（单提交，单一职责），并随 §14 收尾一并推送至 `origin/main`。
- 批次 2 已执行（AC-5/AC-4 及 AC-9 部分用例），详见 §12。
- 批次 3 需先裁决：AC-1/AC-2（A/B）、AC-7（mermaid 取向）。

---

## 十二、执行记录：批次 2（2026-11-23）

### 12.1 范围与结论

按 §8.2「批次 2｜边界与入口」执行 AC-5、AC-4，并顺带补齐 AC-9 中与二者直接相关的 3 条用例。批次 3（AC-1/AC-2/AC-7 人类裁决）未启动。

| 行动项 | 状态 | 说明 |
|---|---|---|
| AC-5 | 已落地 | 新增 `_core._sphinx_build_cmd()` 作为唯一解析点；`build`/`doctest`/`-M` 目标三入口共用 `$SPHINXBUILD`（缺省 `sphinx-build`）；`thebe_setup` 改为先装配扩展、成功后再写 `launch_buttons.thebe=True`，装配失败即返回，不留悬挂开关 |
| AC-4 | 已落地 | `clean`（builder 模式）以解析出的 `builddir` 为构建根，target 越界即 `Exit` 且不删除；空父目录回收限定在构建根内，不再越过 |
| AC-9（部分） | 部分落地 | 新增 3 条用例：`SPHINXBUILD` 三入口一致性、越界 target 拒绝删除、根内 target 正常删除且保留构建根。HTML 转义与 `tree` 输出结构用例仍待批次 4 补齐 |

### 12.2 变更文件

- `src/mystx/tasks/doc/_core.py`：新增 `_sphinx_build_cmd()`；`_run_target` 改为复用。
- `src/mystx/tasks/doc/builder.py`：`clean` 构建根边界；`doctest`、`build` 去除硬编码可执行名。
- `src/mystx/config/runtime.py`：`thebe_setup` 调整为「先装配、后配置」。
- `tests/unit/test_repo_tasks_entry.py`：新增 3 条回归用例。
- `CHANGELOG.md`：追加批次 2 变更条目。

### 12.3 关键设计取舍

- **边界来源**：构建根取 `_resolve(c, "builddir", builddir, "_build")`，与 `mode='targets'` 的清理范围同源，避免两套推导。
- **严格性**：判定用 `Path.resolve().is_relative_to(buildroot)`，可拦截 `..`、相对路径与绝对路径混写。
- **父目录回收**：循环条件含 `parent != buildroot`，即构建根本身保留；原实现会一路向上删空目录，存在误删项目空目录的风险。
- **thebe 失败语义**：`setup_extension` 抛 `ExtensionError` 时直接 `return`，不再开启按钮，避免 UI 宣称可用但功能缺失。

### 12.4 验收输出

```bash
ssh ai@10.16.11.3 'cd .../libs/mystx && /media/pc/data/lxw/envs/anaconda3a/envs/py314/bin/python -m pytest tests/unit -q'
# 57 passed, 3 warnings（批次 1 后为 54，新增 3 条）
```

`compileall` 通过；`src/mystx/tasks` 内可执行名仅剩 `_core.py:59` 一处字面量（唯一解析点）。改动已提交为 `4ed3e31`（5 files changed, 109 insertions(+), 20 deletions(-)）。

### 12.5 待确认

- 批次 2 已按建议提交为 `4ed3e31`（单提交，单一职责），并随 §14 收尾一并推送至 `origin/main`。
- 批次 4 已执行（转义与 `tree` 用例、覆盖率门槛），详见 §14。
- 批次 3 已裁决：AC-1/AC-2 选 A（声明运行期依赖），AC-7 恢复 `sphinxcontrib.mermaid`，详见 §13。

---

## 十三、裁决记录：AC-1/AC-2 与 AC-7（2026-11-23）

### 13.1 裁决结论

| 议题 | 裁决 | 依据 |
|---|---|---|
| AC-1/AC-2 | **选 A：声明运行期依赖** | `theme.toml` 的 `inherit = "sphinx_book_theme"` 决定 sphinx 与 sphinx-book-theme 均为运行期必需；`dependencies = []` 属打包缺陷。方案 B（顶层惰性导入）仅为「无 Sphinx 也能取版本号」服务，收益不足以抵消额外复杂度，故不采用 |
| AC-7 | **恢复 `sphinxcontrib.mermaid`** | 该包已在 `[project.optional-dependencies].doc` 与 `[dependency-groups].doc` 两处声明，`extensions.py` 原注释「不在依赖中」已过时；同时 `defaults.py` 保有 `myst_fence_as_directive = ["mermaid"]`，不恢复即构成死映射 |

### 13.2 落地内容

- `pyproject.toml`：`dependencies` 改为 `["sphinx>=8.0", "sphinx-book-theme>=1.1"]`；`[doc]` extras 不动。
- `src/mystx/extensions.py`：`DEFAULT_OPTIONAL_EXTENSIONS` 恢复 `"sphinxcontrib.mermaid"`，删除过时注释，保留 `has_module` 守卫。
- `tests/unit/test_fence_directive_consistency.py`（新增）：断言 `myst_fence_as_directive` 中每个围栏都有对应装配扩展，防止死映射回归。
- `CHANGELOG.md`：追加两条变更记录。

### 13.3 验收证据

```bash
pytest tests/unit -q          # 58 passed, 3 warnings（批次 2 后为 57，新增 1 条）
python -m pip check | grep -iE "mystx|sphinx"   # 无输出，无缺依赖
```

解析后的声明：`['sphinx>=8.0', 'sphinx-book-theme>=1.1']`；环境实测 `sphinx 9.2.0`、`sphinx-book-theme 1.4.0`，均满足下限；`sphinxcontrib.mermaid` 已安装且声明 parallel 安全，`test_doc_parallel_extensions_p1.py` 三项全绿。

### 13.4 遗留说明

- 裁决落地已提交为 `61e1925`（3 files changed），已推送至 `origin/main`。
- ~~当前 py314 环境为可编辑安装，`importlib.metadata.requires("mystx")` 仍显示旧元数据（仅 extras）。~~ 已重装刷新：`mystx 0.3.8.dev65+gb04772416`，核心依赖显示为 `['sphinx>=8.0', 'sphinx-book-theme>=1.1']`；在 `/tmp` 下 `import mystx` 成功，`pip check` 过滤 mystx/sphinx 无输出。重装前需补装构建期依赖 `setuptools-scm`（已装 10.3.4），并使用 `--no-build-isolation --no-deps` 避免改动既有环境。
- AC-2 因裁决为 A 而不执行；`doc/conf.py` 的 `_has` 与主题回退重复逻辑保持现状，不再纳入本轮。

---

## 十四、执行记录：批次 4（2026-11-23）

### 14.1 范围与结论

补齐 AC-9 剩余用例并达成覆盖率门槛。AC-9 主要验收项全部落地。

| 行动项 | 状态 | 说明 |
|---|---|---|
| HTML 转义用例 | 已落地 | 新增 `tests/unit/test_github_readme_stats.py`：`src`/`alt`/`link` 三处转义、`build_url` 的布尔与空值语义、`github-pinned-repo` 的 URL 拼装与缺参报错，共 5 条 |
| `tree` 输出结构用例 | 已落地 | 断言多层目录每层仅打印一次、文件按字典序、缩进层级正确；另覆盖目标目录缺失的 stderr 分支 |
| 覆盖率门槛 | 已达成 | `tasks/doc/builder.py` 87%，`github_readme_stats/base.py` 与 `pinned_repo.py` 均 100%（原要求 ≥70%） |
| 附带补强 | 已落地 | `browse` 缺目录短路、`clean` 的 `targets`/未知模式/空父目录回收、`-M` 三个包装、`build` 显式 `opts` 与 `nitpick`、`doctest` 显式输出目录 |

### 14.2 变更文件

- `tests/unit/test_github_readme_stats.py`（新增）：5 条扩展用例。
- `tests/unit/test_repo_tasks_entry.py`：新增 6 条任务 CLI 用例（`tree`×2、`browse`、`clean`×3、`-M` 包装、`build`/`doctest` 各 1）。
- `CHANGELOG.md`：新增「测试」小节。

### 14.3 验收输出

```bash
pytest tests/unit -q --cov=mystx.tasks.doc.builder --cov=mystx.ext.github_readme_stats --cov-report=term-missing
# 72 passed, 3 warnings
# builder.py 104 stmts / 87%；base.py 16 / 100%；pinned_repo.py 14 / 100%
```

`builder.py` 当时未覆盖的 14 行为 `browse` 常驻服务循环，该缺口已在 §15 补齐。

### 14.4 待确认

- 批次 4 已提交为 `a793f46`（3 files changed, 225 insertions），已推送至 `origin/main`。

---

## 十五、后续补强：browse 集成测试（2026-11-23）

### 15.1 动机

§14 遗留 `browse` 常驻服务循环 14 行未覆盖，属最易回归且无测试保护的区域。按交付总结建议 A1 补测。

### 15.2 新增用例

| 用例 | 断言要点 |
|---|---|
| `test_browse_serves_built_html_end_to_end` | 真实拉起 `python -m http.server`；从回读的 `index.html` 中取得哨兵字符串；校验命令行参数、`cwd` 指向构建目录、控制台输出 URL；用 `try/finally` 保证子进程终止 |
| `test_browse_skips_occupied_port` | 先占用目标端口，断言实际选择端口严格大于占用端口 |

### 15.3 验收

```bash
pytest tests/unit -q --cov=mystx.tasks.doc.builder --cov-report=term-missing
# 74 passed, 3 warnings
# builder.py 104 stmts / 98%（仅余 105-106 KeyboardInterrupt 分支）
```

### 15.4 状态

改动位于 `tests/unit/test_repo_tasks_entry.py` 与 `CHANGELOG.md`，已提交为 `b047724` 并推送至 `origin/main`。

---

## 十六、后续补强：mermaid 端到端构建用例（2026-11-23）

### 16.1 用例设计

新增 `tests/unit/test_mermaid_build_e2e.py`，共 4 条断言，覆盖三环链路：

| 用例 | 断言 |
|---|---|
| `test_default_optional_extensions_contains_mermaid` | 常量收录 `sphinxcontrib.mermaid`，且 `resolve_extensions(optional=...)` 能解析出该扩展 |
| `test_mermaid_extension_is_assembled` | 构建后 `app.extensions` 含 `sphinxcontrib.mermaid` |
| `test_mermaid_fence_becomes_directive_node` | doctree 中出现 `mermaid` 节点，`code` 保留围栏正文 |
| `test_mermaid_fence_renders_html` | HTML 输出含 `class="mermaid"` 容器 |

构建经 `build_minimal_myst_config({"extensions_optional": ("sphinxcontrib.mermaid",), ...})` 组装 conf，与预设的调用方式一致。

### 16.2 过程发现

1. **`mystx.setup()` 不自动启用可选扩展。** `DEFAULT_OPTIONAL_EXTENSIONS` 只被预设消费；仅写 `extensions = ["mystx"]` 的站点既不会加载 mermaid，也不会应用 `DEFAULT_MYST_CONFIG` 中的 `myst_fence_as_directive`。首次构造 `extensions = ["mystx", "sphinxcontrib.mermaid"]` 时，围栏退化为普通代码块并触发 `Pygments lexer name 'mermaid' is not known` 告警。
2. **同进程多次构建会污染 Sphinx 注册表。** 与既有并行扩展用例同进程运行时出现 `KeyError: 'myst-nb'`（上一实例注册的 filetype 残留）。改用子进程执行构建后稳定通过。
3. **书系主题对最小站点有额外依赖。** `sphinx_book_theme` 的 `add_source_buttons` 在缺少仓库配置时抛 `TypeError`；本用例改用 alabaster 仅验证围栏→指令→HTML 链路，不承担主题集成验证。

### 16.3 验收

```bash
pytest tests/unit -q
# 78 passed, 3 warnings
pytest tests/unit/test_mermaid_build_e2e.py -q
# 4 passed
```

### 16.4 状态与影响

- 现状对本仓库文档站无功能影响：`doc/` 下无 ```` ```mermaid ```` 围栏，且 `doc/conf.py` 未启用 mermaid。
- 若需要「开箱即用」能力，可选二选一：在 `doc/conf.py` 显式加入 `sphinxcontrib.mermaid` 与 `myst_fence_as_directive`；或让 `mystx.setup()` 按 `DEFAULT_OPTIONAL_EXTENSIONS` 自动装配（行为变更，需另行评估）。
- 改动位于新增测试文件与 `CHANGELOG.md`，已提交为 `49d4c66` 并推送至 `origin/main`。

---

*本报告由七概念方法论链路 R→I→F→A→V→C 生成，对应质量门 G1/G2/V/G4 记录见 §9；批次 1 见 §11，批次 2 见 §12，AC-1/AC-2 与 AC-7 裁决见 §13，批次 4 见 §14，browse 补测见 §15，mermaid 端到端补测见 §16。*