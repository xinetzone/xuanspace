# mystx presets 对标 Sphinx Config 的全面优化与瘦身 Spec

* 变更 ID：`slim-mystx-presets`
* 场景：**重构优化**（七概念链路 I → F → A → V → C）
* 对象：`libs/mystx/src/mystx/presets/`（5 个模块，721 行）
* 参照物：上游 `notebook/libs/sphinx-doc/sphinx/sphinx/config.py`（914 行，Sphinx 9.3.0 开发版）
* 日期：2026-11-24

## Why

`presets` 已在上一轮（`bd53ebf`）把站点配置改为 `SITE_CONFIG_SPEC` 单源声明，但声明形状与上游 `_Opt` 同构而不同形：19 个键占 116 行、`_constant` 包装器与值/工厂两套写法并存、同一契约在运行期被校验两次（其中一次分支不可达）、极简预设对同一键重复赋值三次。结果是「单源」已达成，「可读 + 可维护 + 零冗余」未达成。

同时，上游 `config.py` 提供了可直接对标的成熟形状（单行声明表、值或可调用默认值、键集实例化、元数据四字段），本轮以其为基准做一次有据可依的瘦身，而非按行数削足适履。

## What Changes

* `ConfigSpec` 的默认值字段改为 **「值或工厂」两态**，与上游 `_Opt.default` 同形；删除 `_constant()` 包装器。
* `SITE_CONFIG_SPEC` 由 19 条 × 5 行改为 **19 条单行声明**，键序保持不变。
* 键集一致性校验（`_assert_spec_contract`）改为**导入期执行一次**；删除 `apply_config` 中由 spec 派生因而不可达的 `undeclared` 分支。
* `_BuildContext.gettext_compact` 类型与 SPEC 声明对齐（`bool | str`）。
* 极简预设删除 3 处与 `params` 同值的重复写入，收敛 `params` 构造。
* 站点画像（`mystx_docs`）把「逐键 `deepcopy`」收敛为「取值表整体构造后一次深拷贝」，消除新增字典键漏拷贝的风险。
* `_shared.build_project_meta` 的默认值合并收敛为一次字典合并。
* 模块 docstring 明确元数据（`rebuild` / `types` / `description`）的**当前消费状态**。
* 新增 3 条防回归测试：可变默认值不共享、二次装配隔离、键序稳定。
* **无破坏性变更**：公开门面、函数签名、19 个配置键的取值与键序全部不变。

## Impact

* 受影响能力：`mystx.presets` 的声明层、装配层、站点画像层。
* 受影响代码：
  * `src/mystx/presets/mystx_site.py`（368 → 目标 ≤275 行）
  * `src/mystx/presets/minimal_myst.py`（50 → 目标 ≤45 行）
  * `src/mystx/presets/mystx_docs.py`（202 → 目标 ≤190 行）
  * `src/mystx/presets/_shared.py`（55 → 目标 ≤52 行）
  * `tests/unit/test_presets.py`（新增 3 条用例）
* 不受影响：`doc/conf.py`（15 行）、`doc/_config.toml`、`config/defaults.py` 的键值、`extensions.py` 的装配语义。
* 参照物只读，不修改上游 `sphinx-doc` 仓库。

***

## R｜事实基线（G1：26 条，可追溯）

> 前 20 条为上游 `config.py` 源码事实（行号对应该文件），后 6 条为 mystx 侧实测事实。

| #  | 客观事实                                                                                                         | 证据                                |
| -- | ------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| 1  | 全文 914 行，公开构件为 `ConfigValue`、`ENUM`、`_Opt`、`Config` 与模块级 `setup()`                                           | `sphinx/config.py:1-914`          |
| 2  | 失效档类型为 7 值 `Literal`：`''`、`env`、`epub`、`gettext`、`html`、`applehelp`、`devhelp`                                | `sphinx/config.py:28-38`          |
| 3  | `ConfigValue` 是 `(name, value, rebuild)` 的 `NamedTuple`                                                      | `sphinx/config.py:44-48`          |
| 4  | `_Opt` 以 `__slots__` 固定四字段，并拒绝赋值与删除                                                                          | `sphinx/config.py:97-181`         |
| 5  | `_Opt.__repr__` 的 `valid_types` 位置实际输出 `self.rebuild`                                                        | `sphinx/config.py:125-132`        |
| 6  | 内置配置以 `Config.config_values: dict[str, _Opt]` 类属性集中声明                                                        | `sphinx/config.py:215-297`        |
| 7  | 声明条目为单行调用：`_Opt(default, rebuild, frozenset((types,)), description?)`                                        | `sphinx/config.py:217-296`        |
| 8  | 默认值允许 `callable(config)`：`copyright` 与 `root_doc` 两条                                                         | `sphinx/config.py:220-224`、`240`  |
| 9  | 实例初始化时把类属性表复制为 `self._options`                                                                               | `sphinx/config.py:306`            |
| 10 | 点号 override 被展开为嵌套字典键                                                                                        | `sphinx/config.py:309-312`        |
| 11 | `extensions` override 接受逗号分隔字符串                                                                              | `sphinx/config.py:316-322`        |
| 12 | `Config.read()` 在 confdir 缺 `conf.py` 时抛 `ConfigError`                                                       | `sphinx/config.py:339-352`        |
| 13 | `convert_overrides` 按 `valid_types` 与默认值类型分派：bool 取 `'0'/'1'`、list 逗号切分、dict 禁止整体覆盖、int 解析                   | `sphinx/config.py:354-400`        |
| 14 | 未注册的 override 名只产生 warning                                                                                   | `sphinx/config.py:417-422`        |
| 15 | `__setattr__` 维护 `master_doc`/`root_doc` 与 `copyright`/`project_copyright` 双向别名                              | `sphinx/config.py:434-444`        |
| 16 | `__getattr__` 解析顺序为 override → conf.py 原值 → 默认值，结果写入 `__dict__` 缓存；可调用默认值不写缓存                                | `sphinx/config.py:446-476`        |
| 17 | `add()` 对重复名抛 `ExtensionError`，并把 bool 型 rebuild 归一化为 `'env'` 或 `''`                                         | `sphinx/config.py:494-511`        |
| 18 | `filter(rebuild)` 接受单类别字符串或类别集合                                                                              | `sphinx/config.py:513-516`        |
| 19 | pickle 时剔除不可序列化值，仅当该键 `rebuild` 非空才告警                                                                        | `sphinx/config.py:518-553`        |
| 20 | `check_confval_types` 只发 warning（`once=True`），并做 `list→tuple`、`→frozenset` 强制转换                              | `sphinx/config.py:777-851`        |
| 21 | mystx `presets` 共 5 个模块 721 行：`__init__` 46、`_shared` 55、`minimal_myst` 50、`mystx_docs` 202、`mystx_site` 368 | 实测 `Get-Content` 计数（2026-11-24）   |
| 22 | `ConfigSpec` 四字段中只有 `factory` 被生产代码消费；`rebuild`/`types`/`description` 仅被 `tests/unit/test_presets.py` 读取     | 全仓 `grep` 结果                      |
| 23 | `SITE_CONFIG_SPEC` 共 19 键、116 行；其中 4 键经 `_constant()` 包装，15 键为 lambda                                        | `mystx_site.py:130-245`           |
| 24 | `apply_config` 的 `undeclared` 分支不可达：`values` 的键由 `SITE_CONFIG_SPEC` 生成                                       | `mystx_site.py:347-349`、`363-366` |
| 25 | `_assert_spec_contract()` 在每次 `build_config()` 调用时执行                                                         | `mystx_site.py:346`               |
| 26 | 极简预设在 `build_project_meta(params)` 之后重复写入 `extensions`/`html_theme`/`html_theme_options` 三个同值键               | `minimal_myst.py:35-41`           |

补充基线（不计入门槛）：

* 测试基线：`pytest tests/unit` → **131 passed, 3 skipped**；`ruff check src/mystx/presets` → 通过（2026-11-24 实测）。
* 消费面：`doc/conf.py`、`tests/unit/test_presets.py`、`test_conf_py.py`、`test_mermaid_build_e2e.py`、`CHANGELOG.md`。
* 19 键中 12 键取值直接取自 `DEFAULT_EXT_CONFIG` / `DEFAULT_BUILD_CONFIG` / `DEFAULT_SITE_CONFIG` / `DEFAULT_MYST_CONFIG` 的既有条目，7 键由入参或环境派生。

## I｜洞察（G2：5 条四元组）

**I-1 同构不同形的声明表**

* 现象：上游 81 行声明 60+ 键（单行一条），mystx 116 行声明 19 键（每条 5 行）。
* 根因：实现时把「四字段完整表达」理解为「四字段分行书写」，并以 `_constant()` 补齐「静态值没有工厂」的形状缺口。
* 影响：`SITE_CONFIG_SPEC` 占文件 32% 篇幅，键与值的对应关系被格式噪声稀释；新增键成本高，易诱发复制粘贴错误。
* 建议：默认值字段允许「值或工厂」两态（与上游 `_Opt.default` 同形），条目单行化，删除 `_constant()`。

**I-2 对自行生成的数据再做运行期校验**

* 现象：`build_config` 由 spec 派生键，`apply_config` 随后又检查「产出键是否越界」，该分支永假。
* 根因：把「调用方输入校验」与「内部生成数据自检」混为一谈。
* 影响：每次装配多一次 O(n) 集合差；死分支向读者暗示「存在越界可能」，误导后续维护。
* 建议：键集一致性只在导入期校验一次（漂移即 ImportError），删除 `apply_config` 的不可达分支。

**I-3 元数据有声明、无消费者、无状态说明**

* 现象：`rebuild`/`types`/`description` 三字段无生产消费者，但 docstring 只说「供扩展化复用」。
* 根因：为未来 `add_config_value` 扩展化预留接口，未同时声明当前有效边界。
* 影响：维护者无法判断题面：是「已生效的失效档」还是「纯文档」；误判会导致对增量构建行为的错误预期。
* 建议：保留字段（与上游同形是唯一失效档落点），在类与模块 docstring 中明确「声明元数据，运行期不参与」，并由测试锁定合法性。

**I-4 迁移残留造成三写一**

* 现象：`minimal_myst` 在 `params` 已赋值后，又对 `project_meta` 写入三个同值键。
* 根因：引入 `build_project_meta` 白名单机制后未删除旧的三行赋值。
* 影响：默认值语义存在三个书写点，改一处漏两处即产生静默不一致。
* 建议：删去重复赋值，`project_meta` 只承担「白名单过滤 + 派生态补齐」。

**I-5 取值表与隔离策略混写**

* 现象：站点画像 30 键内联于 `namespace.update({...})`，其中 3 个字典用逐键 `deepcopy` 防共享。
* 根因：把「取值」与「隔离」两件事写在同一处，隔离依赖人工逐键判断。
* 影响：新增字典型键时漏写 `deepcopy`，会让同进程二次装配看到上一次的副作用。
* 建议：取值表整体构造后一次深拷贝，隔离由结构保证而非逐键自觉。

## E｜萃取（G3）

**模式：声明表—类型镜像—导入期校验（Declare / Mirror / Verify at import）**

适用场景：包内默认值层要向外部命名空间（`conf.py` 的 `globals()`）批量注入取值时。

步骤：

1. 选定上游同类实现作为**形状基准**，只对齐形状，不复用其私有符号；
2. 用不可变声明类型承载字段（`NamedTuple` ≙ 上游 `_Opt` 的 `__slots__` + 冻结赋值）；
3. 默认值字段允许「值或工厂」两态，与上游 `callable(config)` 语义对应；可变值一律走工厂拷贝；
4. 键集用类型镜像（`TypedDict`）表达，镜像只提供类型契约，不承载取值；
5. 一致性校验放在**导入期**与测试，不放装配热路径；
6. 对「有声明、无消费」的元数据，在 docstring 明写状态，防止被误当生效。

反模式：

1. 同值多处赋值（`params` → `project_meta` 三写一）；
2. 对自行生成的数据再加运行期校验（不可达分支）；
3. 为「未来可能用到」保留字段却不声明其消费状态；
4. 用包装器（`_constant`）填补声明形状缺口，而非让声明类型本身容纳两种形态。

跨场景迁移：任何「包内默认值 → 用户命名空间」的装配层（CLI 默认参数注入、日志配置下发、测试夹具构造）都可套用本模式。

## F｜第一性原理裁决

| 议题                                   | 裁决            | 依据                                                |
| ------------------------------------ | ------------- | ------------------------------------------------- |
| 是否直接复用上游 `_Opt`                      | **不复用**       | 私有 API；其 `__repr__` 已出现字段错位（事实 5），说明内部形状仍会变动      |
| `rebuild`/`types`/`description` 是否删除 | **保留，声明为元数据** | 与上游同形是「失效档位」的唯一落点；删除后下次研究会重新发明同一张表                |
| `SiteConfig` 类型镜像是否保留                | **保留**        | 为 IDE/类型检查提供契约；导入期校验保证镜像不漂移                       |
| 极简预设是否统一为 `(values, setup)`          | **不统一**       | 公开契约与既有测试依赖扁平字典，统一属破坏性变更，收益仅是形式一致                 |
| 站点画像是否迁入 TOML                        | **不迁**        | 已有独立 spec `mystx-config-toml-migration`，本轮只做代码内瘦身 |
| 是否新增公共 API                           | **不新增**       | 门面 4 项已覆盖全部用途，新增只扩大维护面                            |
| 是否以行数作为主验收                           | **否**         | 行数只作辅指标；主验收是 19 键取值/键序等价与测试不减                     |

## V｜对抗审查（V 门：6 条）

1. **魔鬼**：单行化后含中文说明的条目可能超 120 列，诱发折行反而更难读。处置：`ruff` 已 `ignore = ["E501"]`，允许单条超宽；要求「一条一行、不折行」。
2. **新人**：支持字面值后，可能把 `list`/`dict` 直接写进声明表，造成跨次装配共享。处置：docstring 明写「可变值必须用工厂」，并新增测试断言两次装配的列表/字典不是同一对象。
3. **老板**：行数不是目的。处置：验收以「19 键逐键相等 + 公开 API 不变 + 131 测试不减」为主，行数目标（≤620）为辅。
4. **未来**：若真做 `add_config_value` 扩展化，`factory` 形态仍需再改。处置：保留 `_BuildContext` 作为唯一入参类型，扩展化时只需替换消费者，不动声明表。
5. **数据**：导入期校验会把声明漂移从「运行期 RuntimeError」提前为「导入即失败」，可能影响仅导入包的下游。处置：异常信息包含缺失/多余键名，并新增测试覆盖该路径。
6. **顺序**：重排键序可能改变 `values` 的迭代顺序，下游若有顺序敏感行为会静默变化。处置：键序与改动前逐字一致，并新增键序断言。

## ADDED Requirements

### Requirement: 声明表支持「值或工厂」两态

`ConfigSpec` SHALL 接受静态默认值或 `Callable[[_BuildContext], Any]`，并在装配时对可调用值求值；对可变默认值（list/dict/set）SHALL 要求使用工厂以逐次拷贝。

#### Scenario: 静态值条目

* **WHEN** 某键的声明直接给出字符串默认值
* **THEN** 装配结果等于该字符串，且多次装配得到相同的不可变对象

#### Scenario: 可变默认值不共享

* **WHEN** 对同一配置连续调用 `build_config()` 两次
* **THEN** 两次结果中的 `locale_dirs`、`inheritance_graph_attrs`、`nitpick_ignore`、`suppress_warnings` 互不为同一对象，修改其一不影响另一

### Requirement: 键集一致性在导入期校验

键集一致性 SHALL 在模块导入时校验一次；装配路径 SHALL NOT 重复校验由 spec 派生的键集。

#### Scenario: 声明与镜像漂移

* **WHEN** `SITE_CONFIG_SPEC` 与 `SiteConfig` 键集不一致
* **THEN** 导入模块即抛出含缺失键与多余键名的 `RuntimeError`

#### Scenario: 正常装配

* **WHEN** 键集一致时调用 `apply_config(namespace)`
* **THEN** 命名空间获得 19 个键的取值，函数返回 `setup` 钩子，且不执行任何键集差集运算

### Requirement: 装配结果等价

改动前后，19 个配置键的**键序与取值** SHALL 逐键相等（相同入参下）。

#### Scenario: 站点画像装配对比

* **WHEN** 以 `doc/conf.py` 相同入参分别运行改动前与改动后的 `mystx_docs.configure`
* **THEN** 注入命名空间的配置键列表与取值逐项相等

## MODIFIED Requirements

### Requirement: 极简预设装配

`build_minimal_myst_config(override)` SHALL 返回包含 `setup` 的扁平字典；`project_meta` SHALL 只承担白名单过滤与派生态补齐，不得重复写入已由 `params` 提供的同值键。

#### Scenario: 覆盖主题

* **WHEN** 传入 `{"html_theme": "pydata_sphinx_theme"}`
* **THEN** 返回值中的 `html_theme` 为该主题，`html_theme_options` 由 `resolve_theme_options` 补齐，`setup` 可调用

### Requirement: 站点画像注入隔离

`mystx_docs.configure(namespace, root=...)` SHALL 以整体深拷贝方式注入取值表，使同进程二次装配互不影响。

#### Scenario: 二次装配

* **WHEN** 在同一进程中对两个空白命名空间先后调用 `configure`
* **THEN** 两个命名空间中的字典/列表为不同对象，修改前者不影响后者

## REMOVED Requirements

无。本次仅移除内部不可达分支与重复赋值，不删除任何公开能力。

## 验收指标（C 前置）

| 指标            | 目标                                                            |
| ------------- | ------------------------------------------------------------- |
| `presets` 总行数 | 721 → **≤620**（−14%），`SITE_CONFIG_SPEC` 声明块 ≤35 行             |
| 19 键取值/键序     | 与改动前**逐键相等**                                                  |
| 公开 API        | `presets.__all__` 4 项、三个函数签名均不变                               |
| 测试            | `pytest tests/unit` ≥131 passed / 3 skipped；新增用例后 ≥134 passed |
| 静态检查          | `ruff check src/mystx/presets` 零告警                            |
| 文档构建          | `sphinx-build -b html doc <tmp>` 退出码 0，告警数不高于改动前基线            |
| 原子性           | 每项改动一个提交，可单独回滚                                                |

## 非目标

* 不迁移站点画像到 TOML（另属 `mystx-config-toml-migration`）。
* 不修改 `config/defaults.py` 的键值语义，不调整 `extensions.py` 的装配顺序。
* 不引入 `add_config_value` 扩展化、不新增公共 API、不改动公开返回形态。
* 不修改上游 `sphinx-doc` 仓库任何文件。
