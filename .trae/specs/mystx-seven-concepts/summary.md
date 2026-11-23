# 任务执行总结报告：mystx 七概念复盘与修复交付

> **报告元信息**
>
> - **任务名称**：mystx 代码复盘与修复（七概念方法论链路）
> - **报告生成日期**：2026-11-23
> - **任务执行周期**：2026-11-23（单日，按批次推进）
> - **报告版本**：V1.0
> - **代码仓**：`notebook/dev/libs/AI/projects/xuanspace/libs/mystx`
> - **关联复盘报告**：[review.md](review.md)
>
> **归档说明**：2026-11-23 自 `doc/reports/mystx-seven-concepts-20261123/mystx-delivery-summary-20261123.md` 移入 xuanspace 仓 `.trae/specs/mystx-seven-concepts/`，更名 `summary.md`，正文未改。

---

## 第一章 执行概览

### 1.1 任务基本信息

| 项目 | 内容 |
|---|---|
| 任务类型 | 代码复盘 + 缺陷修复 + 测试补强 + 交付发布 |
| 方法论 | 七概念链路 R→I→F→A→V→C，质量门 G1/G2/G3/G4 |
| 执行方式 | 人机协作，按 4 个批次推进，每批次独立验证 |
| 交付形态 | 6 个原子提交 + 1 份复盘报告 + 1 份交付总结 |

### 1.2 核心成果

> 将 mystx 复盘中提出的 10 项行动项（AC-1~AC-10）全部落地或裁决，其中 9 项完成代码/文档修正、1 项经证据证伪后订正复盘结论；测试从 54 条增至 78 条（含 `browse` 与 mermaid 两条端到端链路），目标模块覆盖率由未度量提升至 98%~100%；6 个提交已推送 `origin/main`。

### 1.3 关键数据

| 指标 | 数值 | 说明 |
|---|---|---|
| 行动项闭环 | 10/10 | 9 项落地，1 项证伪并订正结论 |
| 提交数 | 6 | 全部为原子提交，单一职责 |
| 变更规模 | 15 文件，+647 / -43 | 按文件去重；含测试新增 479 行 |
| 单元测试 | 54 → 78（+24） | 全绿，3 条既有告警 |
| 覆盖率 | builder.py 98%；base.py 100%；pinned_repo.py 100% | 门槛 ≥70%；仅余 `KeyboardInterrupt` 分支 |
| 远端状态 | `b047724..49d4c66` 已推送 | 本次交付提交均已同步 |
| 安装元数据 | `mystx 0.3.8.dev65+gb04772416` | 核心依赖已反映 |

### 1.4 主要亮点

1. **证伪优先于修复**：AC-3 原计划「删除重复主题注册」，经阅读 Sphinx 9.2.0 入口点实现后确认入口点本身只是延迟调用 `setup_extension`，重复注册并不存在；若照删会导致两条注册路径同时失效，构建期抛 `ThemeError`。据此撤下该修改并订正复盘结论。
2. **零行为变更的转义修复**：AC-6 将 HTML 拼装收敛到基类唯一入口并统一 `html.escape`，在消除注入风险的同时保证无 `link` 时输出与旧实现逐字符一致。
3. **目标驱动的依赖声明**：从 `theme.toml` 的 `inherit = "sphinx_book_theme"` 反推运行期依赖，修正 `dependencies = []` 这一打包缺陷，而非凭经验罗列。

### 1.5 主要挑战

1. **AC-3 的前提不成立**：需要在「按计划改」与「证伪后不改」之间做判断，且证伪依据必须可复核。
2. **避免范围蔓延**：批次 3 的 AC-1/AC-2 存在 A/B 两条互斥路线，必须先裁决再动手；AC-2 被裁决排除后需克制不去「顺手优化」`doc/conf.py`。
3. **可编辑安装的元数据滞后**：`pyproject.toml` 改动不会即时反映到 `importlib.metadata`，无法用「重装后 import」作为直接证据，需以三重替代证据闭环。

### 1.6 一句话总结

> 以「先取证、再裁决、后动手」的次序完成了一次零回归的复盘修复交付，其最大价值不在改了多少行，而在于阻止了一次基于错误前提的「修复」。

---

## 第二章 任务背景与目标

### 2.1 背景

用户要求以七概念方法论推进 `doc/reports/mystx-seven-concepts-20261123` 下已产出的复盘与计划报告。报告已完成 R（复盘）、I（洞察）、F（发现）与 A（行动项）阶段，产出 AC-1~AC-10 十项行动项与四批次执行计划，但尚未进入 V（验证）与 C（固化）阶段，代码侧无任何改动。

### 2.2 目标拆解

| 批次 | 覆盖行动项 | 目标 | 验收标准 |
|---|---|---|---|
| 批次 1 | AC-6、AC-8、AC-10（AC-3 待核） | 零行为变更修正 | 单测全绿；输出与旧实现等价或仅新增转义 |
| 批次 2 | AC-5、AC-4、AC-9（部分） | 入口统一与边界收紧 | 命令字面量收敛至一处；越界删除被拒；新增用例通过 |
| 批次 3 | AC-1/AC-2、AC-7 | 先裁决再落地 | 裁决有据；依赖声明可由 `theme.toml` 反推 |
| 批次 4 | AC-9（剩余） | 补测并达覆盖率门槛 | 转义与 `tree` 用例落地；目标模块 ≥70% |

### 2.3 约束

| 约束 | 内容 | 应对 |
|---|---|---|
| 环境约束 | 事实核验以服务器 py314 为准，SMB 侧有缓存 | 所有测试与 git 操作经 `ssh ai@10.16.11.3` 执行 |
| 规范约束 | 提交须原子化、单一职责；删除/推送需人类批准 | 每批次独立提交；推送前逐次确认 |
| 兼容约束 | 不引入行为变更，不改公开 API 语义 | 批次 1 限定为等价重构；新增参数带默认值 |

---

## 第三章 执行过程

### 3.1 阶段总览

| 阶段 | 内容 | 产出 |
|---|---|---|
| P1 批次 1 | HTML 转义收敛、文档口径修正、`tree`/`browse` 修正 | 提交 `bffec4d`（7 文件，+28/-19） |
| P2 批次 2 | `SPHINXBUILD` 单点解析、thebe 先装后配、`clean` 边界、部分用例 | 提交 `4ed3e31`（5 文件，+109/-20） |
| P3 裁决 | AC-1/AC-2 与 AC-7 方案选择 | 决策记录（复盘报告 §13） |
| P4 批次 3 落地 | 依赖声明、mermaid 装配恢复、契约测试 | 提交 `61e1925`（3 文件，+31/-3） |
| P5 批次 4 | 转义/`tree`/CLI 用例、覆盖率达标 | 提交 `a793f46`（3 文件，+225） |
| P6 固化 | 报告回写、推送远端、交付总结 | 本报告；`origin/main` 同步 |

### 3.2 批次 1 要点

- `create_image_node` 增加 `link` 参数并对 `src`/`alt`/`href` 统一 `html.escape`；`pinned_repo` 删除手工字符串拼接。
- README 与文档站中 `html_static_path`、`extension`→`extensions`、指令数等口径修正；CHANGELOG 绑定 `SW_MYST_COMPAT_*` 的移除版本为 `mystx>=0.7`。
- `tree` 改为仅由递归打印目录（消除每目录两次输出）并 `dirs.sort()`；`browse` 改为先 `Popen` 后 `webbrowser.open`，避免浏览器先于服务就绪。

### 3.3 批次 2 要点

- 新增唯一解析点 `_sphinx_build_cmd()`（`$SPHINXBUILD` → `sphinx-build`），`build`/`doctest`/`-M` 目标共用。
- thebe 改为「先 `setup_extension`，成功后再写 `launch_buttons.thebe=True`」，杜绝悬挂状态。
- `clean` 以 `builddir` 为界：target 越界直接 `Exit` 且不删除；空父目录回收不越过构建根。

### 3.4 批次 3 要点

- 裁决 A：`dependencies` 声明 `sphinx>=8.0`、`sphinx-book-theme>=1.1`；AC-2 不执行。
- 裁决恢复 `sphinxcontrib.mermaid`，与既有 `myst_fence_as_directive = ["mermaid"]` 配对，消除死映射；新增契约测试防回归。

### 3.5 批次 4 要点

- 新增扩展指令用例 5 条：三处转义、`build_url` 布尔/空值语义、pin URL 拼装、缺参报错。
- 新增任务 CLI 用例 6 条：`tree` 结构与缺目录、`browse` 短路、`clean` 三态、`-M` 包装、`build` 显式 `opts`、`doctest` 输出目录。
- 覆盖率：`builder.py` 98%（含后续 `browse` 集成补测），`base.py`/`pinned_repo.py` 100%。

---

## 第四章 关键决策

### 4.1 决策清单

| # | 决策主题 | 备选 | 结论 |
|---|---|---|---|
| D1 | AC-3 是否按计划删除「重复」主题注册 | 删除 / 撤销 | 撤销并证伪 |
| D2 | AC-1/AC-2 依赖策略 | A 声明运行期依赖 / B 顶层惰性导入 | 选 A |
| D3 | AC-7 mermaid 取向 | 恢复装配 / 删除映射 | 恢复装配 |
| D4 | 提交粒度 | 单提交合并 / 拆分原子提交 | 拆分为 4 个原子提交 |

### 4.2 决策详情

**D1：撤销 AC-3**

- 依据：Sphinx 9.2.0 的 `sphinx.html_themes` 入口点仅延迟调用 `app.setup_extension("mystx")`，最终仍走 `themes.py:118` 唯一一次 `add_html_theme`；「两次注册」是误读。
- 影响：避免了一次会导致 `ThemeError` 的错误修复；复盘报告 §11.3 记录订正过程。
- 事后评估：正确。该决策体现了「修复前先验证缺陷是否真实存在」。

**D2：选 A（声明运行期依赖）**

- 依据：`theme.toml` 的 `inherit = "sphinx_book_theme"` 决定 sphinx 与 sphinx-book-theme 均为运行期必需；`dependencies = []` 属打包缺陷。
- 权衡：方案 B 仅为「无 Sphinx 也能取版本号」服务，需引入 PEP 562 惰性导入与额外测试，收益不足。
- 影响：干净环境 `pip install` 后 `import mystx` 即可用；`[doc]` extras 继续承载文档构建增量。

**D3：恢复 mermaid**

- 依据：`sphinxcontrib-mermaid` 已在 `[project.optional-dependencies].doc` 与 `[dependency-groups].doc` 两处声明，`extensions.py` 原注释「不在依赖中」已过时；不恢复则 `myst_fence_as_directive` 的 mermaid 映射永远无法命中指令。
- 影响：`has_module` 守卫保留，未安装时静默跳过；新增契约测试锁定「围栏必有装配扩展」。

**D4：拆分原子提交**

- 依据：批次 1/2 为修正、批次 3 含依赖与装配变更、批次 4 为纯测试，职责不同；CHANGELOG 随测试批次归档。
- 影响：4 个提交可按需单独回滚。

---

## 第五章 问题与解决

### 5.1 问题总览

| # | 问题 | 严重度 | 处理 |
|---|---|---|---|
| I1 | AC-3 前提不成立 | 🔴 高（会导致构建失败） | 证伪并撤销，订正报告 |
| I2 | 测试构造 `Directive` 时 `state_machine=None` 导致 `AttributeError` | 🟡 中 | 注入带 `reporter.error` 的桩 |
| I3 | 对 `_run_target` 调用形态的断言与实际不符（位置/关键字混用） | 🟢 低 | 读取实现后按实际签名修正断言 |
| I4 | 可编辑安装元数据滞后，无法直接验证「重装后 import」 | 🟡 中 | 以声明解析 + 已装版本满足 + `pip check` 三重证据替代 |

### 5.2 典型问题复盘：I3

- **现象**：新增的 `-M` 包装用例断言 `calls[1][0] == ("linkcheck","doc","_build")`，实际收到 4 元组。
- **根因**：`_target_linkcheck` 以位置参数传入 `opts`，而 `_target_target` 以关键字传入；断言未区分两种调用形态。
- **教训**：对「委托型」代码写断言前必须先读被委托函数的调用点，不能凭签名猜测；同类断言应校验语义（目标名一致）而非精确元组，降低脆性。

### 5.3 问题模式

| 模式 | 次数 | 共性 | 对策 |
|---|---|---|---|
| 基于假设的修复 | 1 | 未核实缺陷即计划改动 | 修复前加「缺陷可复现/可解释」前置门 |
| 测试断言与实现耦合过紧 | 2 | 断言精确调用形态 | 优先断言可观察行为，其次断言调用语义 |

---

## 第六章 资源使用

### 6.1 技术栈

| 技术 | 版本 | 用途 |
|---|---|---|
| Python | 3.14.6（py314） | 全部测试与脚本执行 |
| Sphinx | 9.2.0 | 主题宿主、构建验证 |
| sphinx-book-theme | 1.4.0 | `mystx` 主题 `inherit` 基座 |
| sphinxcontrib-mermaid | 已安装 | 围栏转指令的指令提供方 |
| setuptools-scm | 10.3.4 | 构建期依赖，用于可编辑重装 |
| pytest / pytest-cov | 随环境 | 单测与覆盖率度量 |
| git | 服务器侧 | 原子提交与推送 |

### 6.2 工具与流程

| 工具/手段 | 用途 | 效果 |
|---|---|---|
| `ssh ai@10.16.11.3` | 绕开 SMB 元数据缓存，取磁盘实况 | 关键，避免陈旧结果误判 |
| `pytest -q --cov=...` | 回归 + 覆盖率门禁 | 每批次即时反馈 |
| `compileall` | 语法级兜底 | 低成本高覆盖 |
| 复盘报告回写 | 过程固化 | 决策与证据可追溯 |

### 6.3 资源效率评价

- 未引入任何新依赖；`sphinxcontrib-mermaid` 属「已安装但未装配」的存量资源，本轮将其激活。
- 未新建虚拟环境，节省了构建隔离与安装时间，代价是 I4 所述的证据间接性。

---

## 第七章 协作分析

### 7.1 协作模式

- **模式**：人机串行确认制。AI 负责取证、实现与验证；人类负责方案裁决、提交与推送授权。
- **关键节点**：批次 3 的 A/B 与 mermaid 取向由人类以选择题形式裁决；每次提交与推送均经显式授权。

### 7.2 效能评估

| 维度 | 评分 | 说明 |
|---|---|---|
| 决策效率 | 4/5 | 选项附带事实依据与推荐项，单轮即完成裁决 |
| 信息对称 | 5/5 | 每批次均报告改动、验证输出与遗留项 |
| 风险控制 | 5/5 | 未出现未授权提交/推送；越界删除被代码层拦截 |

---

## 第八章 多维度分析

### 8.1 目标达成度

| 目标项 | 期望 | 实际 | 达成率 | 评价 |
|---|---|---|---|---|
| AC-1~AC-10 闭环 | 10/10 | 10/10 | 100% | ✓ |
| 零行为变更（批次 1） | 输出等价 | 仅新增转义 | 100% | ✓ |
| 覆盖率门槛 | ≥70% | 98%/100%/100% | 100% | ✓ |
| 测试全绿 | 0 fail | 78 passed | 100% | ✓ |
| 原子提交 | 每批次 1 提交 | 4 提交 | 100% | ✓ |

**综合达成度：100%（等级：优秀）**。需说明的是，覆盖率门槛仅覆盖三个指定模块，非全仓口径。

### 8.2 时间效能

本轮未采集精确时间戳，故不做时间偏差量化。按批次观察：批次 1/2/4 均为单轮实现 + 单轮修正；批次 3 因需先裁决而多一轮交互，属预期内开销。

### 8.3 综合效能

| 维度 | 得分(10) | 依据 |
|---|---|---|
| 目标达成 | 10 | 行动项全闭环 |
| 质量水准 | 9 | 零回归；`browse` 常驻循环未覆盖 |
| 风险控制 | 9 | 证伪撤下一次高风险修复；推送前均确认 |
| 知识沉淀 | 9 | 报告含决策、证据与订正过程 |
| 效率 | 8 | 小规模多次返工（测试断言） |

---

## 第九章 经验与方法论

### 9.1 方法论一：缺陷前置核验（Falsify-Before-Fix）

**适用场景**：复盘报告或静态分析给出的「缺陷」尚属推断时。

**步骤**
1. 找到缺陷所依赖的最小事实（入口点、注册路径、调用链）。
2. 阅读第三方实现源码，确认该事实是否成立。
3. 若事实不成立，记录证伪证据并订正原结论；不修改代码。
4. 若成立，再进入修复。

**本次验证**：AC-3 据此被撤销，避免了 `ThemeError`。**局限**：依赖可读源码；对闭源依赖需以运行期取证替代。

### 9.2 方法论二：批次化原子交付

**适用场景**：多行动项、跨模块的复盘修复。

**步骤**
1. 按「零行为变更 → 结构收紧 → 需裁决 → 纯测试」排序。
2. 每批次独立验证（测试 + 语法检查），独立提交。
3. 需裁决项单独成批，裁决后再实现。
4. 报告同步回写，形成「代码—证据—文档」三角。

**效果**：4 个提交均可独立回滚；复盘报告 §11~§14 与提交一一对应。

### 9.3 方法论三：从继承关系反推运行期依赖

**适用场景**：打包元数据与实际运行需求不一致（如 `dependencies = []`）。

**步骤**
1. 读取主题/插件清单的 `inherit`/`extends` 字段。
2. 将基座与宿主列为运行期必需，而非可选。
3. 用「干净环境安装后首个导入」作为验收动作。

**局限**：仅覆盖显式声明的继承关系；动态导入仍需运行期探测。

### 9.4 最佳实践清单

**保持**
- 每批次提交前跑全量单测并保留输出。
- 断言优先校验可观察行为，其次校验调用语义。
- 报告记录「未做什么、为什么不做」。

**改进**
- 覆盖率门槛应扩展到全仓而非三个模块。
- 对委托型代码，先读调用点再写断言。

**避免**
- 在未核实缺陷真实性的情况下按计划直接改代码。
- 用精确元组断言锁定内部调用形态。

---

## 第十章 改进建议与后续行动

### 10.1 建议

**🔴 高优先级**

1. ~~**补齐 `browse` 常驻循环的集成测试**~~（已完成）
   - 落地：`test_browse_serves_built_html_end_to_end` 真实拉起 `http.server` 并回读页面；`test_browse_skips_occupied_port` 覆盖端口自增。
   - 结果：`builder.py` 覆盖率 87% → **98%**，仅余 `KeyboardInterrupt` 分支。

2. ~~**刷新可编辑安装元数据**~~（已完成）
   - 落地：补装构建期依赖 `setuptools-scm 10.3.4` 后，以 `pip install -e . --no-build-isolation --no-deps` 重装。
   - 结果：`importlib.metadata.requires("mystx")` 核心依赖显示为 `['sphinx>=8.0', 'sphinx-book-theme>=1.1']`；`/tmp` 下 `import mystx` 成功；`pip check` 无相关告警；74 条单测复跑全绿。

**🟡 中优先级**

3. ~~**增加 mermaid 端到端构建用例**~~（已完成）
   - 落地：`test_mermaid_build_e2e.py` 经 `build_minimal_myst_config` 装配 mermaid，断言围栏→指令节点→HTML 容器三环；构建置于子进程以隔离 Sphinx 注册表。
   - 结果：4 条用例通过；同时确认 `DEFAULT_OPTIONAL_EXTENSIONS` 可被预设消费。

4. **覆盖率门槛纳入 CI**
   - 问题：本轮覆盖率靠人工命令度量，未形成门禁。
   - 方案：在测试配置中加入 `--cov-fail-under`，按模块或全仓设定阈值。

**🟢 低优先级**

5. **全仓覆盖率基线**
   - 先产出一次全仓覆盖率基线，再决定是否对 `ext`、`config` 施加门槛。

### 10.2 风险预警

| 风险 | 可能性 | 影响 | 预防 |
|---|---|---|---|
| 依赖下限过宽（`sphinx>=8.0`）在旧版 Sphinx 上行为差异 | 中 | 中 | 在 CI 矩阵中加入 8.x 最低版本 |
| 恢复 mermaid 后，未安装该包的用户遇到围栏静默失效 | 低 | 低 | 文档标注需安装 `[doc]` extras |
| `browse` 无测试保护，后续改动易引入回归 | 中 | 中 | 见建议 1 |

### 10.3 行动跟踪

| # | 行动 | 状态 |
|---|---|---|
| A1 | `browse` 集成测试 | ✅ 已完成 |
| A2 | 重装刷新元数据 | ✅ 已完成 |
| A3 | mermaid 端到端构建用例 | ✅ 已完成 |
| A4 | 覆盖率 CI 门禁 | ⬜ 待办 |

---

## 附录 A 提交清单

| 提交 | 信息 | 规模 |
|---|---|---|
| `bffec4d` | `fix(ext,doc,tasks): 批次1 零行为变更修正` | 7 文件，+28/-19 |
| `4ed3e31` | `fix(tasks,config): 批次2 入口统一与 clean 边界` | 5 文件，+109/-20 |
| `61e1925` | `fix(pkg,ext): 声明运行期依赖并恢复 mermaid 装配` | 3 文件，+31/-3 |
| `a793f46` | `test(doc,ext): 批次4 补齐转义与 tree 用例并达覆盖率门槛` | 3 文件，+225 |
| `b047724` | `test(doc): 补齐 browse 端到端与端口自增用例` | 2 文件，+106/-1 |
| `49d4c66` | `test(ext): 补充 mermaid 围栏端到端构建用例` | 2 文件，+148 |
| **合计** | `ffc3902..49d4c66` | **15 文件（去重），+647/-43** |

## 附录 B 文件清单

| 文件 | 变更 | 要点 |
|---|---|---|
| `pyproject.toml` | 修改 | 声明 `sphinx>=8.0`、`sphinx-book-theme>=1.1` |
| `src/mystx/extensions.py` | 修改 | 恢复 `sphinxcontrib.mermaid` |
| `src/mystx/ext/github_readme_stats/base.py` | 修改 | `link` 参数 + 统一转义 |
| `src/mystx/ext/github_readme_stats/pinned_repo.py` | 修改 | 删除手工拼接，改走基类 |
| `src/mystx/ext/github_readme_stats/__init__.py` | 修改 | 导出与注册对齐 |
| `src/mystx/tasks/doc/_core.py` | 修改 | `SPHINXBUILD` 唯一解析点 |
| `src/mystx/tasks/doc/builder.py` | 修改 | `tree`/`browse`/`clean`/构建命令 |
| `src/mystx/config/runtime.py` | 修改 | thebe 先装后配 |
| `README.md`、`doc/ext/github-readme-stats/index.md` | 修改 | 文档口径修正 |
| `CHANGELOG.md` | 修改 | 变更与测试记录 |
| `tests/unit/test_fence_directive_consistency.py` | 新增 | 围栏—扩展配对契约 |
| `tests/unit/test_github_readme_stats.py` | 新增 | 转义与 URL 拼装 |
| `tests/unit/test_repo_tasks_entry.py` | 修改 | `tree`/`clean`/`browse`/`-M` 用例 |
| `tests/unit/test_mermaid_build_e2e.py` | 新增 | mermaid 围栏端到端构建 |

## 附录 C 验证命令与输出

```bash
source /media/pc/data/lxw/envs/anaconda3a/bin/activate /media/pc/data/lxw/envs/anaconda3a/envs/py314
python -m pytest tests/unit -q --cov=mystx.tasks.doc.builder \
  --cov=mystx.ext.github_readme_stats --cov-report=term-missing
# 74 passed, 3 warnings
# builder.py 104 stmts / 98%（仅余 KeyboardInterrupt 分支）；base.py 16 / 100%；pinned_repo.py 14 / 100%

python -m pip check | grep -iE "mystx|sphinx"   # 无输出
git push origin HEAD                             # ffc3902..a793f46

# A2 元数据刷新
python -m pip install setuptools-scm            # 构建期依赖（10.3.4）
python -m pip install -e . --no-build-isolation --no-deps
cd /tmp && python -c "from importlib.metadata import requires; print(requires('mystx'))"
# ['sphinx>=8.0', 'sphinx-book-theme>=1.1', ...extras]
```

---

> **报告结束**
>
> 本报告依据本次会话中的实际命令输出、提交记录与复盘报告 §11~§14 生成；未采集精确耗时，相关章节仅作定性描述。
