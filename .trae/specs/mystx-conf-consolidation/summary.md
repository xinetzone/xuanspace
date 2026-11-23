# mystx 站点配置收敛复盘（conf.py 单文件化 → 细节下沉裁决）

- 日期：2026-11-23
- 仓库：`libs/mystx`（`github.com:xinetzone/mystx.git`）
- 提交区间：`5bce411..1035e71`（三个提交）
- 方法论：七概念（R-I-E-F-A-V-C），场景为「重构优化」

---

## 一、起点与目标

前一轮已把站点**纯取值**迁入 `doc/_config.toml`，`doc/_conf/` 六个分组只剩机制代码，`conf.py` 仅 45 行导入。随后两个问题依次被提出：

1. `_conf` 包能否删除、机制合回单文件？（→ 方案 B，已执行）
2. 合回后的 `conf.py` 中，哪些细节还能下沉到 `src/mystx` 包内？（→ 本轮裁决与实施）

---

## 二、R｜事实基线（关键 8 条）

| # | 事实 |
|---|---|
| 1 | 合回后 `conf.py` 153 行，分 6 段：i18n、扩展装配、主题与字体、MyST/告警、交叉引用、`setup()` |
| 2 | 包内早已存在 `extensions.resolve_extensions()`、`DEFAULT_OPTIONAL_EXTENSIONS`（10 项）、`DEFAULT_CONDITIONAL`，且已在 `__init__` 导出 |
| 3 | 三者使用者只有 `presets/minimal_myst.py` 与 3 个测试——**仓库自己的文档站反而手写等价逻辑** |
| 4 | `conf.py` 原第 87 行再次追加 `sphinx.ext.intersphinx`，与第 46–51 行重复（Sphinx 幂等吞掉，但清单冗余） |
| 5 | `DEFAULT_BUILD_CONFIG["suppress_warnings"]`（7 条 toc 类）与站点 9 条只重叠 3 条，不能直接替换 |
| 6 | 站点 sitemap 用 `resolve_site_url(require_on_ci=True)`：CI 缺 `SITEMAP_URL_BASE` 即抛错 |
| 7 | `DEFAULT_CONDITIONAL` 的 predicate 是 `sitemap_enabled_by_env()`：只看 `GITHUB_ACTIONS` 是否存在，**不校验变量** |
| 8 | `language` / `locale_dirs` / `gettext_compact` 因 Sphinx 在 `config-inited` 之前初始化 i18n，不可移出 Python |

---

## 三、I｜洞察

- **I-1 库能力闲置构成第三种「双实现」**：探测、去重、条件装配各有两份实现，与本仓此前 4 处「双源漂移」缺陷同型。
- **I-2 「下沉」≠「隐藏」**：`extensions` 必须在模块层就位，任何下沉只能是「返回 list/dict 的纯函数」；整块 `globals().update(...)` 会把顺序与条件分支藏进包内，与刚确立的「conf.py 是唯一可读真相」相悖。
- **I-3 唯判据是「是否跨站同形」**：跨站同形 → 包；含项目事实/生态选择 → 留；下沉后看不出启用了什么 → 不下沉。

---

## 四、F｜第一性原理裁决

不可下沉的五类（写进 `conf.py` 头部 docstring）：

1. **加载时机**：`extensions`、`language`、`locale_dirs`、`gettext_compact`；
2. **条件装配的站点语义**：站点 sitemap 的严格门禁与 `DEFAULT_CONDITIONAL` 不同，合并会削弱 CI 校验；
3. **项目事实**：`autoapi_dirs`、`bibtex_bibfiles`；
4. **生态选择**：`intersphinx_mapping` 6 条、`extlinks` 2 条（跨站共用未取证，暂不下沉）；
5. **副作用钩子**：`setup()` 已是最小实现，再包一层属过度抽象。

---

## 五、A｜实施方案与结果

| 提交 | 内容 | 结果 |
|---|---|---|
| `d479a44` | 删除 `doc/_conf/`（7 文件），机制合回 `conf.py` | `conf.py` 153 行；`_config.toml` 键值一字未改 |
| `bd690e1` | 扩展装配改调 `resolve_extensions`；`suppress_warnings` 迁入 `DEFAULT_SITE_CONFIG` | `conf.py` 153 → 141 行；消除手写探测循环、9 处 `append` 与重复 intersphinx |
| `1035e71` | 站点剔除 `sphinxcontrib.mermaid`（保留于包内清单） | 产物体积回落至基线以下；包能力不受影响 |

### 5.1 sitemap 保持严格判定

站点不采用 `conditional=DEFAULT_CONDITIONAL`，而是：

```python
site_url = resolve_site_url(require_on_ci=True)
extensions = resolve_extensions(
    required=("mystx",),
    optional=_SITE_OPTIONAL_EXTENSIONS,
    conditional=(("sphinx_sitemap", lambda: bool(site_url)),),
)
```

原因：`DEFAULT_CONDITIONAL` 只看 `GITHUB_ACTIONS` 是否存在，会放过「CI 未声明 `SITEMAP_URL_BASE`」的情形，把 `127.0.0.1` 写进线上 sitemap。

### 5.2 告警抑制双键并存

- `DEFAULT_SITE_CONFIG["suppress_warnings"]`：8 条 myst-nb / autoapi / docutils 上游噪声，站点取 `*` 展开后追加 1 条 `epub.unknown_project_files`（源起 `_static/switcher.json`）；
- `DEFAULT_BUILD_CONFIG["suppress_warnings"]`：preset 的 7 条 toc/etoc 类，保持不变。
- 两者**不合并**——合并会改变 preset 行为。这是有意的「同名不同域」，已在两处注释说明。

### 5.3 mermaid 的取舍

- 包内 `DEFAULT_OPTIONAL_EXTENSIONS` 仍含 `sphinxcontrib.mermaid`（其他站点可用）；
- 实测代价：向**每一页**注入约 358 行内联引导脚本 + 运行时从 `cdn.jsdelivr.net` 拉 ESM；`search.html` 602 → 960 行，全站 +568 KB；
- 本站正文无任何 mermaid 图表，故站点侧与 `sphinx_sitemap` 一并从包内清单剔除；
- 需要图表时的放行路径：站点侧取消剔除，或 `mermaid_use_local = True` 并把 `mermaid.min.js` 内置到 `_static`。

---

## 六、V｜对抗审查要点

1. **顺序语义外移**：扩展加载顺序对 `myst_parser` / `myst_nb` 互斥有语义，下沉后需跳进包确认——已用「列表顺序即加载顺序」注释 + `required=("mystx",)` 固定首位缓解。
2. **`required` 的陷阱**：`resolve_extensions` 默认 `required=("myst_parser",)`；若照搬，会让站点从「自动选 myst_nb」变成「强制 myst_parser」，故显式传 `required=("mystx",)`。
3. **缺包静默降级**：原先无守卫的 `sphinxcontrib.bibtex` / `autoapi` / `pyscript` 等改走 `has_module`，缺依赖时由硬失败变为静默跳过——依赖由 `.[doc]` 额外项兜底。
4. **同名双键**：`suppress_warnings` 在两份 defaults 中并存，存在误合并风险——已在注释中写明不得合并的理由。
5. **父仓指针**：`libs/mystx` 子模块在 xuanspace 父仓仍显示 `M`，未随本次推送更新。

---

## 七、C｜验收记录

| 项目 | 结果 |
|---|---|
| `ruff check doc src/mystx` | 通过，零告警 |
| `pytest tests/unit` | **119 passed** |
| `sphinx-build -b html` | 退出码 0，日志 **0 WARNING** |
| 扩展清单 | 无重复项（原重复的 `sphinx.ext.intersphinx` 已消除） |
| 产物体积 | 22 620 KB（含 mermaid 时 23 204 KB，基线 22 636 KB） |
| 逐页比对 | 74 个 HTML 对基线（过滤 tippy 哈希、时间戳、`hashed/`、`documentation_options.js?v=`）：**唯一差异**为 `_modules/mystx/config/defaults.html`（viewcode 反映新增键） |
| 远程 | `5bce411..1035e71` 已推送 `origin/main` |

### 补充取证：`genindex` / `py-modindex` 的差异

首轮比对曾出现 `mystx._version` 条目差异。取证结论：`src/mystx/_version.py` 由 setuptools-scm 本地生成、被 `.gitignore` 忽略、从未入库，仅本地 autoapi 扫描收录；干净检出与 CI 不产生该差异。另 `_apply_top_level` 条目差异来自更早一轮 `config/runtime.py` 重构，与本次无关。

---

## 八、遗留事项

1. **父仓子模块指针**：`xuanspace` 中 `libs/mystx` 仍为未提交的 `M`，是否随下次推送更新待定；
2. **覆盖率门禁**：现为 85%，再上调需先补 `wakatime`（25%）、`stats` / `top_langs`（62%）、`tasks/doc/_core`（76%）；
3. **TOML 未注册键**：目前仅告警但仍写入，属有意宽容，未升级为错误；
4. **生态地址表**：`intersphinx_mapping` / `extlinks` 是否下沉为包内常量，需先确认 daoapps / xuanspace 等同族站点是否共用同一组地址。
