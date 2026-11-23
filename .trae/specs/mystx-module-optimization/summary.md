# mystx 模块优化总结（2026-11-23）

> 范围：`libs/mystx` 仓库内 7 个模块的结构优化，共 8 个提交（`876ce69`…`9dace2b`）。
> 全部改动均以「配置/产物等价 + 测试全绿」为验收，不改变对外行为。

## 1. 背景与目标

前置交付（`49d4c66` 及其之前）完成了缺陷修复、依赖声明、覆盖率门禁与文档原子化。本轮目标转为**结构治理**：消除平行实现、重复名单与顺序耦合，使每个决策在包内只有一个实现点。

## 2. 事实基线（R）

| 模块 | 优化前 | 优化后 | 降幅 |
|---|---|---|---|
| `doc/_conf`（含 `conf.py`） | 357 行 | 251 行 | −30% |
| `tasks/doc/builder.py` | 224 行 | 192 行（另拆 `_serve.py` 35 行） | −14% |
| `config/runtime.py` | 185 行 | 110 行 | −41% |
| `ext/gallery.py` | 145 行 | 132 行 | −9% |
| `myst_compat.py` | 133 行 | 113 行 | −15% |
| `myst_docs.py` | 120 行 | 110 行 | −8% |
| `themes.py` + `version_switcher.py` | 211 行 | 169 行 | −20% |

测试：110 → **117 条**，全部通过；覆盖率门禁（≥70%）保持。

## 3. 洞察（I）：三类同源问题

1. **中转层的假权威**：`config.defaults` 转发 `BOOK_THEME_NAMES`，真正的权威 `themes.py` 被遮蔽，长期 `F401` 告警正是错位信号。
2. **名单与顺序的复制**：`myst_docs` 的指令名常量与注册字典各写一遍，一致性靠人保证；`_conf` 的 sitemap 分支与包内 `resolve_site_url` 语义分叉。
3. **决策被拆成多段表达**：`resolve_theme` 先删 alabaster 再末尾补齐；`sphinx_setup` 三处读取同一环境变量；`gallery` 的「正文/选项」边界藏在 40 行循环里。

## 4. 第一性原理裁决（F）

| 议题 | 裁决 | 依据 |
|---|---|---|
| `_resolve_html_baseurl` 与 `utils.resolve_site_url` 是否合并 | **不合并** | 预置站点允许 `site_url` 兜底 CI；站点侧要求 CI 显式声明，合并即功能回退 |
| `SW_MYST_COMPAT_*` 旧别名是否删除 | **保留，收敛实现** | 当前 `0.3.8.dev*` 未到契约的 0.7；把删除条件写入模块文档 |
| 是否新增 `site` 配置 API | **不新增** | 包内已有三套入口，再加只会形成第四套 |
| 站点专属值（`README.md` 排除、`utterances.repo` 等）是否入包 | **不入包** | 会令包被单一站点污染；只提炼机制 |

## 5. 落地清单（A）

| 提交 | 内容 |
|---|---|
| `876ce69` | `utils.resolve_site_url()` 单一机制点；修正原子化引入的**RTD 构建 ImportError 回归**；`_conf` 复用包内 `resolve_theme` / `has_module` / 默认字典 |
| `f7a3524` | `_conf` 瘦身：删 thebe 死配置、`DOC_DIR`、横幅与编号 |
| `4108678` | `tasks/doc`：端口/子进程/浏览器迁入 `_serve.serve()`；`_builder_tokens()` 与 `_m_paths()` 收敛 5 处重复 |
| `cbdd2e0` | `ext/gallery`：`_card_body()` / `_as_options()` 两纯函数，修复尾随空格 |
| `ad5d1fc` | `myst_compat`：`_read_switch()` 单点解析，弃用文案常量绑定 0.7 |
| `2305c12` | `myst_docs`：`_DIRECTIVE_SPEC` 单一来源；先解析后写入，不留半注册 |
| `9dace2b` | `themes`：候选队列一次拼装 + `next()`；`version_switcher`：`on_rtd` / `on_ci` 单次判定 |

## 6. 对抗审查（V）要点

- **魔鬼**：合并 `html_baseurl` 会让「CI 无 `SITEMAP_URL_BASE` 但有 `site_url`」的站点从可用变报错 —— 故不合并。
- **新人**：`_card_body` 就地 `pop` 有副作用 —— docstring 首句说明，测试断言「只消费三键」。
- **老板**：行数不是目标 —— 每步以 34 项配置逐键 diff、`sitemap.xml` 产出、页面 DOM 对比为验收。
- **未来**：0.7 时删别名只需删 `_read_switch` 中一项；`PARALLEL_SAFE_OVERRIDES` 命名空表保留为唯一填表入口。
- **数据**：重写整文件风险 —— 全部以既有测试 + 新增契约测试 + 构建退出码 0 双重锁定。

## 7. 验证证据（C）

| 证据 | 结果 |
|---|---|
| `pytest tests/unit` | 117 passed |
| `ruff check` | 全通过（并消除既有 `F401`、`W292` 待办） |
| `sphinx -b html doc` | 退出码 0；RTD / GitHub Actions / 本地三环境矩阵均通过 |
| Pages 产线 | `sitemap.xml` 仍在，`<loc>` 前缀 `https://xinetzone.github.io/mystx/` |
| 页面级对比 | gallery 三页、`myst-example` 页仅时间戳与 tippy 哈希不同 |
| 推送 | `49d4c66..9dace2b`，14 个提交，43 文件，+1538/−971 |

## 8. 方法论沉淀

1. **同源核查先问「谁是权威」**，再问「是否重复」：`F401` 这类告警往往指向权威错位而非多余导入。
2. **拆分即引入回归风险**：`conf.py` 原子化把「缺失键」升级为 `ImportError`，说明重构后必须跑真实产线矩阵，而非仅单测。
3. **契约测试要断言行为，不断言文本**：`test_version_switcher` 由搜索字面量改为加载模块断言属性，拆分后自动免疫。
4. **不合并也是一种裁决**：语义分叉处强行去重会以牺牲产线行为为代价，保留并写清差异更诚实。

## 9. 后续可选项

- `doc/_conf` 与包内 `DEFAULT_*` 的键级契约测试（防止默认字典改名后站点静默失配）。
- 覆盖率门禁由 70% 逐步上调至 85%。
- `myst_compat` 在 `mystx>=0.7` 时删除 `SW_*` 别名（已写入模块文档与 CHANGELOG）。
