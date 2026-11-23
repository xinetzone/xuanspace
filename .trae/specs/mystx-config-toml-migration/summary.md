# mystx 站点配置迁移总结：`_conf` → `_config.toml`（2026-11-23）

> 提交：`5bce411`（mystx 仓，已推送 `59476ee..5bce411`）
> 目标：站点**纯取值**集中到 `doc/_config.toml`，`doc/_conf/` 只保留机制。

## 1. 动因

前一轮已判定「站点专属值不得入包、机制才可复用」。但反向问题仍在：站点专属值散在 7 个 Python 文件里，改一句公告要碰代码，评审噪声大。`_config.toml` 本已由包内 `ConfigManager` 读取（此前仅处理 `html_theme_options`），具备承接能力。

## 2. 迁移边界（裁决）

| 类别 | 去向 | 判据 |
|---|---|---|
| 元信息：`project`/`author`/`copyright` | toml | 纯取值，无计算 |
| 外观与静态资源：`exclude_patterns`/`html_static_path`/`html_css_files`/`html_last_updated_fmt`/`templates_path`/`html_title`/`html_logo`/`html_favicon`/`html_copy_source` | toml | 纯取值 |
| `version_switcher_json_url`、`comments_config`、`ogp_site_url`、`ogp_social_cards` | toml | 站点专属，非机制 |
| `release` / `version` | Python | setuptools-scm 动态计算 |
| `language` / `locale_dirs` / `gettext_compact` | Python | **时序约束**：Sphinx 在 `config-inited` 之前初始化 i18n |
| 扩展清单与顺序、主题回退、sitemap 环境分支、字体装配、`DEFAULT_*` 引用 | Python | 机制，需代码表达 |

## 3. 实现要点

1. `ConfigManager._apply_top_level()`：遍历 toml 顶层键写入 `Config`，跳过 `html_theme_options`；Sphinx 未注册的键**告警但仍写入**，避免静默丢弃。
2. `config_inited_handler` 顺序调整：**先** `add_config_value("use_thebe"/"version_switcher_json_url")`，**再**合并。否则 toml 中的切换清单路径会被判为未注册而丢失，版本下拉整块失效。
3. `test_repo_conf_points_at_local_manifest` 由「加载 conf.py 断言属性」改为「解析 `_config.toml` 断言取值」，并额外断言清单文件真实存在。

## 4. 过程中发现的真实缺陷

首次迁移把 `language` 一并搬入 toml，构建产物 `<html lang="en">`、`Index`/`Search`、`Skip to main content` 全部退化为英文——`_init_i18n()` 早于 `config-inited`，延迟合并拿不到语言。已回退至 `meta.py`，并把原因写进模块 docstring 与 toml 注释，防止后人在「统一迁入」时重犯。

## 5. 验证

| 项 | 结果 |
|---|---|
| `index.html` / `search.html` 逐行对比（对 HEAD 独立工作树构建） | **完全一致**（过滤时间戳与 tippy 哈希） |
| `genindex.html` / `py-modindex.html` | 差异仅来自新增 `_apply_top_level` 符号 |
| `sitemap.xml` | 存在；本地前缀 `127.0.0.1:8000`，CI 前缀 `https://xinetzone.github.io/mystx/` |
| 切换下拉 / 页脚版权 / OGP `site_name` / Hypothesis | 均在 |
| `pytest tests/unit` | 119 passed（新增 2 条顶层合并契约测试） |
| `ruff check` | 通过 |
| 规模 | `conf.py` 79→45、`appearance.py` 29→14、`extensions.py` −12、`_config.toml` +30 |

## 6. 方法论沉淀

1. **迁移前先问「消费时机」**：Sphinx 配置并非都在 `config-inited` 之后被读取；i18n、`extensions` 属于读取期，天然不可延迟。
2. **契约测试跟着数据走**：配置从 Python 搬到 toml 后，断言对象应改为文件本身，而非继续加载模块。
3. **等价性用产物证明**：对独立工作树构建并按噪声过滤后逐行 diff，比「看起来没变」可靠。
4. **把踩过的坑写进被修改的文件**：i18n 时序说明同时出现在 `meta.py` 与 `_config.toml`，是最高性价比的防回归手段。

## 7. 后续

- 可为 toml 增加键级 schema 校验（当前未注册键仅告警），在 CI 中以 `-W` 提升为错误。
- `html_baseurl` 仍由环境分支在 Python 中计算，若将来固定为单一托管线，可并入 toml。
