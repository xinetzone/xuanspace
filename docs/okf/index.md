# OKF 工具链

`okf` 是 Open Knowledge Format (OKF) v0.2 的命令行工具链，用于在生产/消费 OKF Bundle 时完成一致性校验、脚手架生成、索引/日志合成、信任等级推导等标准化操作。

OKF v0.2 将知识表示为一组携带 YAML frontmatter、互相链接、可被 Git 管理的 Markdown 文件。`okf` 工具链把这些规范约束从「文档约定」变为「可由工具强制校验与自动生成」的能力。

## 特性

- **零运行时依赖**：仅使用 Python 3.14.6 标准库（`dataclasses`/`pathlib`/`argparse`/`re`/`asyncio`/`tomllib`），不依赖 PyYAML/typer/rich 等第三方包。
- **Dataclass 数据模型**：所有核心类型使用 `@dataclass(frozen=True)` 定义，不可变、类型安全。
- **时空可组合性架构**（Cordis 启发）：可逆效应（Reversible Effects）+ 响应式协同效应（Reactive Coeffects）。
- **Harness 架构**（DeepSeek Harness 启发）：一切皆插件、无特权内核、Capability Seam 三角色、事件系统五种分发模式。
- **scikit-build-core 构建**：纯 Python 起步，预留 C/C++ 原生模块扩展点。

## 安装

```powershell
pip install .
```

## 全局选项

| 选项 | 说明 |
|---|---|
| `--version` / `-V` | 显示版本号 |
| `--help` | 显示帮助信息 |

## 命令总览

| 命令 | 说明 |
|---|---|
| `okf validate <path> [--strict]` | 一致性校验（§11） |
| `okf init <path>` | 创建 Bundle 骨架 |
| `okf index <path>` | 生成/更新 `index.md` |
| `okf inspect <path> [concept_id]` | 查看概念详情 |
| `okf trust <path> [concept_id]` | 信任等级与保鲜状态 |
| `okf list <path> [--type X] [--tag Y]` | 列出概念 |

## okf validate

对 Bundle 目录执行一致性校验，输出错误与警告报告；发现错误时退出码为 1。

```bash
okf validate <path> [--strict]
```

## okf init

创建 Bundle 骨架目录结构：

- `index.md`
- `log.md`
- `concepts/`
- `playbooks/`
- `references/`

```bash
okf init <path>
```

## okf index

根据目录中的概念文件生成/更新 `index.md`。

```bash
okf index <path>
```

## okf inspect

查看 Bundle 概览或单个概念的详情（含 frontmatter 与 body 摘要）。

```bash
okf inspect <path> [concept_id]
```

## okf trust

输出概念的信任等级（`unverified`/`machine_confirmed`/`human_reviewed`）与保鲜状态。

```bash
okf trust <path> [concept_id]
```

## okf list

按条件过滤并列出概念（可指定 `--type` 或 `--tag`）。

```bash
okf list <path> [--type X] [--tag Y]
```

## 插件装配

插件通过 `pyproject.toml` 的 `[tool.okf.plugins]` 配置节声明式装配，第三方实现可替换任意默认插件：

```toml
[tool.okf.plugins]
conformance_checker = "my_org.okf:CustomChecker"
```

默认声明 7 个插件：`bundle_loader`、`conformance_checker`、`index_synthesizer`、`log_synthesizer`、`trust_deriver`、`link_resolver`、`cli_adapter`。

## 参考

- OKF 规范：`vendor/knowledge-catalog/okf/SPEC.md`
- 工具源码：`tools/okf/`