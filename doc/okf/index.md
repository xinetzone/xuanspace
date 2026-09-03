# OKF 工具链教程

`okf` 是 Open Knowledge Format（OKF）v0.2 的命令行工具链，用于在生产与消费 OKF Bundle 时完成一致性校验、脚手架生成、索引/日志合成、信任等级推导等标准化操作。

OKF v0.2 把知识表示为**一组携带 YAML frontmatter、彼此链接、可由 Git 管理的 Markdown 文件**，这样一个自包含的知识单元称为一个 *Bundle*。`okf` 的作用，是把这些原本只能靠「文档约定」约束的规则，变成「可由工具强制执行校验与自动生成」的能力。

## 核心特性

- **零运行时依赖**：仅使用 Python 3.14.6 标准库（`dataclasses` / `pathlib` / `argparse` / `re` / `asyncio` / `tomllib`），不依赖 PyYAML / typer / rich 等第三方包。
- **Dataclass 数据模型**：核心类型均以 `@dataclass(frozen=True, slots=True)` 定义，不可变、类型安全。
- **Harness 架构**（DeepSeek Harness 启发）：一切皆插件、无特权内核、声明式装配、任意插件可被第三方替换。
- **可逆效应 + 响应式协同效应**（Cordis 启发）：资源逆序回收，依赖变更时自动重载插件。
- **scikit-build-core 构建**：纯 Python 起步，预留 C/C++ 原生模块扩展点。

## 安装

需要 Python >= 3.14.6。

```powershell
pip install .
```

安装后即可使用 `okf` 命令：

```powershell
okf --version
# okf 0.1.0
```

## 教程导航

```{toctree}
:caption: OKF 工具链教程
:maxdepth: 2

quickstart
bundle
commands
conformance
trust
synthesis
attested
architecture
plugins
```

## 相关资源

- 工具源码：`tools/okf/`
- 命令入口：`src/okf/cli.py`
- 默认插件：`src/okf/plugins/`