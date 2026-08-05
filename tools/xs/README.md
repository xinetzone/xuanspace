# xs-cli

xs-cli 是玄境（Xuanspace）monorepo 的命令行工具，用于统一管理工作区内的项目发现、版本管理、依赖分析等日常开发任务。它基于 Typer 构建，提供友好的终端交互体验。

## 特性

- 项目管理：扫描并列出工作区内的应用、库与工具项目
- 版本管理：查看与升级项目版本（`major` / `minor` / `patch`），同步维护 `CHANGELOG.md`
- 环境诊断：`xs doctor` 检查工作区环境与依赖健康状态
- 依赖分析：构建项目依赖图，定位受变更影响的子项目

## 安装

使用 pip 直接安装：

```bash
pip install xs-cli
```

## 常用命令

```bash
# 查看帮助
xs --help

# 列出工作区内所有项目
xs list

# 环境诊断
xs doctor

# 查看项目版本
xs version show

# 升级版本（major / minor / patch）
xs version bump minor
```

## 开发说明

项目采用 `src/` 布局，源码位于 `src/xs/`。开发时建议在虚拟环境中以可编辑模式安装依赖：

```bash
pip install -e .
```

### 单源版本化

版本号以 `src/xs/__init__.py` 中的 `__version__` 为单一事实来源，`pyproject.toml` 通过 `[tool.setuptools.dynamic]` 动态读取。使用 `xs version bump` 升级版本时，会同步更新 `__init__.py` 与 `pyproject.toml`，保证两者一致。

### 运行测试 / 代码检查

```bash
# 代码检查
ruff check src/

# 测试（如有）
pytest
```

## 许可

MIT License