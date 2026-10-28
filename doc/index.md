# Xuanspace（玄境）

```{include} ../README.md
```

## 文档系统

`doc/` 目录存放 Xuanspace（玄境）项目的全部文档源码，使用 **Sphinx + MyST Markdown** 构建，支持 Markdown 编写、Python API 文档自动生成与交叉引用。

### 技术栈

- **构建工具**：Sphinx
- **Markdown 支持**：MyST Parser（标准 Markdown + Sphinx 扩展语法）
- **API 文档**：`sphinx.ext.autodoc`（配合 napoleon / viewcode，从 Python 源码注释生成）
- **主题**：按可用性回退 `mystx` → `sphinx_book_theme` → `alabaster`（见 [conf.py](conf.py)）
- **输出格式**：HTML（默认）

### 常用构建命令

```bash
# 方式一：xs CLI（推荐）
xs doc build      # 构建 HTML
xs doc serve      # 本地预览（自动刷新）
xs doc linkcheck  # 链接检查

# 方式二：直接调用 Sphinx
pdm run sphinx-build -b html doc/ doc/_build/html
```

构建产物输出到 `doc/_build/`（已在 .gitignore 中）。

> MyST 语法、写作风格与构建细节等文档编写规范，见[贡献指南](contributing.md)。

## 文档导航

```{toctree}
:caption: 开始使用
:maxdepth: 2

intro
quickstart
```

```{toctree}
:caption: 指南
:maxdepth: 2

user-guide/index
cli/index
okf/index
build-system
architecture
```

```{toctree}
:caption: 项目
:maxdepth: 1

contributing
dependency-report
adr/index
```

```{toctree}
:hidden:

genindex
search
```

## 文档 PR 要求

提交文档变更时需满足：

- [ ] `xs doc build` 构建无错误和警告（允许 nitpick 警告）
- [ ] `xs doc linkcheck` 链接检查通过
- [ ] 新增文档已正确添加到对应目录的 index.md 目录树
- [ ] 所有代码示例可正常运行（如有必要，包含对应的测试）
- [ ] 图片等静态资源放入 `_static/` 目录
