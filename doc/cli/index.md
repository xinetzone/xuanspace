# CLI 参考

`xs` 是 Xuanspace 的命令行工具，提供项目管理的统一入口。

## 全局选项

| 选项 | 说明 |
|---|---|
| `--help` | 显示帮助信息 |
| `--version` | 显示版本号 |

## 命令总览

| 命令 | 说明 |
|---|---|
| `xs list` | 列出所有子项目 |
| `xs new` | 从模板创建新项目 |
| `xs build` | 构建项目 |
| `xs doctor` | 环境诊断 |
| `xs init` | 初始化工作区 |
| `xs deps` | 依赖管理 |
| `xs version` | 版本管理 |
| `xs docs` | 文档管理 |
| `xs meta` | 元数据管理 |
| `xs toolchain` | 工具链管理 |
| `xs py-compat` | Python 兼容性检查 |
| `xs update` | 子模块与依赖更新 |
| `xs affected` | 检测受影响的子项目 |

## xs list

列出所有子项目。

```bash
xs list [--type python|native|static|other] [--dir apps|libs|tools|vendor] [--json]
```

| 选项 | 说明 |
|---|---|
| `--type` | 按项目类型筛选 |
| `--dir` | 按目录筛选 |
| `--json` | JSON 格式输出 |

## xs new

从模板创建新项目。

```bash
xs new --type python|native|static <name> [--app] [--dir apps|libs]
```

| 选项 | 说明 |
|---|---|
| `--type` | 项目类型：python（库）、native（C++扩展）、static（静态项目） |
| `--app` | 创建为应用（放在 apps/ 下，仅 python 类型） |
| `--dir` | 指定目标目录 |

## xs build

构建项目。

```bash
xs build [--project <name>] [--type python|native|static]
```

| 选项 | 说明 |
|---|---|
| `--project` | 指定项目名称 |
| `--type` | 按项目类型构建 |

## xs doctor

环境诊断，检查开发环境完整性。

```bash
xs doctor
```

## xs init

初始化 Xuanspace 工作区。

```bash
xs init [--name <name>] [--force] [--scaffold]
```

| 选项 | 说明 |
|---|---|
| `--name` | 工作区名称 |
| `--force` | 覆盖已存在的文件 |
| `--scaffold` | 创建全新工作区脚手架 |

## xs deps

依赖管理命令组。

```bash
xs deps check          # 检查过时依赖
xs deps tree           # 显示依赖树
xs deps outdated       # 列出过时包
xs deps update         # 更新依赖
```

### xs deps update

```bash
xs deps update [--dry-run] [--project <name>] [--type major|minor|patch] [--no-install] [<packages>...]
```

| 选项 | 说明 |
|---|---|
| `--dry-run` | 预览变更，不实际修改 |
| `--project` | 只更新指定项目 |
| `--type` | 限制更新类型 |
| `--no-install` | 仅更新 pyproject.toml，不执行安装 |
| `<packages>` | 指定要更新的包名 |

## xs version

版本管理命令组。

```bash
xs version show           # 显示版本信息
xs version bump <name>    # 版本 bump
```

### xs version bump

```bash
xs version bump <name> --type major|minor|patch [--no-tag] [--no-changelog]
```

| 选项 | 说明 |
|---|---|
| `--type` | bump 类型 |
| `--no-tag` | 不创建 Git 标签 |
| `--no-changelog` | 不更新 CHANGELOG |

## xs docs

文档管理命令组。

```bash
xs docs build              # 构建 HTML 文档
xs docs serve [--port 8000]  # 启动文档预览服务器
xs docs clean              # 清理构建产物
xs docs linkcheck          # 检查文档链接
```

## xs meta

文档元数据管理命令组。

```bash
xs meta init               # 初始化元数据目录
xs meta validate [--fix] [path]  # 验证 frontmatter 合规性
xs meta scan [path]        # 扫描元数据状态
xs meta sync [path]        # 同步 frontmatter 与 TOML 元数据
```

## xs toolchain

工具链管理命令组。

```bash
xs toolchain check         # 检查工具链完整性
xs toolchain list          # 列出所有工具
xs toolchain install <tool>  # 安装指定工具
```

## xs py-compat

检查依赖包与 Python 版本的兼容性。

```bash
xs py-compat [--json] [--py 3.14]
```

## xs update

更新子模块和依赖。

```bash
xs update [--submodules] [--deps] [--all] [--dry-run]
```

## xs affected

基于 git diff 检测受影响的子项目。

```bash
xs affected
```