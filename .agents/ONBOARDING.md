# 玄境（Xuanspace）入门指南

## 快速开始

### 前置条件

- Python 3.14.6+（硬性要求）
- Git
- 可选：CMake + Ninja（仅构建 C++ 原生扩展需要）
- 可选：PDM/uv/pip 任一包管理器

### 环境准备

1. 检查 Python 版本：
   ```bash
   python --version  # 必须 >= 3.14.6
   ```

2. 安装项目依赖（任选一种）：

   **使用 PDM（推荐）：**
   ```bash
   pdm install
   ```

   **使用 pip：**
   ```bash
   pip install -e ".[dev]"
   ```

   **使用 uv：**
   ```bash
   uv pip install -e ".[dev]"
   ```

3. 验证安装：
   ```bash
   xs --help
   xs doctor
   ```

## 能力速查表

| 能力 | 命令/方式 | 说明 |
|---|---|---|
| 环境检查 | `xs doctor` | 检查Python版本、依赖完整性 |
| 列出子项目 | `xs list` | 显示所有apps/libs子项目 |
| 创建Python子项目 | `xs new --type python <name>` | 在libs/下创建Python库 |
| 创建C++扩展 | `xs new --type native <name>` | 在libs/下创建C++原生扩展 |
| 创建静态项目 | `xs new --type static <name>` | 在apps/下创建静态HTML项目 |
| 查看工具链 | `xs toolchain` | 显示构建工具链信息 |

## 常用 xs 命令详解

### xs new - 创建新项目

```bash
# 在 libs/ 下创建 Python 库
xs new --type python my-lib

# 在 libs/ 下创建 C++ 原生扩展
xs new --type native my-ext

# 在 apps/ 下创建静态 HTML 项目
xs new --type static my-app
```

模板位置：`tools/templates/`

### xs list - 列出子项目

```bash
xs list
```

显示所有子项目的名称、类型、路径和状态。

### xs doctor - 环境诊断

```bash
xs doctor
```

自动检查：
- Python 版本是否符合要求
- 必要依赖是否安装
- CMake/Ninja 是否可用（如需要）
- 项目结构是否完整

## 项目结构速览

```
xuanspace/
├── apps/       # 可执行应用（CLI工具、Web服务、静态HTML）
├── libs/       # 可复用库（Python库、C++扩展）
│   ├── xuan-core/      # 核心工具库（纯Python）
│   └── xuan-ext-demo/  # C++扩展示例
├── tools/      # 项目工具链
│   ├── xs/             # xs CLI源码
│   └── templates/      # 新项目模板
├── docs/       # Sphinx文档
├── vendor/     # 第三方依赖（只读）
└── .agents/    # AI智能体规范
```

## 第一次任务？

如果你是第一次在玄境项目中执行任务，请：

1. 已读取根目录 `AGENTS.md`
2. 根据任务类型查阅 `context-routing.md`
3. 如涉及子项目，先读取该子项目的 `README.md` 和配置文件
4. 简单任务可直接执行，复杂任务先规划再动手

## 代码风格

- 行宽：120字符
- 格式化工具：ruff + black + isort
- 类型注解：鼓励但不强制
- 目标版本：Python 3.14+

配置见根目录 `pyproject.toml` 中的 `[tool.ruff]`、`[tool.black]`、`[tool.isort]` 部分。
