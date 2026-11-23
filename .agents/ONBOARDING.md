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
   pip install -e tools/xs   # 本地 CLI 发行版 xs-cli，未发布 PyPI，须单独安装
   ```

   **使用 uv：**
   ```bash
   uv pip install -e ".[dev]"
   uv pip install -e tools/xs
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
| 初始化工作区 | `xs init` | 初始化工作区骨架 |
| 构建项目 | `xs build` | 按项目类型分派构建实现 |
| 依赖管理 | `xs deps` | 依赖关系查看与校验 |
| 版本管理 | `xs version` | 版本 bump 与 CHANGELOG |
| 文档管理 | `xs doc` | 文档构建与本地预览 |
| 元数据管理 | `xs meta` | 由 TOML 单源写入文档元数据 |
| 工具链 | `xs toolchain` | 显示构建工具链信息 |
| Python 兼容性 | `xs py-compat` | 检查目标 Python 版本兼容性 |
| 子模块与依赖更新 | `xs update` | 更新子模块与依赖 |
| 受影响项目 | `xs affected` | 检测改动影响的子项目 |
| 归档管理 | `xs archive` | 子项目归档与恢复 |
| Git LFS | `xs lfs` | 大文件检查与管理 |
| 插件装配 | `xs plugin` | 插件清单、装配图与求解计划 |

> 以上为 `xs` 的全部 16 个子命令（`toolchain`/`deps`/`doc`/`version`/`meta`/`archive`/`lfs`/`plugin` 为命令组）。命令名的事实来源是 [commands.py](../../tools/xs/src/xs/plugins/commands.py) 的登记表，完整性由 `scripts/check_declaration_consistency.py` 机检。

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
├── apps/                     # 可执行应用（预留）
├── libs/                     # 共享库
│   ├── xuan-plugin/          # 插件内核：可逆装配、能力注册、生命周期
│   ├── xuan-build/           # 构建能力层：python/native/static 分派
│   ├── xuan-ffi/             # FFI 能力键与子进程隔离代理
│   ├── xuan-project/         # 工作区项目发现与 pyproject 解析
│   ├── xuan-cmd/             # 命令装配与工程路径解析
│   ├── xuan-core/            # 核心工具库（纯 Python）
│   ├── xuan-ext-demo/        # C++ 扩展示例
│   ├── demo-ffi/             # FFI 扩展模板（Stub/Real 双运行时）
│   ├── xuan-fea/             # OpenType Feature AST 原语
│   ├── xuan-compose/         # Compose 声明式容器编排引擎
│   ├── caffe-ffi/            # Caffe 深度学习框架 FFI 绑定
│   ├── npu-ffi/              # VTA NPU FFI 绑定
│   ├── mystx/（子模块）       # MyST 文档工具链/主题
│   ├── tao/（子模块）         # AgentForge：AI Agent 协作基础设施
│   └── tvm-book/（子模块）    # TVM 中文手册 + flexloopy
├── tools/
│   ├── xs/                   # xs CLI（16 个子命令，命令面以插件登记）
│   ├── okf/                  # OKF v0.2 工具链
│   └── templates/            # 新项目模板
├── src/                      # 根枢纽包 xuanspace（版本与元信息）
├── tests/                    # 根级契约测试
├── doc/                      # Sphinx 文档
├── scripts/                  # 一致性门禁脚本（声明/依赖方向/插件契约）
├── vendor/                   # 第三方依赖（tvm-ffi、caffe，只读子模块）
├── .agents/                  # AI 智能体规范
└── xuan.plugins.lock         # 插件装配锁（由清单派生的规范 JSON）
```

`xs` 的 16 个子命令并非硬编码在 CLI 中，而是作为 `xs.commands` 插件由 `xuan-plugin` 内核装配；装配层次与锁文件机制见根 `README.md` 的「插件装配视图」。

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
