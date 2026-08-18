---
id: "xuanspace-contributing"
version: "0.1.0"
x-toml-ref: ".meta/toml/CONTRIBUTING.toml"
---

# 贡献指南

欢迎为 Xuanspace（玄境）项目贡献代码！无论你是修复 Bug、添加新功能、改进文档，还是优化性能，我们都非常感谢你的参与。

Xuanspace 是一个 Python 3.14.6+ monorepo 项目，支持 PDM/uv/pip 三种包管理器，并支持 C++ 原生扩展构建。请按照本指南设置开发环境并参与贡献。

---

## 1. 欢迎贡献

我们欢迎任何形式的贡献，包括但不限于：

- 代码修复与新功能开发
- 文档改进与翻译
- Bug 报告与功能建议
- 性能优化
- 测试用例补充
- 代码审查

请在开始重大变更前，先通过 Issue 讨论你的想法，避免重复工作或方向偏离。

---

## 2. 环境搭建

### 2.1 前置条件（必需）

- **Git**：版本控制，[安装指引](https://git-scm.com/downloads)
- **Python 3.14.6+**：核心运行环境（必须，不支持更低版本）

#### Python 3.14.6+ 安装指引

**Windows：**
- 推荐从 [Python 官网](https://www.python.org/downloads/) 下载安装包（安装时勾选 "Add Python to PATH"）
- 或使用 winget：`winget install Python.Python.3.14`
- 或使用 conda：`conda create -n xuanspace python=3.14 && conda activate xuanspace`

**macOS：**
- 使用 Homebrew：`brew install python@3.14`
- 或使用 pyenv：`pyenv install 3.14.6 && pyenv global 3.14.6`

**Linux（Ubuntu/Debian）：**
- 使用 deadsnakes PPA：
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install python3.14 python3.14-venv python3.14-dev
  ```
- 或使用 pyenv：`pyenv install 3.14.6 && pyenv global 3.14.6`

验证安装：
```bash
python --version  # 应显示 Python 3.14.x
```

### 2.2 可选工具

- **PDM（推荐）**：最佳 workspace 支持，自动链接所有子项目
- **uv**：极速包管理器，作为 PDM 的快速替代
- **CMake ≥ 3.26 + Ninja**：仅在开发 C++ 原生扩展时需要

### 2.3 包管理器选择说明（重要）

Xuanspace 支持三种包管理器，**完全等价，不强制任何一种**，选择你熟悉的即可：

| 包管理器 | 优点 | 安装命令 |
|---------|------|---------|
| **PDM（推荐）** | workspace 自动链接本地依赖，`pdm install` 一键安装所有子项目 | `pip install pdm` |
| **uv** | 安装速度极快，Rust 实现 | `pip install uv` |
| **pip** | Python 标准工具，无需额外安装 | （Python 自带） |

> **注意**：使用 PDM 时无需手动逐个安装子项目，`pdm install` 会自动处理；使用 uv/pip 时需要手动安装各子项目。

### 2.4 CMake 和 Ninja 安装（仅 C++ 扩展需要）

如果你需要开发或构建 C++ 原生扩展，请安装 CMake ≥ 3.26 和 Ninja。分平台安装方式如下：

**Windows：**
```powershell
# 方式一：winget（推荐）
winget install Kitware.CMake Ninja-build.Ninja

# 方式二：Chocolatey
choco install cmake ninja

# 方式三：pip（通用 fallback）
pip install cmake ninja
```

**macOS：**
```bash
brew install cmake ninja
```

**Linux（Ubuntu/Debian）：**
```bash
sudo apt install cmake ninja-build
```

**通用 Fallback（所有平台）：**
```bash
pip install cmake ninja
```

安装后验证：
```bash
cmake --version  # 应显示 cmake version 3.26.x 或更高
ninja --version
```

还需要安装对应平台的 C++ 编译器：
- Windows：MSVC（Visual Studio 2022+，安装"使用 C++ 的桌面开发"工作负载）或 MinGW-w64 GCC
- macOS：Clang（`xcode-select --install` 安装 Xcode Command Line Tools）
- Linux：GCC 12+ 或 Clang 16+（`sudo apt install build-essential`）

---

## 3. 获取代码

### 3.1 克隆仓库

使用 `--recurse-submodules` 一次性克隆并初始化所有子模块：

```bash
git clone --recurse-submodules https://github.com/xinetzone/xuanspace.git
cd xuanspace
```

如果你已经克隆了但没有初始化子模块：
```bash
git submodule update --init --recursive
```

### 3.2 （可选）Fork 工作流

如果你没有直接推送权限，请先 Fork 本仓库到你的账号，然后克隆你的 Fork：

```bash
git clone --recurse-submodules https://github.com/<your-username>/xuanspace.git
cd xuanspace
git remote add upstream https://github.com/xinetzone/xuanspace.git
```

后续同步上游更新：
```bash
git checkout main
git pull upstream main
git submodule update --init --recursive
```

### 3.3 嵌套子模块说明

Xuanspace 的 `vendor/` 目录下的子模块（如 `tvm-ffi`）自身可能也包含子模块（嵌套子模块）。例如 `tvm-ffi` 依赖 `3rdparty/dlpack` 和 `3rdparty/libbacktrace`。

**现象**：克隆后 `vendor/<name>/3rdparty/` 目录存在但为空，构建时出现 `CMake Error: submodule not found` 等错误。

**原因**：`git clone --recurse-submodules` 默认只初始化一级子模块，不会递归初始化嵌套子模块的内部子模块。

**解决方案**：

| 场景 | 命令 |
|------|------|
| 首次克隆 | `git clone --recurse-submodules https://github.com/xinetzone/xuanspace.git` |
| 已克隆但子模块为空 | `git submodule update --init --recursive` |
| 只初始化特定子模块 | `cd vendor/<name> && git submodule update --init --recursive` |

> **关键**：`--recursive` 参数确保递归初始化所有层级的嵌套子模块。不带此参数只会初始化一级子模块，嵌套子模块的内部依赖仍为空。

**验证方法**：执行以下命令确认所有嵌套子模块已正确初始化：

```bash
# 在项目根目录查看所有子模块状态（含嵌套）
git submodule status --recursive

# 预期输出示例（所有条目不以 '-' 开头，表示已初始化）：
#  <hash> vendor/tvm-ffi (tag)
#  <hash> vendor/tvm-ffi/3rdparty/dlpack (tag)
#  <hash> vendor/tvm-ffi/3rdparty/libbacktrace (hash)
```

**CI 配置**：GitHub Actions 的 `actions/checkout` 需设置 `submodules: recursive`：

```yaml
- name: Checkout code
  uses: actions/checkout@v4
  with:
    submodules: recursive
```

### 3.4 排查子模块问题

如果克隆后构建/测试失败，首先检查子模块状态：

```bash
# 检查所有子模块是否已初始化
git submodule status --recursive

# 如果某行以 '-' 开头，说明该子模块未初始化
# 示例：-84d107bf416c6bab9ae68ad285876600d230490d vendor/tvm-ffi/3rdparty/dlpack
# 运行以下命令修复：
git submodule update --init --recursive
```

---

## 4. 安装开发依赖

三种包管理器的安装方式如下，**选择一种即可**。

### 4.1 方式一：PDM（推荐）

PDM 会自动识别 monorepo 结构，一键安装所有依赖并链接子项目：

```bash
pdm install
```

这将：
- 自动创建虚拟环境（如果不存在）
- 安装根项目和所有 libs/、apps/、tools/ 下子项目的依赖（包括 dev 依赖）
- 以可编辑模式安装所有本地包
- 配置好 `xs` CLI 命令

### 4.2 方式二：uv（快速替代）

uv 速度更快，但需要手动安装各子项目：

```bash
# 安装根项目及开发依赖
uv pip install -e ".[dev]"

# 手动安装各子项目
uv pip install -e libs/xuan-core
uv pip install -e libs/xuan-ext-demo
uv pip install -e apps/xs-cli
uv pip install -e tools/xs
```

### 4.3 方式三：pip（标准方式）

使用 Python 标准 pip：

```bash
# 安装根项目及开发依赖
pip install -e ".[dev]"

# 手动安装各子项目
pip install -e libs/xuan-core
pip install -e libs/xuan-ext-demo
pip install -e apps/xs-cli
pip install -e tools/xs
```

### 4.4 验证安装

安装完成后，验证环境是否正常：

```bash
# 运行环境诊断
xs doctor

# 列出所有可用子项目
xs list

# 查看 CLI 帮助
xs --help
```

如果 `xs doctor` 显示所有检查通过，说明环境搭建成功。

---

## 5. 分支策略

| 分支类型 | 命名格式 | 说明 |
|---------|---------|------|
| `main` | - | 保护分支，**不能直接推送**，只能通过 PR 合并 |
| 功能分支 | `feature/<name>` | 新功能开发，例：`feature/native-image-processor` |
| 修复分支 | `bugfix/<name>` | Bug 修复，例：`bugfix/path-handling-windows` |
| 文档分支 | `docs/<name>` | 文档改进，例：`docs/update-contributing-guide` |

> 也可使用 Conventional Commits 风格的简写：`feat/<name>`、`fix/<name>`，效果相同。

---

## 6. 开发工作流

### 6.1 标准流程

```
Fork → Clone → 创建分支 → 开发 → 测试 → 提交 → 创建 PR → Review → 合并
```

详细步骤：

1. **创建分支**：从最新的 main 分支创建你的工作分支
   ```bash
   git checkout main
   git pull upstream main  # 或 origin/main（如果你没有 fork）
   git checkout -b feature/your-feature-name
   ```

2. **开发代码**：按照代码规范编写代码，详见第 8 节

3. **本地测试**：运行测试和代码质量检查，确保所有检查通过
   ```bash
   pytest          # 运行测试
   xs build        # 构建 C++ 扩展（如有）
   xs docs build   # 构建文档（如修改了文档）
   ruff check .    # Lint 检查
   mypy .          # 类型检查
   ```

4. **提交代码**：遵循 Conventional Commits 规范，详见第 7 节

5. **推送分支**：
   ```bash
   git push origin feature/your-feature-name
   ```

6. **创建 Pull Request**：在 GitHub 上创建 PR，填写 PR 模板，等待 Review

### 6.2 新增子项目流程

使用 `xs new` 命令快速创建符合规范的子项目：

**纯 Python 库/应用：**
```bash
# 创建可复用 Python 库（自动放到 libs/）
xs new --type python <project-name>

# 创建可执行应用/CLI（自动放到 apps/，加 --app 参数）
xs new --type python --app <app-name>
```

**C++ 原生扩展：**
```bash
# 确保已安装 CMake、Ninja 和 C++ 编译器
xs new --type native <extension-name>
# 原生扩展统一放到 libs/ 目录
```

**静态前端项目（HTML/CSS/JS）：**
```bash
xs new --type static <site-name>
# 静态项目放到 apps/ 目录
```

`xs new` 会自动生成：
- 标准的 `pyproject.toml`
- `src/<package_name>/` 包结构（Python 项目）
- `tests/` 目录与基础测试用例
- `README.md` 模板
- `CHANGELOG.md`
- C++ 项目还会生成 `CMakeLists.txt` 和 `CMakePresets.json`

### 6.3 本地常用命令

| 命令 | 说明 |
|-----|------|
| `xs doctor` | 环境诊断检查 |
| `xs list` | 列出所有子项目 |
| `xs build` | 构建所有 C++ 扩展 |
| `xs build --debug` | Debug 模式构建 |
| `xs docs build` | 构建 Sphinx 文档 |
| `xs docs serve` | 启动本地文档服务器预览 |
| `xs new` | 创建新子项目（交互式向导） |
| `pytest` | 运行所有测试 |
| `pytest tests/test_file.py -v` | 运行指定测试 |
| `pytest --cov` | 运行测试并生成覆盖率报告 |
| `ruff check .` | 代码 Lint 检查 |
| `ruff check --fix .` | 自动修复可修复的 Lint 问题 |
| `ruff format .` | 代码格式化 |
| `mypy .` | 类型检查 |

---

## 7. Commit Message 规范

我们遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 规范。

### 7.1 格式

```
<type>(<scope>): <subject>
```

### 7.2 type 类型说明

| type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档变更 |
| `style` | 代码格式（不影响代码运行的变动，如空格、格式化、分号等） |
| `refactor` | 重构（既不是新增功能，也不是修改 bug 的代码变动） |
| `perf` | 性能优化 |
| `test` | 增加测试或修正测试 |
| `chore` | 构建过程或辅助工具的变动 |
| `build` | 构建系统或外部依赖变更 |
| `ci` | CI 配置变更 |
| `revert` | 回滚提交 |

### 7.3 scope 说明

`scope` 用于说明提交影响的范围，通常是子项目名或模块名，例如：
- `xuan-core`
- `xuan-ext-demo`
- `xs-cli`
- `docs`
- `build`

如果影响多个范围或全局变更，可以省略 scope。

### 7.4 subject 说明

- 使用中文简短描述
- 结尾不加句号
- 动词开头，使用第一人称现在时（如"添加"、"修复"、"更新"，而非"添加了"、"修复了"）

### 7.5 示例

```
feat(xuan-core): 添加路径处理工具函数
fix(xs-cli): 修复 Windows 下路径分隔符问题
docs: 更新安装指南，添加 uv 安装方式
refactor(xuan-ext-demo): 重构张量运算接口
test: 为 xuan-core 添加单元测试覆盖
chore: 更新 ruff 到 0.5.x
build: 升级 CMake 最低要求到 3.26
```

### 7.6 强制检查

提交前请确保：
- [ ] 所有测试通过
- [ ] ruff check 无错误
- [ ] ruff format 已格式化
- [ ] mypy 类型检查通过
- [ ] Commit message 符合上述规范

---

## 8. 代码规范

### 8.1 Python 版本

所有代码必须使用 **Python 3.14.6+** 语法，允许使用 Python 3.14 新特性：
- 类型参数语法（`def func[T](x: T) -> T`）
- 更强大的类型推断
- 新的标准库功能

最低要求是 Python 3.14.6，不向下兼容。

### 8.2 类型注解

- 所有公共函数、方法、类必须添加类型注解
- 模块级公开变量建议添加类型注解
- 内部函数根据复杂度酌情添加
- 配置见 `pyproject.toml` 中的 `[tool.mypy]`

示例：
```python
from pathlib import Path
from typing import Iterable


def find_files(root: Path, pattern: str = "*.py") -> Iterable[Path]:
    """查找指定目录下匹配模式的文件。"""
    return root.rglob(pattern)
```

### 8.3 Lint 与格式化工具

我们使用以下工具保证代码质量，配置均在 `pyproject.toml` 中：

| 工具 | 用途 | 配置 |
|-----|------|------|
| **ruff** | Lint 检查 + 导入排序 | `line-length = 120`，`target-version = "py314"` |
| **black** | 代码格式化 | `line-length = 120`，`target-version = ["py314"]` |
| **isort** | 导入排序 | `profile = "black"`，与 black 兼容 |
| **mypy** | 静态类型检查 | `python_version = "3.14"` |

> 注意：ruff 已包含 isort 功能（`I` 规则），通常不需要单独运行 isort。

### 8.4 常用命令

```bash
# 格式化代码（black 风格）
ruff format .

# Lint 检查
ruff check .

# 自动修复可修复的 Lint 问题
ruff check --fix .

# 类型检查
mypy .

# 运行测试
pytest

# 运行测试并查看覆盖率
pytest --cov=libs --cov=apps --cov-report=term-missing
```

### 8.5 测试覆盖率要求

- 核心库（`libs/xuan-core/`）覆盖率应 ≥ 80%
- 新增功能必须包含对应的测试用例
- Bug 修复应先添加能复现问题的测试，再修复代码
- 测试文件命名：`test_*.py` 或 `*_test.py`
- 测试函数命名：`test_*`
- 测试类命名：`Test*`

### 8.6 C++ 代码规范

对于 C++ 原生扩展代码：
- 使用 C++17 或更高标准
- 遵循项目根目录的 `.clang-format` 风格（如有）
- 头文件使用 `#pragma once` 保护
- 导出函数使用 `extern "C"` 避免名称修饰
- 通过 pybind11 或 nanobind 提供 Python 绑定

---

## 9. Python 3.14 兼容性

Xuanspace 严格要求 **Python 3.14.6+**，确保代码兼容性：

1. **所有代码必须兼容 Python 3.14.6+**，不允许使用已废弃的语法或 API

2. **添加新依赖前必须检查兼容性**：
   ```bash
   xs py-compat <package-name>
   ```
   该命令会检查 PyPI 上该包是否支持 Python 3.14。不兼容的依赖原则上不允许添加，除非有充分理由且经过讨论。

3. **CI 中仅使用 Python 3.14 运行测试**，不测试更低版本

4. 可以使用的 Python 3.14 新特性包括但不限于：
   - 原生类型参数（PEP 695）
   - `typing` 模块的改进
   - 性能优化带来的新特性
   - 新的标准库模块

---

## 10. 跨平台开发注意事项

Xuanspace 支持 Windows、macOS、Linux 三大平台，开发时请注意跨平台兼容性：

### 10.1 路径处理

- **必须**使用 `pathlib.Path` 处理路径，**禁止**硬编码 `/` 或 `\` 分隔符
- 不要手动拼接路径字符串，使用 `Path` 对象的 `/` 运算符或 `joinpath()`
- 路径比较时注意大小写敏感性（Windows/macOS 默认不敏感，Linux 敏感）

正确示例：
```python
from pathlib import Path

config_path = Path(__file__).parent / "config" / "settings.toml"
data_dir = Path.home() / ".xuanspace" / "data"
```

错误示例：
```python
# 禁止这样写！
config_path = os.path.dirname(__file__) + "/config/settings.toml"
data_dir = f"{os.path.expanduser('~')}/.xuanspace/data"
```

### 10.2 CMake 跨平台构建

- 项目使用 `CMakePresets.json` 定义跨平台预设，包括：
  - `debug` / `release` / `release-with-debug`：通用构建预设
  - `debug-windows` / `release-windows`：Windows MSVC 预设
  - `debug-linux` / `release-linux`：Linux GCC 预设
  - `debug-macos` / `release-macos`：macOS Clang 预设
- 添加新的 C++ 代码时，确保在三大平台都能编译
- 使用 `xs build` 命令统一构建，不要直接调用 cmake

### 10.3 Shell 命令兼容性

- 避免直接调用平台特定的 Shell 命令（如 `cmd.exe` 的 `dir`、Unix 的 `ls`）
- 尽量使用 Python 标准库功能替代 Shell 命令
- 如果必须调用外部命令，使用 `subprocess` 模块，并注意：
  - 设置 `shell=False`（默认）
  - 使用列表形式传参，不要拼接字符串
  - 处理不同平台的可执行文件扩展名（Windows 上的 `.exe`）

### 10.4 换行符

- 项目配置了 `.gitattributes`，Git 会自动处理换行符
- Python 文件中统一使用 `\n`（LF）
- 不要在代码中手动处理 `\r\n` vs `\n`，使用通用换行模式（`newline=None`，Python 默认）

### 10.5 文件编码

- 所有源代码文件使用 **UTF-8** 编码
- Python 文件不需要（也不应该）添加 `# -*- coding: utf-8 -*-` 注释（Python 3 默认 UTF-8）
- 打开文件时显式指定 `encoding="utf-8"`：
  ```python
  with open(path, encoding="utf-8") as f:
      content = f.read()
  ```

---

## 11. Git LFS 大文件管理

Xuanspace 使用 **Git LFS**（Large File Storage）管理二进制大文件，避免仓库体积膨胀。

### 11.1 LFS 跟踪规则

项目根目录的 `.gitattributes` 预设了以下 LFS 跟踪规则：

| 文件类型 | 后缀 | 典型用途 |
|---------|------|---------|
| 图片 | `*.png`, `*.jpg`, `*.jpeg`, `*.gif` | 文档截图、示意图 |
| 文档 | `*.pdf` | 参考资料、论文 |
| 预编译包 | `*.whl` | Python wheel 包 |
| 动态库 | `*.so`, `*.dylib`, `*.dll` | 编译产物 |
| 模型 | `*.pth`, `*.onnx`, `*.pt`, `*.model` | 机器学习模型、序列化文件 |

### 11.2 环境要求

- **Git LFS 客户端**：提交前必须安装，[安装指引](https://git-lfs.com/)
- 验证安装：`git lfs version`
- 克隆仓库时使用 `--recurse-submodules` 会自动拉取 LFS 文件

### 11.3 新增 LFS 跟踪规则

如需跟踪新的文件类型：

```bash
# 添加跟踪规则（示例：跟踪 .zip 文件）
git lfs track "*.zip"

# 这将自动更新 .gitattributes
git add .gitattributes
git commit -m "chore: 添加 .zip 到 LFS 跟踪"
```

### 11.4 检查 LFS 状态

使用 `xs lfs` 命令检查 LFS 配置是否正确：

```bash
# 查看当前 LFS 跟踪模式
xs lfs patterns

# 检查是否有遗漏的大文件
xs lfs check

# 使用自定义阈值（默认 5MB）
xs lfs check --threshold 10
```

### 11.5 最佳实践

- **提交前检查**：每次提交前运行 `xs lfs check` 确保没有大文件遗漏
- **5MB 阈值**：超过 5MB 的非代码文件应使用 LFS 跟踪
- **CI 集成**：CI 流水线中可通过 `xs lfs check --json` 输出 JSON 结果，用于自动化检查
- **不要提交编译产物**：`build/`、`dist/`、`_build/`、`attic/` 目录已自动排除

---

## 12. 文档贡献

文档使用 **Sphinx + MyST Markdown** 编写，位于 `docs/` 目录。

### 12.1 文档结构

```
docs/
├── conf.py          # Sphinx 配置
├── index.md         # 文档首页
├── _static/         # 静态资源（图片、CSS 等）
├── Makefile         # Unix 构建脚本
├── make.bat         # Windows 构建脚本
└── ...              # 其他文档页面
```

文档依赖在根 `pyproject.toml` 的 `[project.optional-dependencies].docs` 中配置，安装方式：`pip install -e ".[docs]"` 或 `pdm install`（dev 组已包含 docs）。

### 12.2 MyST Markdown

我们使用 MyST（Markedly Structured Text），这是一种功能强大的 Markdown 风味，支持：
- 标准 CommonMark 语法
- Sphinx 指令和角色
- 目录树（toctree）
- 交叉引用
- 代码块语法高亮
- Mermaid 图表
- Admonitions（提示、注意、警告等）

### 12.3 构建文档

```bash
# 构建 HTML 文档
xs docs build
# 或
sphinx-build -b html docs docs/_build/html

# 启动本地预览服务器（自动重载）
xs docs serve
# 或
sphinx-autobuild docs docs/_build/html
```

构建完成后，打开 `docs/_build/html/index.html` 即可查看。

### 12.4 文档风格建议

- 中文文档使用中文标点
- 代码示例必须可以直接运行
- 复杂概念配图表说明
- 每个公开 API 都要有文档字符串
- README 保持简洁，详细内容放到官方文档

---

## 13. 问题反馈

我们使用 GitHub Issues 跟踪 Bug 和功能请求。提交 Issue 时，请使用对应的模板：

### 13.1 Bug 报告

请使用 Bug Report 模板，包含：
- **环境信息**：操作系统、Python 版本、Xuanspace 版本（`xs --version`）
- **复现步骤**：清晰的步骤描述，最好附上最小复现代码
- **期望行为**：你认为应该发生什么
- **实际行为**：实际发生了什么，包括完整的错误信息和堆栈跟踪
- **截图/日志**：如有必要，附上截图或日志

### 13.2 功能请求

请使用 Feature Request 模板，包含：
- **功能动机**：这个功能解决什么问题，你的使用场景
- **期望的解决方案**：你希望如何实现
- **替代方案**：你考虑过的其他方案
- **附加上下文**：其他相关信息、截图、参考链接等

### 13.3 Issue 处理流程

1. 提交 Issue 后，维护者会尽快进行 triage（分类）
2. Bug 会被标记为 `bug`，功能请求标记为 `enhancement`
3. 欢迎提交 PR 解决已确认的 Issue，PR 中请关联对应 Issue 编号（如 `Closes #123`）
4. 在开始处理较大的功能前，建议先在 Issue 中讨论设计方案，避免走弯路

---

感谢你为 Xuanspace 做出贡献！🎉
