# scripts/ - 脚本目录

## 概述

`scripts/` 目录存放 Xuanspace（玄境）项目的维护脚本。这些脚本用于项目维护、CI/CD、版本发布、代码生成等一次性或运维类任务，不作为库被其他代码导入。

## 用途

本目录存放以下类型的脚本：

- CI/CD 流水线脚本
- 版本发布与变更日志生成脚本
- 代码生成/模板更新脚本
- 数据迁移脚本
- 一次性数据修复脚本
- 开发环境初始化脚本
- 批量重构/格式化脚本
- 性能基准测试脚本

## 分类标准

一个脚本应放入 `scripts/` 当且仅当满足以下条件：

1. 是**一次性或运维类**脚本，不是日常开发中反复使用的可复用工具
2. 不作为 Python 库被其他代码 `import`
3. 通常是单文件或少量文件，没有复杂的模块结构
4. 面向项目维护者，而非最终用户

## scripts/ 与 tools/ 的区别

这两个目录容易混淆，区分标准如下：

| 特性 | scripts/ | tools/ |
|------|----------|--------|
| 执行场景 | 手动执行或在 CI 中触发 | 作为 CLI 命令被开发者反复调用 |
| 复用性 | 低，特定任务专用 | 高，通用工具模块 |
| 导入方式 | 不被导入，直接运行 | 可以作为库被导入 |
| 结构 | 通常单文件 | 可能是多文件包，有 pyproject.toml |
| 用户 | 项目维护者 | 所有开发者 |

**判断原则**：如果你需要反复调用某段代码，并且它需要参数解析、配置处理等能力，考虑放入 `tools/`；如果是执行一次就完事（或 CI 定期执行）的任务，放入 `scripts/`。

## 命名规范

- 使用 **kebab-case** 或 **snake_case** 命名风格（推荐 snake_case 与 Python 模块名一致）
- 名称应清晰反映脚本的动作和目标
- 动词开头，表明执行的操作
- 可以包含版本信息或适用范围标识（如有必要）

**正确示例**：
- `bump_version.py` - 版本号升级脚本
- `generate_changelog.py` - 变更日志生成脚本
- `init_dev_env.py` - 开发环境初始化脚本
- `migrate_config_v1_to_v2.py` - 配置 v1 到 v2 迁移脚本
- `run_benchmarks.py` - 性能基准测试运行脚本

**错误示例**：
- `do_it.py`（含义不明）
- `script.py`（无意义名称）
- `fix.py`（修复什么？）
- `temp.py`（临时脚本应在完成后删除或归档）

## 脚本编写规范

### 1. Shebang 和编码声明

所有 Python 脚本应在文件开头包含：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```

### 2. 文档字符串

每个脚本必须有模块级 docstring，说明：

```python
"""
脚本名称：bump_version.py
用途：升级项目版本号并同步更新所有相关文件
使用方法：python scripts/bump_version.py [--major|--minor|--patch] [--dry-run]
作者：维护者姓名
创建日期：2024-01-01
"""
```

### 3. 参数解析

对于需要参数的脚本，使用标准库 `argparse` 解析命令行参数：

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="版本号升级脚本")
    parser.add_argument("--major", action="store_true", help="升级主版本号")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不实际修改文件")
    args = parser.parse_args()
    # ... 脚本逻辑

if __name__ == "__main__":
    main()
```

### 4. 幂等性要求（重要）

**所有 scripts/ 中的脚本必须满足幂等性**：

- 重复运行同一脚本，使用相同参数，不应产生副作用
- 对于修改操作，应先检查目标状态，如果已经是期望状态则直接返回成功
- 非幂等操作（如发送通知、创建发布）需要明确标记，并提供 `--dry-run` 选项

**幂等性示例（好的做法）**：
```python
def update_version_in_file(filepath: Path, new_version: str) -> bool:
    content = filepath.read_text()
    if f'version = "{new_version}"' in content:
        print(f"  {filepath} 已是目标版本，跳过")
        return False
    # ... 执行修改
    return True
```

### 5. 错误处理

- 脚本遇到错误时应返回非零退出码
- 使用 `try/except` 捕获预期异常，并给出清晰的错误信息
- 不使用空的 `except:` 吞掉所有异常
- 关键操作前后打印日志，便于排查问题

### 6. 不导入项目内部包

scripts/ 中的脚本**不应该**依赖 `libs/` 或 `apps/` 中的内部包（除非脚本明确需要操作项目代码，且在文档中说明）。原因：

- 脚本需要在项目未完全安装时也能运行（如 CI 环境）
- 减少耦合，脚本可以独立执行
- 如确需使用项目代码，通过 `sys.path` 方式显式添加并说明原因

## 现有脚本示例

| 文件名 | 类型 | 说明 | 幂等 |
|--------|------|------|------|
| `bump_version.py` | 发布类 | 升级所有子项目版本号，更新 changelog | 是 |
| `generate_changelog.py` | 发布类 | 从 Git commits 生成 CHANGELOG.md | 是 |
| `init_dev_env.py` | 环境类 | 初始化开发环境，安装 git hooks 等 | 是 |
| `run_benchmarks.py` | 测试类 | 运行性能基准测试并生成报告 | 是 |
| `migrate_toml_schema.py` | 迁移类 | 将旧版 TOML 配置迁移到新版 Schema | 否（需备份） |

## 脚本目录结构

简单脚本直接放在 `scripts/` 根目录。如果脚本复杂到需要多个文件，创建子目录：

```
scripts/
├── README.md                    # 本文件
├── bump_version.py              # 单文件脚本
├── generate_changelog.py
└── complex_release_tool/        # 复杂脚本的子目录
    ├── __init__.py
    ├── main.py
    ├── git_utils.py
    └── README.md                # 复杂脚本的单独说明
```

## 执行脚本

通过 PDM 运行脚本（推荐，自动加载项目环境）：

```bash
# 运行脚本
pdm run python scripts/bump_version.py --minor

# 带参数的示例
pdm run python scripts/generate_changelog.py --since v0.1.0
```

或直接运行（需要激活虚拟环境）：

```bash
# Windows PowerShell
.\.venv\Scripts\python.exe scripts\bump_version.py --patch

# Linux/macOS
./.venv/bin/python scripts/bump_version.py --patch
```

## 注意事项

1. **临时脚本用完即删**：不要保留 `test.py`、`temp.py` 这类无意义脚本，用完及时删除或移入 `attic/`
2. **敏感信息不提交**：包含密钥、密码、生产环境地址的脚本不要提交到 Git
3. **跨平台兼容**：脚本应考虑 Windows、Linux、macOS 兼容性（使用 `pathlib` 处理路径）
4. **日志清晰**：脚本运行过程应有清晰的输出，告诉用户正在做什么、结果如何
5. **测试验证**：重要脚本（特别是发布、迁移类）在使用前应在测试分支验证
