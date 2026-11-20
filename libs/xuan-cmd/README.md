# xuan-cmd

命令装配与工程路径解析工具库，萃取自 xmapps 交付包的 `apps/demo/tasks/common.py`
（invoke 任务通用件），面向"把 CLI 参数拼成命令字符串 + 按约定定位工程目录"这一类需求。

## 功能

| 模块 | 公开 API | 作用 |
|------|----------|------|
| `xuan_cmd.quoting` | `quote_arg`、`format_command` | 按平台给单个参数加引号、把参数序列拼成命令字符串 |
| `xuan_cmd.paths` | `resolve_dir`、`resolve_work_dir` | 显式值 > 环境变量的目录解析；工作目录默认值回退 |
| `xuan_cmd.projects` | `list_projects`、`locate_project`、`DEFAULT_MARKER` | 按标记文件发现并定位工程子目录 |
| `xuan_cmd.errors` | `XuanCmdError` | 统一错误类型，调用方映射为自身 CLI 异常 |

本库**不依赖 invoke**：任务框架只负责执行，命令字符串由 `format_command` 生成，
边界清晰、可独立测试。

## 安装

```bash
pip install -e ".[test]"
```

## 快速使用

```python
from pathlib import Path

from xuan_cmd import format_command, list_projects, locate_project, resolve_dir

sdk = resolve_dir(None, envvar="XS_SDK_ROOT", label="SDK 根目录")
app = locate_project(Path("apps"), "cpp_deploy", label="应用")
command = format_command(["cmake", "-S", app, "-B", app / "build", f"-DXS_SDK_ROOT={sdk}"])

for project in list_projects(Path("apps")):
    print(project.name)
```

配合 invoke 时，执行侧只需一行：

```python
from invoke import task
from xuan_cmd import format_command


@task
def build(c, name):
    c.run(format_command(["cmake", "--build", name]), echo=True)
```

## 设计边界

- **只做无状态纯函数**：不读业务配置、不落盘、不吞异常；错误统一抛 `XuanCmdError`。
- **平台差异收敛在 `quoting`**：Windows 走 `subprocess.list2cmdline`，其余平台走 `shlex.quote`。
- **不复制目录清单**：`list_projects` 以标记文件为唯一事实来源，避免第二份应用表。
- **不迁入领域逻辑**：SDK 语义校验、glibc/RUNPATH 产物检查等仍留在原工程。
- **与 `xuan-core` 的分工**：`xuan-core.path` 回答"本 monorepo 根在哪"（自定位）；
  `xuan-cmd.paths` 回答"调用方给的目录/环境变量指向哪"（外部输入解析）。

## 版本

0.1.0
