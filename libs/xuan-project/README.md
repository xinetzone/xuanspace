# xuan-project

工作区项目发现与 `pyproject.toml` 解析工具库，萃取自 `xs` CLI 的
`tools/xs/src/xs/config.py` 与 `tools/xs/src/xs/discovery.py`。面向
"扫描 monorepo 子项目、判定项目类型、构建依赖图、计算受影响项目"这一类需求。

## 功能

| 模块 | 公开 API | 作用 |
|------|----------|------|
| `xuan_project.config` | `find_workspace_root`、`load_pyproject` | 按 `pyproject.toml` + `apps/`/`libs/` 向上定位工作区根；解析 TOML |
| `xuan_project.config` | `get_build_backend`、`get_project_dependencies`、`get_python_version_requirement` | 从已解析的 TOML 数据中提取指定字段 |
| `xuan_project.config` | `ProjectType`、`ProjectInfo` | 项目类型枚举与信息数据类 |
| `xuan_project.config` | `get_python_version`、`check_python_version` | 当前解释器版本读取与最低版本判断 |
| `xuan_project.discovery` | `detect_project_type`、`discover_projects` | 按构建后端/静态页判定类型；扫描 `apps/`、`libs/`、`tools/` |
| `xuan_project.discovery` | `resolve_dynamic_version` | 解析 setuptools-scm / setuptools 动态版本 |
| `xuan_project.discovery` | `build_dependency_graph`、`find_affected_projects` | 名称归一化后的依赖图；含反向依赖传播的受影响项目 |
| `xuan_project.discovery` | `find_project_by_name`、`filter_projects` | 归一化名称查找；按类型/目录过滤 |

本库**不依赖 typer/rich**：CLI 只负责渲染，发现与解析逻辑可独立测试。

## 安装

```bash
pip install -e ".[test]"
```

## 快速使用

```python
from pathlib import Path

from xuan_project import discover_projects, find_affected_projects, find_workspace_root

root = find_workspace_root(Path.cwd())
projects = discover_projects(root)

affected = find_affected_projects(root, [Path("libs/xuan-core/src/xuan_core/__init__.py")])
for project in affected:
    print(project.name, project.project_type, project.version)
```

## 设计边界

- **只做无状态纯函数**：不读业务配置、不落盘、不吞解析异常（`detect_project_type`
  对坏 TOML 退化为 `PYTHON` 是既有语义，保留）。
- **工作区约定集中在本库**：`apps/`、`libs/`、`tools/` 的扫描规则与根目录标识
  （`pyproject.toml` + `apps/` + `libs/`）在此单一维护，调用方不再各自实现。
- **与 `xuan-core` 的分工**：`xuan-core.path` 回答"本 monorepo 根在哪"（自定位）；
  本库回答"给定起点如何发现子项目并计算影响面"。
- **与 `xuan-cmd` 的分工**：`xuan-cmd` 负责命令串装配与外部目录参数解析；
  本库只处理 monorepo 结构。

## 版本

0.1.0
