---
id: "xuanspace-workspace-discovery"
x-toml-ref: "toml/.agents/protocols/workspace-discovery.toml"
---

# 工作区发现协议

## 协议目标

从任意位置自动定位 Xuanspace 工作区根目录，无需用户手动配置路径。

## 五步发现流程

### 步骤 1：当前目录检查

检查当前目录是否包含 `pyproject.toml` 且同时存在 `apps/` 和 `libs/` 目录。

```
✓ 当前目录 = 工作区根目录 → 直接使用
↓ 否则进入步骤 2
```

### 步骤 2：向下递归

从当前目录向上逐级查找，直到找到同时满足以下条件的目录：

- 存在 `pyproject.toml`
- 存在 `apps/` 目录
- 存在 `libs/` 目录

### 步骤 3：根目录标记

工作区根目录的唯一标识：

| 标识文件 | 必需 | 说明 |
|---|---|---|
| `pyproject.toml` | ✓ | PEP 621 项目配置 |
| `apps/` | ✓ | 应用项目目录 |
| `libs/` | ✓ | 库项目目录 |
| `AGENTS.md` | 推荐 | AI 智能体路由入口 |

### 步骤 4：环境验证

```
xs init  → 验证 Python ≥ 3.13、Git 可用
xs doctor → 完整环境诊断报告
```

### 步骤 5：就绪报告

```
✓ 工作区根目录: /path/to/xuanspace
✓ Python 3.13.x
✓ 可用命令: xs list, xs build, xs deps check, xs docs build
```

## 实现

`xs` CLI 内置 `config.find_workspace_root()` 函数，所有命令自动调用此协议定位工作区。

## 与 SpecWeave 的差异

Xuanspace 简化版：
- 工作区标识：`pyproject.toml` + `apps/` + `libs/`（SpecWeave 使用 `.agents/` + `apps/`）
- 不依赖 SpecWeave 的 `.agents/scripts/` 工具链
- 发现逻辑内置于 `xs` CLI 而非独立脚本