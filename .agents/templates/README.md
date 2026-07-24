---
id: "xuanspace-templates-index"
x-toml-ref: "toml/.agents/templates/README.toml"
---

# 模板库

Xuanspace 项目模板索引。

## 模板位置

模板文件统一存放在 `tools/templates/` 目录下，按项目类型分类：

| 类型 | 目录 | 说明 |
|---|---|---|
| Python | `tools/templates/python/` | 纯 Python 库/应用模板 |
| Native | `tools/templates/native/` | C++ 原生扩展模板（CMake + pybind11） |
| Static | `tools/templates/static/` | 静态前端项目模板 |

## 使用方式

```bash
# 创建 Python 项目
xs new --type python my-lib

# 创建 Python 应用
xs new --type python --app my-app

# 创建原生扩展项目
xs new --type native my-ext

# 创建静态项目
xs new --type static my-site
```

## 模板内容

每个模板包含：

- `pyproject.toml` — PEP 621 项目配置（Python/Native 类型）
- `README.md` — 项目说明框架
- `CHANGELOG.md` — 变更日志模板
- `src/<package>/` — 源代码目录骨架
- `tests/` — 测试目录骨架
- `CMakeLists.txt` — CMake 构建配置（Native 类型）
- `CMakePresets.json` — CMake 构建预设（Native 类型）

## 自定义模板

在 `tools/templates/` 下添加新目录即可扩展模板类型。`xs new --type` 会自动发现新模板。