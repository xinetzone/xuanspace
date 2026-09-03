# 上下文路由表

根据任务类型确定需要读取的规范文件。**按需读取，不要一次加载全部。**

## 必读基础规范

所有任务必读：
- [../AGENTS.md](../AGENTS.md) - 根目录智能体入口（步骤1已读取）
- [global-core-rules.md](global-core-rules.md) - 全局核心规则

## 任务类型→规范映射

| 任务类型 | 必读规范 | 可选参考 |
|---|---|---|
| **环境配置/快速开始** | [ONBOARDING.md](ONBOARDING.md) | - |
| **创建新子项目** | [ONBOARDING.md](ONBOARDING.md) | [../tools/templates/](../tools/templates/) |
| **编写/修改 Markdown 文档** | [rules/frontmatter.md](rules/frontmatter.md) | [../docs/README.md](../docs/README.md) |
| **修改 Python 代码** | - | [../pyproject.toml](../pyproject.toml)（代码风格配置） |
| **编写/修改 C++ 原生扩展** | - | [../libs/xuan-ext-demo/](../libs/xuan-ext-demo/)（示例项目）、[../tools/templates/native/](../tools/templates/native/) |
| **修改 xs CLI 工具** | - | [../tools/xs/](../tools/xs/)、[../apps/xs-cli/](../apps/xs-cli/) |
| **构建问题/编译错误** | [ONBOARDING.md](ONBOARDING.md)（xs doctor） | - |
| **修改 .agents/ 规范本身** | [README.md](README.md) | - |
| **子项目内任务** | 该子项目的 README.md、pyproject.toml/CMakeLists.txt | 该子项目 AGENTS.md（若有） |
| **静态HTML项目** | - | [../tools/templates/static/](../tools/templates/static/) |
| **Sphinx文档构建** | - | [../doc/conf.py](../doc/conf.py)、[../doc/README.md](../doc/README.md) |

## 子项目路由

当任务涉及特定子项目时，按以下规则处理：

### apps/ 下的项目
- 先读取 `apps/<project>/README.md`
- 读取 `apps/<project>/pyproject.toml`（Python项目）
- 如有 `apps/<project>/AGENTS.md`，优先遵循该文件

### libs/ 下的项目
- 纯Python库：读取 `libs/<project>/README.md` 和 `pyproject.toml`
- C++原生扩展：额外读取 `libs/<project>/CMakeLists.txt`
- 参考同类型现有项目：xuan-core（Python）、xuan-ext-demo（native）

### tools/ 下的项目
- tools/xs/：xs CLI 核心实现
- tools/templates/：项目模板，新增模板时参考现有模板结构

## 不需要额外读取规范的场景

以下简单任务可在读取完基础规范后直接执行：
- 修改单个 Python 文件的 bug
- 更新单个文档的错别字
- 添加简单的测试用例
- 运行现有命令（xs list、xs doctor 等）

## 需要完整规划的场景

以下任务建议先规划再执行：
- 创建新的子项目
- 修改构建系统（CMakeLists.txt、pyproject.toml 重大变更）
- 重构多个文件
- 添加新的功能模块
