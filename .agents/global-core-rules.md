# 全局核心规则

本文件包含玄境项目所有智能体必须遵守的基础规则。

## 1. 启动协议

所有任务必须严格遵循根目录 `AGENTS.md` 中的启动协议（步骤1-4），包括：
- 步骤1：读取 AGENTS.md 全文
- 步骤2：按上下文路由表确定规范
- 步骤2.0：子项目预检（如在 apps/*/ 或 libs/*/ 下）
- 步骤3：读取对应规范（按需读取，不要一次加载全部）
- 步骤3.5：自检清单
- 步骤4：执行任务

## 2. 内容敏感度分流

### 公开内容（Public）
- 公开发布的开源代码、官方文档、示例项目
- **工作流**：标准工作流，可使用完整规划流程

### 私域内容（Private）
- 个人笔记、内部讨论、用户明确指定目录的内容
- **工作流**：跳过规划，直接在目标目录执行
- 若用户已明确指定目标路径，直接在该路径下操作
- 不确定时默认按私域处理或向用户确认

## 3. Python 3.14.6+ 严格要求

- 所有 Python 子项目 `requires-python>=3.14.6`
- 可以使用 Python 3.14 新特性（如类型参数、更严格的类型检查、free-threaded模式、t-strings等）
- 代码格式化目标版本为 py314（见 pyproject.toml）
- 遇到 Python 版本问题时，提示用户使用 xs doctor 检查环境

## 4. 多包管理器支持

- **不强制 PDM**：支持 pdm、uv、pip 三种包管理器
- 根据用户习惯或环境自动选择，不要强迫用户使用特定工具
- 安装依赖示例：
  ```bash
  # PDM
  pdm add <package>
  # pip
  pip install <package>
  # uv
  uv pip install <package>
  ```
- pyproject.toml 使用 PEP 621 标准格式，兼容所有工具

## 5. 文档 YAML/TOML 二分法

所有 Markdown 文档遵循内容-元数据分离原则：
- **YAML frontmatter**：仅保留 id、x-toml-ref、source、version 四个字段
- **TOML 元数据**：存放于 `.meta/toml/` 镜像路径下
- 详细规范见 [rules/frontmatter.md](rules/frontmatter.md)
- 当前阶段（Task 20之前）不要添加 YAML frontmatter

## 6. 三阶段递进工作法

处理问题时遵循三阶段递进：

1. **修复（Fix）**：先解决当前的具体问题
2. **预防（Prevent）**：分析问题根因，添加必要的检查或文档避免再次发生
3. **闭环（Close）**：验证修复有效，必要时添加测试或更新文档

不要过度设计，简单问题直接修复即可。

## 7. 按需读取规范

- **不要**一次性读取 .agents/ 下所有规范文件
- 根据 `context-routing.md` 只读取与当前任务相关的规范
- 简单任务（如修改单个文件bug）可只读取核心规则后直接执行
- 涉及新领域（如C++扩展、文档构建）时再读取对应规范

## 8. 其他基础规则

- **路径引用**：Markdown 中使用相对路径，禁止 file:/// 绝对路径
- **提交规范**：Conventional Commits，type(scope): subject，中文主体
- **代码风格**：遵循 ruff + black + isort 配置（行宽120，py314）
- **子项目创建**：必须使用 xs new 命令从模板创建，不要手动复制
- **vendor 目录**：只读，不直接修改第三方代码
- **attic 目录**：归档用，不删除废弃内容，移动到此处即可
