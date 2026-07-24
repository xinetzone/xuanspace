# .agents/ 规范目录索引

本目录包含玄境（Xuanspace）项目 AI 智能体协作的所有规范文件。所有文件按需读取，不要一次性加载全部。

## 规范文件列表

| 文件 | 用途 | 适用场景 |
|---|---|---|
| [ONBOARDING.md](ONBOARDING.md) | 入门指南 | 第一次接触项目、快速上手、查找常用命令 |
| [global-core-rules.md](global-core-rules.md) | 全局核心规则 | 所有任务必须遵守的基础规则 |
| [context-routing.md](context-routing.md) | 上下文路由表 | 根据任务类型确定需要读取哪些规范 |
| [rules/frontmatter.md](rules/frontmatter.md) | 文档元数据规范 | 编写Markdown文档、处理YAML/TOML元数据时 |

## 目录结构

```
.agents/
├── README.md              # 本文件 - 规范目录索引
├── ONBOARDING.md          # 入门指南
├── global-core-rules.md   # 全局核心规则
├── context-routing.md     # 上下文路由表
└── rules/                 # 具体规则目录
    ├── .gitkeep
    └── frontmatter.md     # 文档元数据二分法规范
```

## 使用方式

1. 从根目录 `AGENTS.md` 启动，遵循启动协议
2. 根据任务类型查阅 `context-routing.md`
3. 按需读取对应的规范文件
4. 子项目内的任务优先读取子项目自身的 README 和配置

## 注意事项

- 所有规范文件均为中文编写
- 内容精简实用，只包含 monorepo 项目必需的规范
- 不复制 SpecWeave 的完整规范体系，保持轻量
- 规范更新请遵循 Conventional Commits 提交规范
