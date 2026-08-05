# .agents/ — caffe-ffi 智能体规范目录

本目录存放项目级 AI 智能体配置和规则。完整规范体系请参考上游 SpecWeave 工作区 `.agents/` 目录。

## 结构

```
.agents/
└── README.md          # 本文件（目录说明与约定）
```

## 路由

- 项目级 AI 智能体路由入口为根目录 [AGENTS.md](../AGENTS.md)，承载技术栈、目录结构、开发约定与关键约束
- 通用方法论、角色、工作流、协议等规范直接引用上游 SpecWeave，不在此重复定义
- 如需新增项目特定的规则/脚本/模板，在本目录创建对应子目录（`rules/`、`scripts/`、`templates/`）并在此登记

## 约定

- 本目录遵循 SpecWeave 规范体系，作为独立 git submodule 入口，仅维护 caffe-ffi 项目级增量规范
- 上游方法论入口：SpecWeave `.agents/commands/`（复盘/洞察/萃取/原子提交/CI 检查等）
- 关键项目约束（构建选项、COW 语义、dtype 守卫、内存安全、测试环境）见根目录 [AGENTS.md](../AGENTS.md)「关键约束」与「构建选项」
- 新增 `.agents/scripts/` 脚本前先检查 SpecWeave `.agents/scripts/lib/` 共享库，避免重复实现