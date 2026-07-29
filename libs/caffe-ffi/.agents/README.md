# .agents/ — caffe-ffi 智能体规范目录

本目录存放项目级 AI 智能体配置和规则。完整规范体系请参考上游 SpecWeave 工作区 `.agents/` 目录。

## 结构

```
.agents/
└── README.md          # 本文件
```

## 约定

- 本目录遵循 SpecWeave 规范体系，项目级规范通过根目录 [AGENTS.md](../AGENTS.md) 路由
- 如需项目特定的规则/脚本/模板，在此目录下创建对应子目录（rules/、scripts/、templates/）
- 通用规范直接引用上游，不重复定义
