---
id: "xuanspace-protocols-index"
x-toml-ref: "toml/.agents/protocols/README.toml"
---

# 协作协议

Xuanspace 项目 AI 智能体协作协议集合。

## 协议列表

| 协议 | 文件 | 说明 |
|---|---|---|
| 工作区发现 | `workspace-discovery.md` | 五步发现流程，从任意位置定位工作区 |
| 提示词自举 | `prompt-bootstrap.md` | 一句话装载，零配置接入 |
| 会话启动 | `session-startup.md` | 新会话启动检查清单 |
| 任务交接 | `task-handoff.md` | 任务上下文传递规范 |

## 核心原则

1. **最小侵入**：不修改系统配置、不安装未授权的包
2. **幂等安全**：重复执行结果一致，已在工作区内则跳过
3. **渐进加载**：按需读取规范，不扫描整个文件系统
4. **内容敏感**：区分公开/私域内容，走不同工作流