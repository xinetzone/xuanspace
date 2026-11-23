# xuan-plugin 项目智能体资产

本目录是 xuan-plugin 的项目内 AI 资产位，只放**可复用、可机检**的技能与归档台账，不复制根工作区的正文载荷。

## 目录

| 路径 | 用途 |
|---|---|
| `skills/spec-tasks-volumize/` | spec 长台账无损分卷 + 快照链归档 + 保真复验（脚本、模板、案例洞察） |
| `archive/2026-11-23-plugin-kernel-tasks/` | plugin-kernel `tasks.md` 四代快照（r1–r4）的哈希台账、plan 副本与复验命令 |

## 技能速用

```powershell
# 复验：分卷 + lock + 索引 三者是否仍自洽（当前 spec 实况）
py -3.14 .\skills\spec-tasks-volumize\scripts\volumize.py check `
  --plan .\archive\2026-11-23-plugin-kernel-tasks\plans\plan-r4.json
# 自检：本副本可用性（tempdir 全链路，不触碰真实台账）
py -3.14 .\skills\spec-tasks-volumize\scripts\selftest.py
```

Windows 侧直接跑 `check` 前须把 plan 里的 `/media/pc/data/ai` 前缀换成 `//10.16.11.3/ai`；规范执行环境是服务器 Linux 侧（`/media/pc/data/ai/...`）。

## 来源与口径

- 脚本/模板逐字节复制自根工作区 `.agents/skills/md-lossless-volumize`（v1.0.1），哈希见 `skills/spec-tasks-volumize/SKILL.md` 开头。
- 本次实测：`selftest` 16/16 PASS；当前 spec `check` 退出码 0（`BYTE_EXACT True`、`LOCK_OK volumes=11`、`INDEX_OK`）。
- 技能 `SKILL.md` 顶部的 YAML 仅为 Skill 规范所需的名称/描述元数据；本项目「当前阶段不加文档 frontmatter」的规则仍适用于其余 Markdown。

## 纪律

新增快照一律 `-rN+1`，不覆盖既有目录；分卷编辑后立刻 `check`；lock 与索引随分卷同批提交。详见 `skills/spec-tasks-volumize/SKILL.md` 与案例 `references/case-2026-11-23-plugin-kernel.md`。