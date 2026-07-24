---
id: "submodule-placement-checklist"
source: "retrospective-xuanspace-mono-repo-20260724/insight-extraction.md#洞察2"
---

# 子模块放置检查清单（Submodule Placement Checklist）

> 在 `git submodule add` 之前逐项确认，避免目录语义与 Git 跟踪规则的隐性冲突。

## 放置前检查（必做）

- [ ] **目标路径不被 `.gitignore` 排除**：执行 `git check-ignore -v <目标路径>` 确认无输出
- [ ] **目录语义与 Git 语义一致**：如 `vendor/` 被 `.gitignore` 排除但需要放置子模块，需添加 `!` 白名单
- [ ] **子模块 URL 可访问**：确认远程仓库存在且当前用户有读取权限
- [ ] **分支策略明确**：子模块跟踪的分支（默认 main）与主仓库的预期一致

## `.gitignore` 白名单规则

当需要在不被跟踪的目录下放置子模块时：

```gitignore
# 排除 vendor 目录内容
vendor/*
# 但保留子模块的 gitlink（关键！）
!vendor/*/
!vendor/*/.git
```

## 放置后验证（必做）

- [ ] `.gitmodules` 文件已正确生成，路径和 URL 正确
- [ ] `git submodule status` 显示正确的 commit 指针
- [ ] 主仓库 `git status` 显示子模块为正常状态（非 `modified` 或 `untracked`）
- [ ] 子模块目录在目标路径下可见且可读取

## 推荐目录

| 目录 | 用途 | .gitignore 状态 |
|------|------|----------------|
| `projects/` | 自建项目子模块 | 默认跟踪（推荐） |
| `vendor/` | 第三方依赖子模块 | 需白名单 |
| `external/` | 外部依赖 | 通常被排除，不推荐放置子模块 |

## 反模式

- 不检查 `.gitignore` 就直接 `git submodule add`
- 子模块添加后才发现未被跟踪，通过修改 `.gitignore` 后重新添加
- 同一目录既放被忽略的第三方源码，又放需要跟踪的子模块