# 快速上手

本页用 5 分钟带你完成从创建 Bundle 到校验信任等级的完整一次流程。所有命令均要求已安装 `okf`（见[首页](index)）。

## 1. 创建 Bundle 骨架

```powershell
okf init my-bundle
```

该命令会在 `my-bundle/` 下创建如下结构：

```text
my-bundle/
├── index.md       # 概念索引（首页）
├── log.md         # 变更日志
├── concepts/      # 概念文件
├── playbooks/     # 剧本（playbook）
└── references/    # 引用材料
```

对于已存在的目录，`init` 会跳过而不覆盖。

## 2. 写入第一个概念

在 `my-bundle/concepts/metrics/` 下新建 `active_users.md`：

```markdown
---
type: Metric
title: Active Users
description: Monthly active users count
verified:
  by: metric-collector
  at: 2026-01-01T08:00:00
---

# Active Users

Monthly active users count.
```

要点：frontmatter 中 `type` 是必填字段，其余为可选。正文写在第二个 `---` 之后。

## 3. 查看 Bundle 内容

```powershell
# 列出所有概念
okf list my-bundle
# 输出形如：concepts/metrics/active_users<TAB>Metric<TAB>Active Users

# 查看单个概念详情
okf inspect my-bundle concepts/metrics/active_users
```

`inspect` 会打印概念的类型、标题、描述、标签、完整 frontmatter 以及正文前 5 行。

## 4. 生成索引

```powershell
okf index my-bundle
```

该命令扫描 Bundle 内的全部概念，按 `type` 分组写回 `my-bundle/index.md`。

## 5. 一致性校验

```powershell
okf validate my-bundle
```

通过时输出 `Result: PASS`；存在错误时输出 `Result: FAIL` 并以退出码 1 结束。详见[一致性校验](conformance)。

## 6. 查看信任与保鲜状态

```powershell
# 查看单个概念
okf trust my-bundle concepts/metrics/active_users

# 查看全部概念
okf trust my-bundle
```

输出形如 `信任等级=human_reviewed, 保鲜状态=保鲜中`。推导规则见[信任与保鲜](trust)。

## 下一步

- 理解 [Bundle 与概念](bundle) 的目录结构与 frontmatter 字段
- 查阅 [CLI 命令参考](commands) 了解全部命令与选项
- 阅读 [架构原理](architecture) 理解背后的插件化设计