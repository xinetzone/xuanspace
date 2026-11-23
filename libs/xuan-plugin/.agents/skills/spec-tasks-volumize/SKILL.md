---
name: "spec-tasks-volumize"
version: "1.0.0"
description: "xuan-plugin 项目本地技能：spec 的 tasks.md 长台账无损分卷 + 归档快照链 + 保真复验。把单文件台账按语义增量切成多卷、源文件原地降级为可重算的索引页，用 SHA-256 重建断言与双域指纹证明正文一字未改。当需要拆分/追加 spec 任务台账、为新增量建立 rN 归档、或复验索引与分卷是否已漂移时使用。"
---

# spec 任务台账无损分卷（xuan-plugin 本地版）

本技能是根工作区 `md-lossless-volumize` 在 xuan-plugin 项目内的**自包含萃取版**：脚本与模板逐字节复制（`scripts/volumize.py` 为 `d44e6c58…`、`scripts/selftest.py` 为 `97485779…`、`templates/index.md.tmpl` 为 `33a77b90…`），另外补入本项目从 plugin-kernel 案例中萃取的快照链纪律。

核心不是拆分本身，而是**拆分之后仍然可证伪**：任何一次后续追加或手改，都能在一条命令内被判定为「仍在保真」或「已漂移」。

## 1. 适用与不适用

适用：

- `xuanspace/.trae/specs/*/tasks.md` 这类结构化任务台账超过约 600 行，且还会持续追加；
- 要求「正文一字不改」并留下可机检证据（跨通道、跨人交接、SMB 缓存不可信）；
- 外部链接、CI、spec 约定都指向 `tasks.md` 这个固定文件名，所以源文件只能原地变成索引页。

不适用：

- 需要改写正文（合并重复、调整层级）→ 属文档重构，不走本流程；
- 只求内容不丢、允许重排 → 用 `okf-atomize-migrate`；
- 纯删除/迁移取证 → 用 `assertive-fs-ops`。

## 2. 五条不变量（前四条继承上游，P1 为本项目新增）

- **I1 横幅可逆**：每卷页首 N 行由 `banner` 模板统一渲染；剥离（去至多一个尾换行 → 切行 → 丢前 N 行）必须精确还原原行切片。N 只由模板长度导出，禁止三处各写一个数字。
- **I2 覆盖完备**：区间首段 `0`、严格递增、末段 `=总行数`，无空卷；去横幅拼接的 SHA-256 必须等于冻结件（有 `rewrites` 时等于改指归一化后的冻结件）。
- **I3 指纹分域**：`body_sha256`（去横幅正文，对冻结件）与 `file_sha256`（含横幅落盘文件，抓手工改动）是两域，索引必须分列。
- **I4 索引派生**：索引每一格（区间/字节/任务数/指纹/状态）都由分卷重算；`check` 在内存重渲染后与磁盘逐字节比对，不相等即失败。索引页不承载任何手写事实。
- **P1 快照只增不改**：`archive/<name>-rN` 一旦落盘，**不得原地重冻结**；内容有变就 `+1` 新目录。重冻结会同时重写 `MANIFEST.sha256`，让历史失去可证伪性（本案例已实地发生，见 `references/case-2026-11-23-plugin-kernel.md` §1.3）。

配套口径：

- **行数只有一种口径**：一律用脚本/`split("\n")` 计数（含末尾空段）。编辑器与 `Get-Content` 给出的是换行符数，比脚本少 1，不可混用。
- **lock 与索引同生共死**：`.volumize.lock.json` 必须与分卷、索引页一起提交；删 lock 等于放弃复验能力。

## 3. 标准流程

1. **冻结**：`freeze` 把原件复制到 `<archive>/raw/<basename>` 并写 `MANIFEST.sha256`（`sha256sum -c` 兼容，双空格）。归档名遵循 `tasks-YYYYMMDD` / `tasks-YYYYMMDD-rN`。
2. **定界**：只在**围栏代码块之外**、文档既有语义标题处落刀（`# 增量N`、`## Task <id>`）。锚点必须唯一命中；歧义时显式给 `occurrence`，不允许 `startswith` 取首个命中蒙混。
3. **切卷**：`split` 写分卷 → 断言重建逐字节等价 → 落 lock（双域指纹）→ 渲染索引。切完后源文件只剩索引，原文只在 `archive/raw`。
4. **改指**（仅当卷目录层级变化）：`relink --old A --new B` 对全部卷替换、重算指纹、重渲染索引，并做归一化等价断言；零命中视为错误。
5. **门禁**：`check` = `verify` + 索引重渲染比对，幂等。**每次编辑分卷后、每次交付前都必须跑**，退出码 0 才算交付。

## 4. 快速开始

Linux（规范执行环境，服务器直连）：

```bash
cd /media/pc/data/ai/notebook/dev/libs/AI/projects/xuanspace/libs/xuan-plugin
python .agents/skills/spec-tasks-volumize/scripts/volumize.py check \
  --plan .agents/archive/2026-11-23-plugin-kernel-tasks/plans/plan-r4.json
```

Windows / SMB 侧（只读复验；`plan` 内的 `/media/pc/data/ai` 需换成 `//10.16.11.3/ai`）：

```powershell
py -3.14 "<repo>\.agents\skills\spec-tasks-volumize\scripts\volumize.py" check --plan <改动过路径前缀的 plan.json>
```

自检（验证本副本可用，不触碰真实台账）：

```powershell
py -3.14 "<repo>\.agents\skills\spec-tasks-volumize\scripts\selftest.py"   # 期望 SUMMARY 16/16 PASS
```

## 5. plan.json 契约

必填 `source` `archive` `out_dir` `volumes` `banner`；可选 `date` `rewrites` `index` `guards` `tasks`。

- `volumes[]{file,topic,anchor,occurrence?}`：`anchor:null` 表示从首行起；第 n 卷终点 = 第 n+1 卷起点。
- `banner` 行数即 N；占位符：`{source_name} {start} {end} {date} {k} {index_rel} {prev} {next} {topic}`（`{k}`=N−1）。
- `index{path,template}`：模板占位符 `{{source_name}} {{source_sha256}} {{source_bytes}} {{source_lines}} {{volume_count}} {{task_count}} {{volume_table}} {{task_table}} {{date}} {{archive_rel}} {{rewrites_note}}`；未定义占位符会报错。
- `guards.forbid_body_contains`：命中即拒切，用于识别「源已被索引覆盖」。

可运行样例：`templates/plan.example.json`（即本项目 r4 实况计划，11 卷）。

## 6. 退出码与判据

| 退出码 | 含义 | 典型输出 |
|---|---|---|
| 0 | 通过 | `BYTE_EXACT True` / `LOCK_OK` / `INDEX_OK` |
| 1 | 保真失败 | `RAW_SHA ≠ REBUILT_SHA`、`LOCK_DRIFT`、`INDEX_DRIFT L<n>` |
| 2 | 计划或前置错误 | 锚点条数不符、区间不合法、源已被覆盖 |

`LOCK_MISSING` 是警告不改变退出码——但交付时缺 lock 视为不合格。

## 7. 已知限制（上游 v1.0.1 原样继承，未擅自改动）

1. `split` 生成 `task_ids` 时固定使用默认标题正则，`tasks.heading_regex` 自定义时不生效（本项目用默认正则，暂无影响）。
2. `relink` 对 plan 文件的 `rewrites` 回写发生在命令成功之后；中途失败则 `rewrites` 不落盘。

## 8. 出处

- 上游技能：`/media/pc/data/ai/.agents/skills/md-lossless-volumize`（v1.0.1）
- 本案例事实与洞察：`references/case-2026-11-23-plugin-kernel.md`
- 快照台账（含 plan 副本与源哈希）：`.agents/archive/2026-11-23-plugin-kernel-tasks/ARCHIVE.md`