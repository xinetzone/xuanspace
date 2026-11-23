# 案例：plugin-kernel tasks.md 快照链归档与无损分卷（2026-11-23）

> 本文是 `spec-tasks-volumize` 技能的洞察底稿，按七概念「知识沉淀」链路 R → I → E 组织。
> 事实层每条都可用文末命令现场复核；案例档案的记载与他处记载冲突时，以磁盘实测为准并显式标注。

## 1. 事实层（R）

### 1.1 快照链

四个归档目录同属根工作区 `.agents/archive/`，均含 `MANIFEST.sha256`（格式 `<sha>  raw/tasks.md`，双空格，`sha256sum -c` 兼容）；r2 及之后另含 `plan.json`。

| 快照 | 目录 mtime | raw 字节 | `\n` 数 | `split("\n")` 行 | `## Task` 标题 | raw SHA-256（前 12 位） | 清单一致 |
|---|---|---:|---:|---:|---:|---|:---:|
| r1 | 2026-11-23 09:40:49 | 101461 | 1359 | 1360 | 24 | `39b4c9e99f05` | ✅ |
| r2 | 2026-11-23 10:08:42 | 135512 | 1778 | 1779 | 31 | `b6731e8a3f8e` | ✅ |
| r3 | 2026-11-23 11:02:18 | 140655 | 1851 | 1852 | 33 | `af0546dca6b7` | ✅ |
| r4 | 2026-11-23 11:24:08 | 143421 | 1890 | 1891 | 34 | `28e0c1e77ab6` | ✅ |

> r4 不在用户点名的三个目录内，是本次学习时按 `.trae/specs/plugin-kernel-extraction/tasks.md` 索引页的 `archive_rel` 反查发现的续作快照；纳入本表是为了让「现状」与「台账」对得上。

增量形态（`git diff --no-index --numstat`）：

- r1 → r2：`+424 / −5`（新增 P2-D 等增量，另有 5 行被改写）
- r2 → r3：`+74 / −1`
- r3 → r4：`+39 / −0`（纯追加）

即：**快照之间不是纯追加**，早期内容会被局部改写，所以「只保留最新件」不能替代快照链。

### 1.2 plan 契约实况

| 快照 | 冻结来源 | 卷数 | 分卷文件 |
|---|---|---:|---|
| r2 | `/tmp/vd2/tasks.md` | 8 | `00-overview` … `07-p2-c-invariants` |
| r3 | `.temp/p3-2-contract-gate/tasks.md` | 10 | 追加 `08-p2-d-doc-grouping`、`09-p3-a-contract-gate` |
| r4 | `.temp/p3-1-lock/tasks.md` | 11 | 追加 `10-p3-b-lock` |

三代 plan 的 `out_dir` 与 `index.path` **完全相同**（`.trae/specs/plugin-kernel-extraction/tasks/` 与 `tasks.md`）；`banner` 3 行、9 个占位符；索引模板 11 个占位符。

### 1.3 现状复验（本次实测）

- 索引页 `tasks.md`：6969 字节 / 65 行（`Get-Content` 与 `split` 同为 65，因该文件按模板生成且无尾空段歧义）。
- 分卷目录：11 个 `NN-topic.md` + `.volumize.lock.json`（5548 字节）。
- `volumize.py check --plan <r4>` 输出：

```text
RAW_SHA     28e0c1e77ab6…5dba
REBUILT_SHA 28e0c1e77ab6…5dba
BYTE_EXACT True
NORMALIZED_EQUAL True rewrites=0
LOCK_OK volumes=11
INDEX_OK …/.trae/specs/plugin-kernel-extraction/tasks.md
```

退出码 0。即当前磁盘状态与 lock、索引三者自洽。

### 1.4 交叉核对发现的矛盾

上游案例档案（`md-lossless-volumize/references/case-2026-11-23-spec-tasks.md`）记载：r2 为 111956 B / 1487 行 / 26 任务，其后又被原地重冻结至 116968 B、121183 B；lock 被两次删除，`check` 曾以 `RAW 678aa95d ≠ REBUILT 65bfbf97` 退出 1。

磁盘现值 r2 为 135512 B / 1778 行 / 31 任务，且 `MANIFEST.sha256` 与现值哈希一致（`b6731e8a…`）。两个数对不上，而清单与文件自洽——唯一自洽解释是：**r2 目录连同其 MANIFEST 被再次原地重写过**。归档没有做到「只增不改」。

## 2. 洞察层（I）

### I1 归档目录本身可被原地重冻结，历史因此不可证伪

- 现象：r2 现值与案例档案记载的体量、任务数均不符，但 MANIFEST 与现值哈希一致。
- 根因：`freeze` 允许对同名归档覆盖（`--force`），清单随冻结一起重写；归档名由人决定，没有「存在即拒绝」的硬约束。
- 影响：跨快照比对（r2 vs r3 的增删行数）失去审计意义；一旦发生争议，无法证明哪一版是当初交付物。
- 建议：归档目录命名自动化（`-rN` 单调递增）；对已存在目录默认拒绝写入；冻结后立即做一次「清单 → 文件」回读校验并把结果写进 `ARCHIVE.md`（本次已按此执行）。

### I2 冻结来源落在易失临时区，冻结件成了唯一可信源

- 现象：r2 源自 `/tmp/vd2/tasks.md`，r3/r4 源自 `.temp/p3-*-/tasks.md`；这些路径都不具备长期留存保证。
- 根因：分卷期间需要「可自由编辑的工作副本」，操作者顺手把工作副本放在了临时区，再把它当冻结源。
- 影响：`plan.source` 事后不可读，无法重放；只能依赖 `archive/raw` 与 lock 中的 `raw_sha`。
- 建议：把 `source` 视作「一次性输入」并在归档记录中固化其 sha、字节数、行数与获取时间；禁止从 `/tmp` 冻结；`.temp` 工作副本只作为编辑暂存，冻结前必须落到归档。

### I3 多代 plan 共用同一 out_dir，产物无法反查归档

- 现象：r2/r3/r4 的 `out_dir`、`index.path` 完全相同，分卷文件同名覆盖。
- 根因：分卷目录按 spec 固定，归档按快照递增，两者天然 1:N。
- 影响：看到一份分卷目录，无法判断它对应哪一代快照；并发执行两次 `split` 会互相覆盖且无冲突提示。
- 建议：以 lock 的 `raw_sha` 作为唯一反查键（它精确指向归档 raw）；同一 spec 同一时间只允许一个写入者；把 lock 与索引、分卷同批提交。

### I4 行数口径混用会制造「体量对不上」的假警报

- 现象：索引写 1891 行，`Get-Content` 数出 1890 行，差 1。
- 根因：`split("\n")` 把末尾换行后的空段计入，换行符计数不计；两种口径都合理，但不可混用。
- 影响：人工核对时误判为内容缺失，触发不必要的返工。
- 建议：体量一律采用脚本口径（`split("\n")`），索引表头注明口径；跨工具核对时先对齐口径再比数。

### I5 保真门禁若只在交付时跑，追加漂移会漏检

- 现象（案例档案记载）：首例交付 9 分钟后即出现追加式漂移，当时无任何机检发现。
- 根因：`check` 被当成「交付前一次性动作」，而非「每次编辑后的钩子」。
- 影响：漂移窗口内的评审、引用、下游解析都建立在错误版本上。
- 建议：编辑分卷后立刻 `check`；CI 或 pre-commit 中加同一条命令；`LOCK_MISSING` 即使退出码为 0 也应视为不合格。

## 3. 萃取层（E）

### 3.1 可复用模式：快照链式无损分卷（S1–S7）

1. **冻结登记**：复制原文到 `archive/<name>[-rN]/raw/`，写 `sha256sum -c` 兼容清单，回读校验；目录已存在则拒绝，改 `+1`。
2. **语义定界**：只在围栏代码块外、既有语义标题处落刀；锚点唯一命中，歧义显式给 `occurrence`。
3. **切分断言**：写卷 → 去横幅拼接 → SHA-256 必须等于冻结件；不等即停。
4. **双重指纹**：`body_sha256` 对冻结件、`file_sha256` 对落盘文件，写入 lock 与索引两列。
5. **索引派生**：区间、字节、任务数、任务状态、指纹全部由分卷重算；索引页不接受手写事实。
6. **编辑后门禁**：每次追加/改指后立刻 `check`（verify + 索引重渲染逐字节比对），幂等。
7. **交接三元组**：交付说明里固定给 `(archive raw_sha, lock raw_sha, index sha)`，三个值一致才算可交接。

### 3.2 反模式清单

| 编号 | 反模式 | 后果 | 替代做法 |
|---|---|---|---|
| A1 | 同名归档原地重冻结 | 历史不可证伪（I1） | 新内容 → `-rN+1` |
| A2 | 从 `/tmp` 或未登记临时区冻结 | 来源不可重放（I2） | 先落归档再切分 |
| A3 | 手改索引页补数 | `INDEX_DRIFT`，且不可复算（I4） | 改分卷后重跑 `check` |
| A4 | 删 lock 重来掩盖漂移 | 丢失双域指纹（I5） | 用 `relock` 并在记录中说明原因 |
| A5 | 两人/两进程并发 `split` 同一 out_dir | 静默互相覆盖（I3） | 单写者窗口 |
| A6 | 用编辑器行号核对体量 | 假警报（I4） | 统一 `split("\n")` 口径 |

### 3.3 迁移边界

- 适用：单文件长台账、正文必须逐字节保真、文件名不可变、需要机检证据。
- 不适用：需要重写正文（→ 文档重构）；只求内容不丢可重排（→ `okf-atomize-migrate`）；一次性删除/迁移取证（→ `assertive-fs-ops`）。
- 成本线：≤300 行且不再增长时不引入索引与 lock；600 行左右按「是否还会追加」判断。

## 4. 复验命令

```powershell
# 1) 三个（含 r4 共四个）快照的哈希与清单一致性
foreach ($d in 'plugin-kernel-tasks-20261123','plugin-kernel-tasks-20261123-r2','plugin-kernel-tasks-20261123-r3','plugin-kernel-tasks-20261123-r4') {
  $f = "\\10.16.11.3\ai\.agents\archive\$d\raw\tasks.md"
  "$d $((Get-FileHash $f -Algorithm SHA256).Hash.ToLower()) $((Get-Item $f).Length) B"
}

# 2) 快照之间的增删形态
git diff --no-index --numstat <r2>/raw/tasks.md <r3>/raw/tasks.md

# 3) 当前 spec 保真复验（需在 Linux 规范环境；Windows 侧只读复验见 SKILL.md §4）
python .agents/skills/spec-tasks-volumize/scripts/volumize.py check --plan .agents/archive/2026-11-23-plugin-kernel-tasks/plans/plan-r4.json

# 4) 萃取副本可用性（不触碰真实台账，tempdir 内全链路）
py -3.14 .agents\skills\spec-tasks-volumize\scripts\selftest.py
```

本次实测：命令 3 退出码 0（`BYTE_EXACT True`、`LOCK_OK volumes=11`、`INDEX_OK`）；命令 4 输出 `SUMMARY 16/16 PASS`。