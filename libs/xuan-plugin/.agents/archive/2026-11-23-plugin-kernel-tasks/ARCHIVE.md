# 归档记录：plugin-kernel tasks.md 快照链（2026-11-23）

本目录是 `spec-tasks-volumize` 技能的项目内归档台账。正文载荷体积较大且已在根工作区归档，这里**不重复存放 raw 正文**，只固化「源路径 + 哈希 + 体量 + 可复验命令」，并用 `plans/` 存下每代计划文件（可重放分卷定义）。

## 1. 快照清单

来源根：`/media/pc/data/ai/.agents/archive/`（SMB 等价路径 `\\10.16.11.3\ai\.agents\archive\`）。

| 快照 | 目录 mtime | raw 字节 | 换行数 | `split` 行数 | `## Task` | raw SHA-256 |
|---|---|---:|---:|---:|---:|---|
| r1 | 2026-11-23 09:40:49 | 101461 | 1359 | 1360 | 24 | `39b4c9e99f05d688b296b36821b3100c5e343d6c446f98a142d0652ec3ececf3` |
| r2 | 2026-11-23 10:08:42 | 135512 | 1778 | 1779 | 31 | `b6731e8a3f8ea96040b9be00bf73d027be308f306ca49fc7f6aa17c4e55779f0` |
| r3 | 2026-11-23 11:02:18 | 140655 | 1851 | 1852 | 33 | `af0546dca6b7e4f935b6a93a4035290db8aae2c2aa2d3025fcc1a2d833a4b3dc` |
| r4 | 2026-11-23 11:24:08 | 143421 | 1890 | 1891 | 34 | `28e0c1e77ab6707b2ec24b9a71a2a542afe17c1981a183a03eca52f380265dba` |

四份 `MANIFEST.sha256` 均与磁盘现值哈希一致（本次逐条核对）。

增量形态：r1→r2 `+424/−5`、r2→r3 `+74/−1`、r3→r4 `+39/−0`。

## 2. 计划文件副本

| 副本 | 对应快照 | 冻结来源（plan.source） | 卷数 |
|---|---|---|---:|
| `plans/plan-r2.json` | r2 | `/tmp/vd2/tasks.md` | 8 |
| `plans/plan-r3.json` | r3 | `.temp/p3-2-contract-gate/tasks.md` | 10 |
| `plans/plan-r4.json` | r4 | `.temp/p3-1-lock/tasks.md` | 11 |

r1 无 `plan.json`（当时仅冻结 + 手工分卷）。三代 plan 的 `out_dir` / `index.path` 相同，均指向 `.trae/specs/plugin-kernel-extraction/tasks/` 与 `tasks.md`。

## 3. 复验

```powershell
# 源哈希是否仍与台账一致（任一不符即说明源归档被改写）
Get-FileHash '\\10.16.11.3\ai\.agents\archive\plugin-kernel-tasks-20261123-r4\raw\tasks.md' -Algorithm SHA256

# 计划副本完整性（sha256sum -c 兼容格式）
Get-FileHash .\plans\*.json -Algorithm SHA256
```

现状（本次实测，2026-11-23）：`.trae/specs/plugin-kernel-extraction/tasks.md` 为 6969 字节 / 65 行索引页，分卷 11 卷 + `.volumize.lock.json`（5548 字节），`check` 退出码 0，`BYTE_EXACT True`、`LOCK_OK volumes=11`、`INDEX_OK`。

## 4. 注意事项

- 本目录是**说明性归档**；保真权威件仍是根工作区 `.agents/archive/<snapshot>/raw/tasks.md` 与其 `MANIFEST.sha256`。
- r2 的现值与上游案例档案记载的体量不一致（见技能 `references/case-2026-11-23-plugin-kernel.md` §1.4），说明该目录曾被原地重写；引用 r2 时以本表哈希为准。
- 新增快照必须 `-rN+1`，不得覆盖既有目录。

详见技能：`.agents/skills/spec-tasks-volumize/SKILL.md`。