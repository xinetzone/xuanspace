# xuan-compose 文档索引（dev 证据区）

> 本目录（`doc/dev/`）是 xuan-compose 对 [containers/podman-compose](https://github.com/containers/podman-compose)
> v1.6.0 分层重构（translate-then-move）的**证据区**。文档集总入口见 [../index.md](../index.md)。
>
> 2026-10-10 由 4 份长文档原子化重写为「索引 + 常青台账 + 时点报告」结构：
> 同一事实只保留一个权威陈述位置，时点报告中的数字/结论/编号保持 2026-09-16
> 交付快照原样，未做增删改（仅去重、引用化与结构重排；标题与导航性文字随原子化调整）。
> 旧→新迁移表与 V 阶段验证记录属冻结历史，已移至
> [2026-10-10 原子化重写档案](2026-10-10-atomization-record.md)（2026-10-12 自本页拆出）。

## 一句话结论

- **行为对等 pass**：142/142 上游公开符号有落点；argparse 参数表 11 维快照零差异；
  T11 独立对抗审查 **0 blocker / 0 major**；功能完整性独立复验 **0 个 P0**。
- **关键数字**：940 测试 / 0 skip（2026-09-16 交付时点口径；2026-10-10 同步后 965 例），整体覆盖率 92%（[91%/92% 双时点口径](#覆盖率口径说明91-与-92)并存）；
  39 个源文件分层 7 层，内部 import 零反向边。
- **遗留风险口径**：P2-1…P2-10 均为登记/加固类、非功能丢失（[台账 §C](findings-ledger.md#c-p2)）；
  U1-U13 为上游固有问题，对等重构刻意不改（[台账 §B](findings-ledger.md#b-upstream)）。

## 文档地图

| 我想…… | 文档 | 性质 |
|---|---|---|
| 理解对等性全貌、查基线与统计口径 | [parity/index.md](parity/index.md) | 常青入口 |
| 查某个上游符号迁到了哪个模块 | [parity/symbol-map.md](parity/symbol-map.md) | 机器台账 |
| 查 D1-D14 结构差异 / M1-M7 现代化项 | [parity/differences.md](parity/differences.md) | 常青台账 |
| 查逐字保留的上游怪癖清单 | [parity/upstream-quirks.md](parity/upstream-quirks.md) | 常青台账 |
| 了解 T1-T12 交付了什么（规模/提交链/架构/边界） | [delivery/2026-09-16-delivery-summary.md](delivery/2026-09-16-delivery-summary.md) | 时点快照 |
| 复用单体重构的六个可迁移模式 | [patterns/2026-09-16-refactor-patterns.md](patterns/2026-09-16-refactor-patterns.md) | 模式沉淀 |
| 读 T11 独立对抗审查过程、修复与 AC 证据 | [review/2026-09-16-t11-adversarial-review.md](review/2026-09-16-t11-adversarial-review.md) | 时点快照 |
| 读功能完整性独立复验（七维证伪/P1 修复/复现命令） | [verification/2026-09-16-full-parity-verification.md](verification/2026-09-16-full-parity-verification.md) | 时点快照 |
| 查所有已登记问题的处置结论（R/N/U/P2/P1） | [findings-ledger.md](findings-ledger.md) | 常青台账（单一事实源） |
| 查 2026-10-10 原子化的旧→新流向与 V 阶段验证证据 | [2026-10-10-atomization-record.md](2026-10-10-atomization-record.md) | 时点档案（冻结） |

## 上游基线（单一事实源）

- 上游：[containers/podman-compose](https://github.com/containers/podman-compose) v1.6.0
- pinned commit：`1f1cb09bfee4d79857ed23bfe2699febd7d20a53`（2026-10-10 由 `e3df104` 推进，区间增量已对等移植，见 [findings-ledger.md §G](findings-ledger.md)）
- 许可证：GPL-2.0-only（继承上游，不得重新许可）
- 重构方式：translate-then-move（翻译式搬迁，非兼容封装、非重写）
- 术语注：2026-09-16 交付快照抬头中的 `translate-then-mount` 是本词的旧写法，
  与 translate-then-move 指同一概念；以本页写法为准。

本目录各文档凡引用该基线，均以此处为准，不再各自重复声明。

## 编号体系

七套编号来自交付过程的不同阶段，全程保持原编号不变以维持历史可追溯性；
跨文档引用时按下表定位权威位置。

| 前缀 | 含义 | 权威位置 |
|---|---|---|
| `D1`-`D14` | 结构性重构差异登记（语义等价、架构必需） | [parity/differences.md](parity/differences.md) |
| `M1`-`M7` | Python 3.14 兼容修复与机械现代化（无语义变化） | [parity/differences.md](parity/differences.md) |
| `F1`-`F4` | T11 独立审查本轮修复项（fail→fix→回归） | [review 报告 §5](review/2026-09-16-t11-adversarial-review.md#5-本轮修复failfix回归全部完成) |
| `R1`、`N1`-`N3` | T11 审查「经评估保留」项（重构有意决策/形式项） | [findings-ledger.md §A](findings-ledger.md#a-keep) |
| `U1`-`U13` | 上游固有问题（对等重构不改，列上游回馈候选） | [findings-ledger.md §B](findings-ledger.md#b-upstream) |
| `P2-1`-`P2-10` | 独立复验登记/加固类遗留（非功能丢失） | [findings-ledger.md §C](findings-ledger.md#c-p2) |
| `AC-2`-`AC-12` | spec 验收标准（Acceptance Criteria） | [review 报告 §7](review/2026-09-16-t11-adversarial-review.md#7-各验收标准ac证据索引) |
| `TR-x.y` | spec 任务记录（Task Report） | 主仓 `.trae/specs/xuan-compose-refactor/tasks.md`（历史档案） |
| `T1`-`T12` | spec 任务切片链（每片一门禁的 12 个实施 Task） | [交付报告 §3](delivery/2026-09-16-delivery-summary.md#3-切片交付链r12-笔提交事实)；任务记录在主仓 tasks.md |
| `G1`-`G4` | 七概念方法论阶段代号（G2=洞察四元组、G3=模式萃取等） | G2 见[交付报告 §5](delivery/2026-09-16-delivery-summary.md#5-i-洞察为什么对等目标达成g2-四元组)，G3 产出见 [patterns](patterns/2026-09-16-refactor-patterns.md) |
| `I→F→A→V→C` / `F→V→I` | 七概念方法链（洞察→第一性原理→作用域→验证→闭环；复验子链） | [verification 报告文首](verification/2026-09-16-full-parity-verification.md) |

## 证据与复现总账

| 维度 | 证据 |
|---|---|
| 功能对等 | [parity/symbol-map.md](parity/symbol-map.md)；tests/test_cli_parity.py + tests/snapshots/cli_parity.json（25 命令/11 维/锚定 e3df104）（快照锚点刻意停在 e3df104：本次 1f1cb09 增量无 argparse 参数增删、快照内容不变；防篡改守卫见 tests/test_cli_parity.py） |
| 符号对等 | scripts/symbol_inventory.py 输出 142/3（复现：`python scripts/symbol_inventory.py`）；[parity/symbol-map.md](parity/symbol-map.md) |
| 测试保真 | tests/ 56 文件 940 例（2026-09-16 交付时点）；2026-10-10 同步后 57 文件 965 例；上游 26 文件 484 例对拍（拆分守恒：42=18+24）；详见 [交付报告 §2](delivery/2026-09-16-delivery-summary.md#2-交付物规模实测可复现) |
| 覆盖率 | WSL `pytest --cov`：整体 92%（T11 修复后回归/独立复验时点），核心 ≥90%，runner 96%；口径说明见下节；2026-10-10 同步后复测仍为 92%（965 passed） |
| 静态质量 | ruff(src+tests+scripts) 0；mypy src 0（39 源文件，178/178 公开 API 有注解） |
| 架构边界 | [review 报告 §3](review/2026-09-16-t11-adversarial-review.md#3-tr-113-分层单向性修复后-55)：108 条 import 零反向边；argv 单点；subprocess 单模块；包根零副作用（运行时实测）。口径注：2026-10-10 AST 复核为 107 条 import 语句，108 是 T11 grep 命中数（多计 engine.py 一条含 `from .env` 的英文注释），零反向边结论不变；快照数字保持 108 |
| 安全对抗 | [review 报告 §4](review/2026-09-16-t11-adversarial-review.md#4-tr-114-五攻击者对抗审查实质度-55)：五视角无重构新增 blocker；[findings-ledger.md §B](findings-ledger.md#b-upstream) 13 条上游固有问题台账 |
| 许可合规 | 96/96 .py 带 SPDX 头；LICENSE 与上游归一化 sha256 一致；README 署名三要素 |
| vendor 洁净 | `git -C vendor/podman-compose status --porcelain` 空，HEAD=1f1cb09（2026-10-10 上游增量同步后；2026-09-16 T11 复核时点为 e3df104，见 [review 报告 §1](review/2026-09-16-t11-adversarial-review.md#1-tr-111-vendor-子模块零改动)） |
| 功能完整性独立复验 | 七维证据链与运行时实证见 [verification 报告](verification/2026-09-16-full-parity-verification.md) |

完整复现命令（WSL 权威门禁、纯 Python wheel/sdist 干净构建）：
[verification 报告 §7](verification/2026-09-16-full-parity-verification.md#7-复现命令)。

### 覆盖率口径说明（91% 与 92%）

文档集中并存两个覆盖率数字，来自不同测量时点，均如实保留、未做归一化：

- **91%**：T9/T10 门禁时点口径，见 [parity 入口的快照证据](parity/index.md#对等证据)与
  库 README 测试节；
- **92%**：T11 修复后回归（[review §5](review/2026-09-16-t11-adversarial-review.md#5-本轮修复failfix回归全部完成)）
  与独立复验（[verification §4.3](verification/2026-09-16-full-parity-verification.md#43-修复后实证临时目录干净-pep-517-构建仓库零构建残留)）时点口径。

## 历史档案

- [2026-10-10 原子化重写档案](2026-10-10-atomization-record.md)：4 份长文档→本结构的
  旧→新迁移表与 V 阶段验证记录（冻结历史，除勘误外不再更新；2026-10-12 自本页拆出）。

```{toctree}
:hidden:
:maxdepth: 2

findings-ledger
2026-10-10-atomization-record
parity/index
delivery/2026-09-16-delivery-summary
patterns/2026-09-16-refactor-patterns
review/2026-09-16-t11-adversarial-review
verification/2026-09-16-full-parity-verification
```
