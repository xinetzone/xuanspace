# xuan-compose 文档原子化重写档案（2026-10-10）

> 本档案是 2026-10-10「4 份长文档 → 索引 + 常青台账 + 时点报告」原子化重写的
> 动作证据，2026-10-12 自 [index.md](index.md) 拆出，使文档索引页只承载常青内容。
> 本页两节均为**冻结历史**：迁移表记录旧→新内容流向，验证记录留存 V 阶段证据，
> 除勘误外不再更新；时点报告中的数字/结论/编号保持 2026-09-16 交付快照原样。
> 当前文档结构与文档地图见 [index.md](index.md)。

## 文件迁移表（旧 → 新）

原子化重写（2026-10-10）的内容流向。主仓 `.trae/specs/xuan-compose-refactor/tasks.md`
中对旧路径的引用属交付时点历史档案，保持原样并以本表作为溯源对照。

| 旧文件 | 新位置 |
|---|---|
| parity.md（头部基线/口径/复现） | [parity/index.md](parity/index.md) |
| parity.md §1-§4 | [parity/symbol-map.md](parity/symbol-map.md) |
| parity.md §5.1-§5.2 | [parity/differences.md](parity/differences.md) |
| parity.md §5.3 | [parity/upstream-quirks.md](parity/upstream-quirks.md) |
| parity.md §6 | [index.md](index.md)「证据与复现总账」+ [parity/index.md](parity/index.md) |
| 2026-09-16-xuan-compose-delivery-summary.md §1-§5、§8 | [delivery/2026-09-16-delivery-summary.md](delivery/2026-09-16-delivery-summary.md) |
| 2026-09-16-xuan-compose-delivery-summary.md §6 | [patterns/2026-09-16-refactor-patterns.md](patterns/2026-09-16-refactor-patterns.md) |
| 2026-09-16-xuan-compose-delivery-summary.md §7 | [index.md](index.md)「证据与复现总账」 |
| review.md §0-§5、§7、§8 | [review/2026-09-16-t11-adversarial-review.md](review/2026-09-16-t11-adversarial-review.md) |
| review.md §6 | [findings-ledger.md](findings-ledger.md) §A/§B |
| 2026-09-16-xuan-compose-full-parity-verification.md §1、§2、§6、§7 | [verification/2026-09-16-full-parity-verification.md](verification/2026-09-16-full-parity-verification.md) |
| 2026-09-16-xuan-compose-full-parity-verification.md §4（P1 修复全文） | verification 报告 §4（正文保留）；摘要另登记至 [findings-ledger.md §D](findings-ledger.md#d-p1-fixed) |
| 2026-09-16-xuan-compose-full-parity-verification.md §3（环境 P1） | [findings-ledger.md](findings-ledger.md) §E |
| 2026-09-16-xuan-compose-full-parity-verification.md §5（P2 清单） | [findings-ledger.md](findings-ledger.md) §C |

## 原子化重写验证记录（2026-10-10，V 阶段）

- 事实守恒：与 git HEAD 四份旧文做编号集合机械比对——D/M/U/P2/F/N/TR-11/AC/T
  九套编号集合**零缺失、零新增**；关键数字（5541/6472/10844/142/940/484/478/481/
  97651/271449/39/56/96/108/91%/92% 等）全部在位；vendor 行号抽验无误
  （normalize_service_final L2201、normalize_final L2214）。
- 现状实测：`symbol_inventory.py` 输出 142=138+4、无落点 3（与 D1/D2/D6 一致）；
  src=39 文件、tests=56 文件、src+tests+scripts=96 个 .py 全部带 SPDX 头
  （与 T11 F4 口径一致）。
- 链接门禁：107 个相对链接 + 锚点全绿（子模块不被主仓 check-links 覆盖，改用
  一次性检查器按 GitHub slug 规则校验，含中文标题锚与显式 `<a id>`）。
- 四视角对抗审查：fresh-context 审查员提出 13 条（0 P0/1 P1/6 P2/6 P3），
  采纳 10 条已修复（P2-3 闭环登记、P2-6 路径更正、结论卡、编号表补 T/G/方法链、
  mount/move 术语注、26/27 口径、迁移表细化、死锚启用、快照导航文字声明、
  台账维护说明）；驳回 3 条并记录理由：
  ①「恢复 CMakeLists 链接」——该文件已被子模块后续提交 `f293ed6` 删除、
  且新位置深一级，链接必断，故降级为纯文本（快照事实保留）；
  ②「review §6 中文锚点」——复核时已统一为显式锚 `#b-upstream`；
  ③「常青层 142/commit 重复」——属证据入口的必要交叉引用，刻意保留。
- 外部引用同步修复：库 README×2、CHANGELOG×4、symbol_inventory.py×2、
  主仓 vendor-snapshot-parity SKILL×1、主仓 sticker-wiki 教程×2（源属性与断链）。
  主仓 `.trae/specs/xuan-compose-refactor/tasks.md` 中的旧路径按历史档案保持原样。
