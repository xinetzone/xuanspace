# xuan-compose 对等审计入口

> xuan-compose 对 [containers/podman-compose](https://github.com/containers/podman-compose)
> v1.6.0 分层重构的**对等性证据**（对应 spec AC-3/AC-10、TR-10.1/10.3）。
> 基线 commit 与许可证见 [文档索引「上游基线」](../README.md#上游基线单一事实源)。

## 统计口径

AST 解析上游单文件 `podman_compose.py` 的**顶层公开符号**（不以 `_` 开头的
`def`/`async def`/`class`，以及全大写/`__dunder__` 模块级常量；
`@overload` 重复定义按 distinct 名称计 1 个）。

- 复现命令（子项目根）：`python scripts/symbol_inventory.py`

## 证据分册

| 分册 | 内容 |
|---|---|
| [symbol-map.md](symbol-map.md) | 总账（TR-10.1）、逐模块符号映射表、结构性移除（AC-3 点名）、新库新增符号 |
| [differences.md](differences.md) | M1-M7 py3.14 兼容/机械现代化、D1-D14 结构性重构差异登记（TR-10.3） |
| [upstream-quirks.md](upstream-quirks.md) | 逐行保留、未顺手「修正」的上游怪癖（T9 测试固化） |

## 对等证据

- argparse 参数表快照（25 命令，11 维，锚定 e3df104）：
  `tests/snapshots/cli_parity.json`；对拍测试 `tests/test_cli_parity.py`、
  `tests/test_command_surface.py`；重新生成 `tests/generate_parity_snapshots.py`。
  （快照锚点刻意停在 e3df104：本次 1f1cb09 增量无 argparse 参数增删、快照内容不变；防篡改守卫见 `tests/test_cli_parity.py`）。
- 上游 26 个单元测试文件全量同名移植（484 例 ≥ 上游 478），见 tasks.md T9 TR-9.2；2026-10-10 同步 1f1cb09 增量后本库移植用例增至 57 文件 / 965 例（+25，全部来自本次上游测试移植）。
  口径注：[review 报告 §7](../review/2026-09-16-t11-adversarial-review.md#7-各验收标准ac证据索引)
  AC-5 沿用 spec 原始措辞记作「27 测试移植」，交付实测为 26 个测试文件 / 484 例（TR-9.2）；
  两数并存，以 TR-9.2 实测为准。
- 覆盖率：整体 91%（T9/T10 门禁时点），核心模块 interpolation/normalize/merge/engine/translate 全 ≥90%；
  T11 修复后回归为 92%，口径说明见 [文档索引](../README.md#覆盖率口径说明91-与-92)；2026-10-10 同步后复测 92%（965 passed）。
