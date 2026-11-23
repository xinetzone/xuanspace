> 原子化来源：`retrospective-20261121.md` 原文 §一（L11–L58）；该独立文件已于 2026-11-23 并入本目录并删除，唯一权威为本目录
> 文档索引：[README.md](README.md)

## 一、R 阶段：客观事实（G1）

### R1 提交链与改动量

| 序号 | 提交 | 任务 | 文件数 | 增 / 删 |
|:--:|------|------|:--:|:--:|
| F1 | `f6fd5dd` | P2-4 生命周期七态化、显式重试、静态插件抑制热换 | 11 | +405 / −38 |
| F2 | `8d3b173` | P2-5 热替换事务化、apply 期效应归属 | 6 | +418 / −29 |
| F3 | `beb5fbb` | P2-6 下游通知批处理、消除 INACTIVE 静默缺口 | 5 | +456 / −66 |
| F4 | `62d42a0` | P2-7 身份摘要去地址化 | 7 | +191 / −24 |
| F5 | `be6f6b7` | P2-8 状态机不变量收口（error 语义与迁移表机检） | 6 | +184 / −11 |
| F6 | `f42c37d` | P2-9 ACTIVE 与可解性互锁 | 5 | +148 / −3 |
| F7 | 合计 | 六次提交、单主线连续 | 40 文件次 | +1802 / −171 |

### R2 代码与测试体量

- F8：`libs/xuan-plugin/src/xuan_plugin/plugin.py` 共 **556 行**。
- F9：`libs/xuan-plugin/tests/` 共 **8 个测试文件、1559 行**（`test_assembly` 127、`test_digest` 162、`test_disposal_errors` 153、`test_hot_reload` 224、`test_lifecycle` 267、`test_manifest` 200、`test_notification_batch` 227、`test_solver` 199）。
- F10：第一批测试（`tests` + `xuan-plugin` + `xuan-build` + `xuan-ffi`）当前 **197 passed**；P2-8 收尾时为 194，P2-9 净增 3。
- F11：`tools/okf` 独立测试当前 **259 passed**；P2-7～P2-9 三轮均为 259，无增减。

### R3 质量门禁实测（2026-11-21）

- F12：`ruff check libs/xuan-plugin tools/xs libs/xuan-ffi` → `All checks passed!`
- F13：`ruff format --check` 同范围 → **58 files already formatted**。
- F14：`mypy libs/xuan-plugin/src libs/xuan-ffi/src tools/xs/src` → `Success: no issues found in 40 source files`。
- F15：`check_pyproject_style.py` → 扫描 16 个 `pyproject.toml`，全部通过。
- F16：`check_dependency_direction.py` → 扫描 337 个 Python 文件，全部通过。
- F17：`check_spec_integrity.py` → 3 个 `tasks.md`、**35 个任务编号**，全部通过。
- F18：`check_declaration_consistency.py` → 失败 2 项：`libs/mystx` 指针漂移（当前 `a346e83`）与 `scripts/declaration-snapshot.json` 的 `submodule_status` 过期；两项在 P2-7/P2-8 轮次即已存在。
- F19：`python -m xs.cli plugin apply` → 「装配完成：3 个节点 ACTIVE，16 条命令已登记；对账：一致 20 键，缺 0，多 0」。
- F20：`git status --porcelain` 在 `f42c37d` 之后仅剩 `M libs/mystx`。

### R4 状态机与关键常量（代码事实）

- F21：`FiberState` 共 **7 态**，`INACTIVE = 6` 追加在末尾，既有六态数值未漂移。
- F22：`_TRANSITIONS` 共 **15 条合法边**，`DISPOSED` 出度为 0；7×7 共 49 个状态对由 `test_transition_rejects_every_edge_outside_the_table` 穷举。
- F23：`_DIGEST_LEN = 16`（64 bit）；`_ADDRESS_RE = 0x[0-9a-fA-F]{8,}`，命中后折叠为常量 `0xADDR`。
- F24：`_MAX_FLUSH_ROUNDS = 100`，超限抛 `RuntimeError` 并列出待处理键。
- F25：P2-9 的收敛路径为 `LOADING → ACTIVE → UNLOADING → PENDING → INACTIVE`，共 **4 次迁移**（迁移表不允许 `LOADING → INACTIVE`）。
- F26：`Fiber.error` 与状态互锁口径为 `FAILED ⇔ error is not None`；`_unload` 成功分支显式清零。
- F27：P2-9 新增测试 3 条（`test_hot_reload.py` 2 条、`test_lifecycle.py` 1 条），全部为可失败回归用例。
- F28：`tasks.md` 现有 9 个 P2 任务块（P2-1～P2-9，均带 `-1` 序号后缀），`check_spec_integrity.py` 计数 35 与本轮新增一致。

### G1 结论

事实条目 28 条（F1–F28），全部可由提交、文件、命令输出或代码常量复核，无因果词。**G1 通过。**

---

相关页面：[文档索引](README.md) · 下一篇：[I — 洞察（G2）](12-retro-insights.md)
