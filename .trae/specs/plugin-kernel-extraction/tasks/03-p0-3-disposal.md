> 分卷来源：`tasks.md` 原文 L412–L465（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：02-p0-2-identity.md · 下一卷：04-p1-kernel-cli.md
> 主题：增量二 P0-3：卸载异常聚合
# 增量二：P0-3（撤销加固）— 执行记录

> 2026-11-21 追加。自设计文档 §7 P0-3 派生，原文判据为「卸载异常聚合上报；FAILED 前强制撤销
> 已登记 effect」，验收要求为「注入故障插件的测试断言 effect 数 = 0 且异常被上报」。

## Task P3-1: 卸载异常聚合与装配失败回滚

- **Status**: `completed`
- **Priority**: high
- **Depends On**: P2-1（`a39c72d`）
- **Description**:
  - `disposable.py` 新增 `DisposalError(errors)`：持有 `list[tuple[label, Exception]]`，
    消息形如「N 个效应撤销失败: label: Type: msg; …」；
  - 新增 `DisposableList.dispose_all(report=None) -> list[tuple[str, Exception]]`：逆序取出全部逆函数，
    逐项隔离执行；label 取 `__name__`，否则 `dispose#index`；可选 `report` 逐项即时上报；
    与 `clear()` 的分工是「异常隔离由本方法负责」；
  - `Fiber._unload()` 改用 `dispose_all()`：效应数必归零；有失败则
    `_error = DisposalError(errors)` 并置 `FAILED`，否则置 `PENDING`；
  - `Fiber._reload()` 的 `apply` 失败路径先回滚已登记效应，把 `("apply", e)` 与清理期失败
    合并为 `DisposalError`，避免失败组件留下部分装配痕迹；
  - `Fiber.effect().undo()` 与 `Context.effect().undo()` 改为聚合抛 `DisposalError`，废弃静默吞异常；
  - `Context.dispose(report=None)` 汇总全部 fiber 与根效应失败，**不主动抛出**（在 `__exit__`
    抛出会掩盖代码块原有异常），结果留存供 `get_disposal_errors()` 查证；
    `Fiber.dispose()` 在卸载后仍为 `FAILED` 时保留 `FAILED`，否则置 `DISPOSED`。
- **Acceptance Criteria Addressed**: 卸载后效应数必为 0；失败以聚合异常上报而非静默丢失
- **Test Requirements**:
  - `rule` TR-P3.1: `dispose_all` 逆序执行且单点失败不中断；证据：`calls == ["last", "first"]`
  - `rule` TR-P3.2: `report` 回调逐项收到失败；证据：回调收集列表长度与 label 相符
  - `rule` TR-P3.3: `_unload` 后 `len(fiber._disposables) == 0` 且状态 `FAILED`、`error` 为 `DisposalError`
  - `rule` TR-P3.4: `_reload` 失败时已登记 effect 被回滚；证据：故障插件用例
  - `rule` TR-P3.5: 卸载失败后 `dispose()` 保留 `FAILED` 供查证
  - `rule` TR-P3.6: `Context.dispose()` 同时收集 `fiber:*` 与 `root:*` 两类失败
  - `rule` TR-P3.7: `tools/okf` 通过数不回归
- **Completion Evidence**（2026-11-21 实测）:
  - 提交：`2dadf4f`（完整 `2dadf4f841eb3accf725b137d6dbc8390c4809da`），
    信息 `feat(plugin): 卸载异常聚合上报，废弃静默吞异常`，7 文件 +246 / −32。
  - 新增 `libs/xuan-plugin/tests/test_disposal_errors.py`（121 行、7 项）：`dispose_all` 逆序不中断、
    `report` 逐项、`_unload` 后效应数归零并置 `FAILED`、`_reload` 失败回滚、`dispose` 保留 `FAILED`、
    `Context` 收集 `fiber:bad`、`Context` 收集 `root:_boom`。
  - TR-P3.1～P3.6 通过：新库 **18 passed**（11 摘要 + 7 撤销）。
  - TR-P3.7 通过：`cd tools/okf && python -m pytest -q` → **259 passed**（无回归）。
  - 门禁：`ruff check` All checks passed；`ruff format --check` 54 files already formatted；
    `mypy libs/xuan-plugin` → Success: no issues found in 7 source files。
  - 旧契约改写 3 处：`tools/okf/tests/test_context_extra.py` 的 `test_effect_undo_aggregates_exception`、
    `test_plugin_extra.py` 的同名用例与 `test_unload_aggregates_dispose_exception`——此前断言
    「静默吞异常」，现改为断言 `DisposalError` 聚合、`len(errors) == 1`、效应数归零。
  - 记录订正：`plugin.py` 中 `except OSError, TypeError:` 曾被误判为语法错误，实为 PEP 758
    （Python 3.14 允许 `except` 省略括号）的合法写法；`ast.parse` 与 `py_compile` 均通过，
    `ruff format` 的规范化结果即为此形式。
- **Notes**: 设计取舍是「效应归零」优先于「首个异常立即抛出」——`DisposalError` 只承载汇总，
  不改变既有 `try/except Exception` 的捕获面；`Context.dispose()` 保持不抛出属刻意选择。

---

