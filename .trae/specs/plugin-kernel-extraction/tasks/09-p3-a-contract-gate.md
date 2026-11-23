> 分卷来源：`tasks.md` 原文 L1817–L1852（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：08-p2-d-doc-grouping.md · 下一卷：10-p3-b-lock.md
> 主题：P3-A：H 类插件契约断言接入 CI
## Task P3-2-1: H 类插件契约断言接入 CI（设计 §5.7）

- **Status**: `completed`
- **目标**：把设计 §5.7 的 H 类不变量（H1 可解性、H2 隔离性、H3 可撤销性、
  H4 视图一致性、H5 结构兼容）从文档约定变为 CI 可断言门禁，失败信息须能直接定位。
- **改动**：
  - 新增 `scripts/check_plugin_contract.py`：发现全仓 `[tool.xuan.plugin]` 清单后，
    H1/H2 复用内核 `build_graph` / `find_conflicts` 判定（与求解器同口径），
    H3 扫描插件入口模块的模块级可变全局与函数内 `global`（给出文件:行号），
    H4 在 `xuan.plugins.lock` 存在时按规范 JSON 重算比对（未落地时跳过），
    H5 校验 `inject.constraint` 的 `methods:` 方法集；
  - `.github/workflows/ci.yml` quality job 新增步骤 `Check plugin contract`（1d），
    门禁计数由七项更新为八项；
  - `scripts/README.md` 脚本清单补一行。
- **验收**：
  - 新增 `tests/test_check_plugin_contract.py` 8 例：6 类反例（H1 缺失提供者、
    H2 同键多 single、H2 multi 同 priority、H3 模块级可变全局、H4 lock digest
    不一致、H5 约束方法缺失）全部被拦截，消息含能力键、双方 id 或文件:行号；
    另含无 lock 时跳过与 lock 一致两条正例；
  - `pytest`：根 65 例、三库 173 例全通过；`ruff check .` 与
    `ruff format --check .` 通过；
  - 五个治理脚本（pyproject / declaration / dependency / spec / plugin-contract）
    在真实仓库均 rc=0。
- **Notes**：
  - **V1 取舍：H1/H2 复用内核而非重写判定**——自建判定会与求解器口径漂移，
    故仅自建「key → (id, mode, priority)」用于生成可读消息，裁决仍交
    `find_conflicts`。
  - **V2 取舍：H3 不拦截任意模块级构造调用**——`MANIFEST = PluginManifest(...)`
    与 `DEFAULT_DISPATCHER = BuildDispatcher()` 是既有合法写法，一律拦截会产生
    3 处误报；故只识别可变字面量与 `dict()/list()/set()` 等可变容器构造，并豁免
    `__all__` 一类双下划线导出名。
  - **V3 取舍：H4 无 lock 时跳过而非失败**——`xuan.plugins.lock` 尚未落地
    （全仓 0 个），强行要求会阻塞无关改动；lock 一旦存在即按规范 JSON 全量比对。
  - **V4 边界**：未改动 kernel 判定逻辑，未新增 lock 生成器；H5 仅支持
    `methods:<m1,m2>` 语法，其他约束语法判失败并提示。

