> 分卷来源：`tasks.md` 原文 L1853–L1891（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：09-p3-a-contract-gate.md · 下一卷：（无）
> 主题：P3-B：xuan.plugins.lock 生成器与 H4 生效
## Task P3-1-2: `xuan.plugins.lock` 生成器与 H4 视图一致性生效（设计 §7 P3-1）

- **Status**: `completed`
- **目标**：落地设计 §7 P3-1——`xuan.plugins.lock` 成为插件契约唯一的派生快照，
  由当前清单重算得到；H4 从「无 lock 时跳过」转为 CI 实际对账。
- **改动**：
  - 新增 `libs/xuan-plugin/src/xuan_plugin/lock.py`：`contract_payload` /
    `plugin_digest` / `build_lock` / `render_lock` / `lock_diff`；规范形态
    `schema=1`、`plugins` 按 id 排序、契约字段剔除 `source`；
  - `xuan_plugin/__init__.py` 导出上述 lock API；
  - `scripts/check_plugin_contract.py` 删除本地摘要与 diff 重复实现，改为复用
    内核；新增 `--write-lock` 生成模式（幂等、LF 落盘）；
  - 生成并提交仓库根 `xuan.plugins.lock`（3 个插件：`xs.commands` /
    `xuan.build` / `xuan.ffi`），H4 转入真实对账；
  - `scripts/README.md`、`doc/design/07-roadmap.md`、
    `doc/appendix/17-retro-appendix-a.md` 状态回填。
- **验收**：
  - 根 `pytest` 68 例、内核 `pytest` 138 例全通过（新增 lock 单测 6 例、
    门禁回归 3 例：生成→H4 通过、二次生成幂等、清单变更后 lock 过期被拦截）；
  - `ruff check .` 与 `ruff format --check .` 通过；
  - 五个治理脚本（pyproject / declaration / dependency / spec /
    plugin-contract）在真实仓库均 rc=0；
  - 「删除人工快照」演练：临时移走 `scripts/declaration-snapshot.json` 后
    `check_declaration_consistency.py` 仍 rc=0（仅 NOTE 提示），随后原样恢复。
- **Notes**：
  - **V1 取舍：生成器落在门禁脚本 `--write-lock` 而非新增 `xs plugin lock`**——
    保证「扫描集合」与 H4 判定同源（同一 `load_plugins`），避免 CLI 与门禁两套
    发现口径漂移；面向开发者的 `xs plugin verify/why` 留给 P3-3。
  - **V2 取舍：摘要与 diff 上移内核**——`plugin_digest` / `lock_diff` 从脚本搬
    到 `xuan_plugin.lock`，未来消费者（`xs plugin verify`、跨语言代理）共用同一
    契约定义。
  - **V3 边界：未删除 `scripts/declaration-snapshot.json`**——该文件仍是
    `submodule-alignment-gate` spec G2 的指针快照载体（`submodule_status`），
    退役需同步修订该 spec，属独立决策；本轮仅以演练证明「缺失时不阻断 CI」。
  - **V4 说明**：lock 不含 `source`，同一 commit 在不同检出路径下重算逐字节
    一致；`--write-lock` 仅在内容变化时写盘。
  - **V5 编号**：台账既有 `Task P3-1` 为设计 P0-3 派生的卸载加固任务（编号历史
    占用），本任务沿用设计 §7 编号记为 `P3-1-2`，两者无依赖关系。

