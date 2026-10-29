# 子模块指针—文档一致性门禁 — 实施计划

> Plan 阶段产物，2026-10-29。自 spec.md（v0.1.0）派生。
> **需求对齐记录（Specify 微调）**：实施前对照 CONTRIBUTING.md §3.5 发现
> spec 草案 G1「已初始化（无 `-` 前缀）才通过」与仓库**延迟初始化策略**冲突——
> 未初始化子模块是正常态（克隆后目录为空属正常）。已将 G1 修正为
> 「未初始化记 note 不阻断；已初始化但指针漂移（`+` 前缀）才阻断」。
> spec.md G1 已同步修正。

## Task 1: 扩展 check_declaration_consistency.py（G1-G3 断言 + submodule_status 快照字段）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新增 `extract_submodule_status(root)`：解析 `git submodule status` 输出，
    返回 `(status_map: {path: sha}, version_map, anomalies, notes)`；
    `-` 前缀（未初始化）记入 notes，`+` 前缀（checkout 与索引不一致 = 指针漂移）记入 failures；
  - `collect_manifest` 新增 `submodule_status` 字段（仅正常条目）；
  - 新增 G 类断言：G1 状态可解析（rule）、G2 指针=快照且无漂移（rule）、
    G3 文档版本引用对账（rubric，warn）；
  - `--generate` 时：存在 `+` 漂移则拒绝刷新并退出码 2（禁止固化漂移状态），
    未初始化条目不写入快照（延迟初始化策略）；
  - 校验消息与 docstring 同步更新（六类 → 六类 + G 类）。
- **Acceptance Criteria Addressed**: G1/G2/G3、快照字段、--generate 保护
- **Test Requirements**:
  - `rule` TR-1.1: 脚本含 `extract_submodule_status` 且 `collect_manifest` 返回
    `submodule_status` 键；证据：源码 grep + 运行 `--json` 输出含该键
  - `rule` TR-1.2: `+` 漂移场景进入 failures（可用构造的 status 输出单测）；
    证据：`--json` 输出 failures 含漂移描述
  - `rule` TR-1.3: `--generate` 在漂移存在时退出码 2 且不写盘；证据：命令输出
- **Completion Evidence**:
  - TR-1.1 通过：`--json` 输出含 `submodule_status` 键（快照文件实测含 4 个子模块 SHA）
  - TR-1.2 通过：门禁首跑即拦截 `vendor/tvm-ffi` 真实漂移（`+e9fdc58...`，checkout 与索引不一致），failures 含漂移描述
  - TR-1.3 通过：漂移存在时 `--generate` 输出 `[ERROR] ... 拒绝刷新快照` 且退出码 2，快照未被固化
- **Notes**: 脚本自包含（不导入内部包），`subprocess` 调用 `git`（CI 环境
  checkout 后 git 可用）

## Task 2: 刷新 declaration-snapshot.json（新增 submodule_status）

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 运行 `python scripts/check_declaration_consistency.py --generate`，
    使快照包含 `submodule_status` 且与 `git submodule status` 当前输出一致；
  - 确认未初始化子模块不进入 `submodule_status`（当前 4 个子模块均已初始化）。
- **Acceptance Criteria Addressed**: 快照字段
- **Test Requirements**:
  - `rule` TR-2.1: 快照含 `submodule_status` 且键集合 = 已初始化子模块路径；
    证据：`Get-Content declaration-snapshot.json`
  - `rule` TR-2.2: 快照 SHA 与 `git submodule status` 输出一致；证据：比对输出
- **Completion Evidence**:
  - TR-2.1/TR-2.2 通过：快照 `submodule_status` 含 libs/mystx、libs/tvm-book、
    vendor/caffe、vendor/tvm-ffi 四个 SHA，与 `git submodule status` 实测逐项一致
    （tvm-ffi 恢复索引指针后刷新为 `6628c464...`）

## Task 3: 更新 scripts/README.md（G 类断言说明）

- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 1
- **Description**:
  - `check_declaration_consistency.py` 一行说明由「六类」更新为
    「六类 + G 类（子模块指针对账）」；补充 G1-G3 简述与 `--generate` 刷新行为。
- **Acceptance Criteria Addressed**: 文档与门禁同步
- **Test Requirements**:
  - `rule` TR-3.1: README 含「子模块指针对账」与 G1-G3 描述；证据：grep
- **Completion Evidence**:
  - TR-3.1 通过：scripts/README.md 已更新为「六类 + G 类子模块指针对账」，
    含 G1 状态可解析 / G2 指针=快照（拦截 `+` 漂移与裸 `modified: vendor`）/
    G3 文档版本引用对账

## Task 4: 更新 CONTRIBUTING.md（指针变更三步流程）

- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 1
- **Description**:
  - §3.5 子模块表后新增「子模块指针变更流程」小节：
    ①更新指针（`git submodule update --remote <path>` 或 checkout 目标提交）
    ②刷新快照（`python scripts/check_declaration_consistency.py --generate`）
    ③提交说明注明上游范围（如「更新 vendor/tvm-ffi 至 0932956（上游 main 快进 35 提交）」）；
  - 说明裸 `modified: vendor` 提交会被 G2 拦截。
- **Acceptance Criteria Addressed**: 指针变更流程显式化
- **Test Requirements**:
  - `rule` TR-4.1: CONTRIBUTING 含三步流程与 G2 拦截说明；证据：grep
- **Completion Evidence**:
  - TR-4.1 通过：CONTRIBUTING.md 新增 §3.6「子模块指针变更流程（G2 门禁）」，
    含三步命令与「禁止裸指针变更」说明

## Task 5: 门禁全量验证

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 2, 3, 4
- **Description**:
  - 运行 `python scripts/check_declaration_consistency.py`（校验模式）退出码 0；
  - 运行 `python scripts/check_declaration_consistency.py --generate`（幂等）无写盘；
  - 运行 `--json` 确认 `ok: true` 且无 G 类 failures；
  - CI 引用确认（`check declaration consistency` 步骤自动继承新断言）。
- **Acceptance Criteria Addressed**: 全部
- **Test Requirements**:
  - `rule` TR-5.1: 校验模式退出码 0、输出无 failures；证据：命令输出
  - `rule` TR-5.2: `--generate` 幂等（重复运行跳过写盘）；证据：命令输出
  - `rule` TR-5.3: `--json` 输出 `ok: true`；证据：JSON 输出
- **Completion Evidence**:
  - TR-5.1 通过：校验模式退出码 0，输出「六类 + G 类全部通过」
  - TR-5.2 通过：重复 `--generate` 输出「清单快照已是目标状态，跳过」退出码 0
  - TR-5.3 通过：`--json` 输出 `ok=True failures=0`
  - CI 引用确认：`.github/workflows/ci.yml` 的 `run: python scripts/check_declaration_consistency.py`
    无需改动，自动继承 G 类断言
  - 语法/风格：`py_compile` 通过；ruff 配置（E501 已忽略、I001 导入排序）与新增代码兼容
