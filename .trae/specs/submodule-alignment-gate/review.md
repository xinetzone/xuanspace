# 子模块指针—文档一致性门禁 — 独立审查

> 审查者上下文：独立复核轮（与实施者自验分离的检查点），对照 spec.md（v0.1.0，含 G1 修正）与 tasks.md 全部 AC/TR 逐项核验，2026-10-29。
> 关键独立证据：门禁首跑即捕获 `vendor/tvm-ffi` **真实指针漂移**（checkout `e9fdc58` vs 索引 `6628c46`，领先 30 提交），G2 拦截与 `--generate` 拒绝均经真实世界验证。

- [x] CP-R1: 声明快照含 `submodule_status` 字段且与 `git submodule status` 一致
  - **Type**: `rule`
  - **Covers**: spec 快照字段、TR-2.1/2.2
  - **Evidence**: `declaration-snapshot.json` 含 4 个子模块 SHA；与 `git submodule status` 实测逐项一致（mystx `18f3486…`、tvm-book `103bea7…`、caffe `f938d88…`、tvm-ffi `6628c464…`）

- [x] CP-R2: G2 拦截指针漂移，`--generate` 拒绝固化漂移状态
  - **Type**: `rule`
  - **Covers**: G2、TR-1.2/1.3
  - **Evidence**: 实施首跑时 tvm-ffi 存在 `+e9fdc58…` 漂移——校验模式 failures 含「子模块指针漂移（checkout 与索引不一致）」；`--generate` 输出 `[ERROR] 拒绝刷新快照` 且退出码 2，快照未被固化

- [x] CP-R3: 一致性门禁整体通过且幂等
  - **Type**: `rule`
  - **Covers**: G1、TR-5.1/5.2/5.3
  - **Evidence**: 校验模式退出码 0（「六类 + G 类全部通过」）；重复 `--generate` 跳过写盘退出码 0；`--json` 输出 `ok=True failures=0`

- [x] CP-R4: 文档与门禁同步
  - **Type**: `rule`
  - **Covers**: 文档同步 AC、TR-3.1/4.1
  - **Evidence**: `scripts/README.md` 更新为「六类 + G 类子模块指针对账」；`CONTRIBUTING.md` 新增 §3.6「子模块指针变更流程（G2 门禁）」（三步流程 + 禁止裸指针变更）

- [x] CP-U1: 指针变更流程显式化
  - **Type**: `rubric`
  - **Covers**: 指针变更流程显式化 AC（0-2 分）
  - **Scale**: 0-2
  - **Anchors**: 0 = 门禁未覆盖指针，依赖人工；1 = G2 落地但非结构化版本引用仍偶发失真（G3 仅 warn）；2 = G2 完整拦截，历史中不再出现 `modified: vendor` 类裸提交
  - **Pass Threshold**: >= 1
  - **Evidence**: 得分 **2**——G2 拦截真实漂移、`--generate` 拒绝固化、CONTRIBUTING §3.6 禁止裸指针变更，门禁首跑即验证「漂移被探照灯照出」

- [x] CP-U2: 文档版本引用对账（G3）
  - **Type**: `rubric`
  - **Covers**: G3
  - **Scale**: 0-2
  - **Anchors**: 0 = 无对账；1 = 有对账但误报；2 = 有对账且无对象时静默通过
  - **Pass Threshold**: >= 1
  - **Evidence**: 得分 **2**——CONTRIBUTING §3.5 当前无版本号引用（G3 无对象），脚本路径执行无误报；未来文档引入版本号即触发对账

## Review History

### Review R1
- **Result**: `pass`
- **Evidence**:
  - 全部检查点（CP-R1~R4、CP-U1~U2）通过
  - 每条 spec AC/TR 均有独立证据（真实命令输出 + 真实漂移捕获）
  - 无未初始化子模块干扰（4 个子模块均已初始化）
  - 无可行动发现、无受阻检查
