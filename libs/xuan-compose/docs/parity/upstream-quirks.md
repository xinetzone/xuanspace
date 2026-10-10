# 逐行保留的上游怪癖（未顺手「修正」，固化级别逐点标注 L0-L3）

> 新写测试时若实际行为与「合理预期」冲突，先假设测试错、回到上游代码核对；
> 确认是上游行为则按真实输出断言并登记本清单——为让断言符合直觉而改生产代码，
> 就是在制造与上游的不对等。本项目有 10+ 次此类纠正。

## 固化级别图例

| 级别 | 判定标准 | 变异敏感度 |
|---|---|---|
| **L0** 精确断言 | 断言直接落在怪癖输出值上（等值/集合/文本） | 怪癖点一变，测试即红 |
| **L1** 弱断言 | 仅断言异常类型、返回码等粗粒度形态 | 细节变异不红，假绿风险最高 |
| **L2** 注释固化 | 仅 docstring/代码注释登记，无行为测试 | 无自动化拦截 |
| **L3** 清单固化 | 仅本清单（及台账）文本登记 | 改动须人工对拍上游 1.6.0 |

> L0/L1 为 T9 测试固化段；L2/L3 汇总见文末「T7 清单固化段」。
> 级别可随 fake/桩能力增强升级（L3→L0），随上游正式修复关闭条目。

- argparse：
  - 无参数/help 退出码 **-1**（非 1）。【L0】
  - `--parallel` 默认 `sys.maxsize`、读 `COMPOSE_PARALLEL_LIMIT`。【L0】
  - `PullPolicyAction` 对 `--pull-always false` 静默 return 不赋值。【L0】
  - cp 子命令 parser 函数名沿用上游笔误 `compose_parse_cp`。【L3】
- 时间解析 `str_to_seconds` 正则 `[m:]`：
  - 合法写法是 `"1:30"`/`"1m30s"`，`"1m:30s"` 反而解析失败返回 None。【L0】
  - 只接受 int 秒（podman 限制，原文注释保留；`"1.5s"` 取整为 1）。【L0】
- `is_list(bytes)` 为 True（bytes 可迭代且非 str/dict）。【L0】
- 路径：
  - normalize 对 volumes 短格式/env_file/sub_dir 的 `./` 前缀**不剥离**（仅 build.context 分支剥离）。【L0】
  - 对整个卷规格串做 `os.path.join`（含 `:/target`）。【L0】
- 合并：
  - service 自身的 `extends` 声明在 `resolve_extends` 后保留在合并结果。【L0】
  - `norm_as_list` 对 None 值输出裸键（`B` 而非 `B=None`）。【L2】
  - `--service-ports=False` 时先删 ports 再由 `--publish` 重建（原 ports 不保留）。【L0】
  - OverrideTag 的自定义 from_yaml 直接取 ScalarNode.value（无隐式类型解析，标量即字符串）。【L0】
- 依赖：`rec_deps` 对自依赖/未知依赖"保留边但不递归展开"；A→B→A 环按起点剪枝。【L0】
- 镜像上下文 `_resolve_context_dependencies`：`urllib.parse.quote` 会百分号编码镜像 tag 的冒号（`b:1` → `b%3A1`）。【L0】
- handler 怪癖原样保留清单（T7 登记）：
  - build 拓扑 as_completed 调度。【L1】
  - up 的 SIGINT 监督循环。【L3】
  - up 的 config-hash 重建判定。【L3】
  - down 的 `--rmi local` 删除恰为 is_local 的镜像。【L0】
  - kill 的 all/services 重复分支。【L0】
  - logs `max()` 空序列 ValueError。【L2】
  - ls `except Exception: break`。【L3】
  - ls json `print(list)` 非标准输出。【L0】
  - create_secrets 报错串第二个片段漏 f 前缀（2026-10-10 由 L1 补全文断言升级）。【L0】
  - networks invalid network_mode 的 `log.fatal`+sys.exit(1)。【L3】

## T7 清单固化段（L2/L3，无精确行为断言）

以下各点无 L0 自动化拦截，修改前须沿真实入口人工对拍上游 1.6.0
（pinned `e3df10472e194ab6d547b5ad25542c5c79e1a5fb`）输出并登记对拍结果；
不得在对等重构中顺手改动：

- 【L3】cp parser 函数名笔误 `compose_parse_cp`（另见 [symbol-map.md](symbol-map.md) CLI 层登记）。
- 【L2】`norm_as_list` 对 None 值输出裸键（函数 docstring 有示例，无断言）。
- 【L1】build 拓扑 as_completed 调度：现有失败批次用例仅断言返回码，调度形态变异（改顺序 await/gather）不红。
- 【L3】up 的 SIGINT 监督循环（深层背景与不可测性见 [findings-ledger.md P2-8](../findings-ledger.md)）。
- 【L3】up 的 config-hash 重建判定。
- 【L2】logs `max()` 空序列 ValueError（logs handler docstring 登记）。
- 【L3】ls `except Exception: break`。
- 【L3】networks invalid network_mode 的 `log.fatal`+sys.exit(1)。
