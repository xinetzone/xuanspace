# xuan-compose 问题处置台账（单一事实源）

> 本台账合并自两份时点报告的发现清单，是 R/N/U/P2/P1 编号的**唯一权威位置**：
>
> - §A/§B 来自 T11 独立对抗审查（[review 报告](review/2026-09-16-t11-adversarial-review.md) §6，2026-09-16）；
> - §C/§D/§E 来自功能完整性独立复验（[verification 报告](verification/2026-09-16-full-parity-verification.md) §3-§5，2026-09-16）。
>
> 处置判据（T11 审查原话）：发现按「评估保留（重构有意决策）」与
> 「上游固有（对等重构不改，列上游回馈候选）」两类闭环；对等重构中夹带修改
> 上游行为会偏离 1.6.0、破坏对等快照。

<a id="a-keep"></a>
## A. 评估保留（重构有意决策）

<a id="ledger-r1"></a>
### R1 argparse.Namespace DTO 渗透（架构 minor，保留）

commands/engine/pull/translate 共 15 文件用 `argparse.Namespace` 作参数袋类型。
审查确认 parser 机械（add_argument/parse_args/argv）完全封在 cli，下层只读
Namespace 属性、无任何解析；Handler 契约 `Callable[[Any, Namespace], Awaitable]`
为 T7 显式设计（[differences.md](parity/differences.md) M6）。改为中立 `Args`
Protocol 触及 15 文件、940 测试，收益边际、回归面大，**本期保留**；
后续若要让纯库脱离 argparse 类型，可在 types.py 引入 Protocol 并由 cli 适配，列入 backlog。

<a id="ledger-n1"></a>
### N1/N2 跨层下划线名与缺失 `__all__`（架构 nit，保留）

`_task_cancelled` 下划线名跨 2 个命令模块、4 个 BASE 模块缺 `__all__`：
实证无符号泄漏、行为等价（[differences.md](parity/differences.md) D9 已登记），
纯封装形式，本期不动以控范围。

<a id="ledger-n3"></a>
### N3 ruff 版本地板不封顶（工程化 backlog）

不同 ruff 版本对 UP038 默认集有漂移；建议 CI pin `ruff==0.15.*`（工程化 backlog，非代码缺陷）。

<a id="b-upstream"></a>
## B. 上游固有问题（对等重构不修改，列上游回馈候选）

按 translate-then-move 纪律，修这些会偏离 1.6.0 行为、破坏对等快照，故保持原样并登记：

<a id="ledger-u1"></a>
- **U1** `--verbose` 下完整 argv（含 `-e` 明文密钥）入 INFO 日志（security，最有回馈价值，建议脱敏）。
- **U2** `rec_deps` 对不含起点的子环（A→B→C→D→B）剪枝失效，可 RecursionError（integrity）。
- **U3** `include` 无 visited 去重：自包含无限追加、菱形包含重复 extend list（integrity）。
- **U4** 负数/0 `scale` 产生零容器后下游 IndexError 裸栈；`--scale svc` 无等号同（boundary）。
- **U5** `config_hash` 的 json.dumps 无 `default=str`，YAML 日期标量/残留标签 TypeError（fuzz）。
- **U6** `existing_containers` 对非数组 JSON/空 Names 缺防御（boundary）。
- **U7** `COMPOSE_PARALLEL_LIMIT=0` → Semaphore(0) 永久挂起（boundary）。
- **U8** 命令执行期 podman 二进制中途消失无 FileNotFoundError 守卫（仅版本探测有）（boundary）。
- **U9** compose 文件非法 UTF-8 抛 UnicodeDecodeError 裸栈（fuzz）。
- **U10** `PODMAN_COMPOSE_POD_ARGS` 环境变量为字符串时与 list 拼接 TypeError（boundary）。
- **U11** `_resolve_context_dependencies` 缺失服务先 KeyError，其后 None 检查为死代码（integrity；T9 测试按真实行为断言 KeyError 路径）。
- **U12** bind/secret/extends 支持绝对路径/`..`/`~`（security，**设计如此**：compose 规格即允许挂载宿主路径，argv 无 shell 语义，逃逸边界归 podman）。
- **U13** runner 取消后不重抛 CancelledError、不 await 日志 drain 任务（timing，上游刻意的 abort 语义）。

以上均不影响「重构相对上游行为等价」的结论；建议未来以独立 PR 向上游回馈 U1-U3，
本库若修复须同步偏离登记并在快照/测试中显式标注，不得在对等重构中夹带。

<a id="c-p2"></a>
## C. 独立复验 P2 遗留（登记/加固类，非功能丢失，均未改代码）

<a id="ledger-p2-1"></a>
| # | 项 | 证据 | 定级理由 |
|---|---|---|---|
| P2-1 | `normalize_service` 新增 3 处 `assert`（depends_on override 分支） | 新库 [normalize.py:207](../src/xuan_compose/normalize.py#L206-L214) vs 上游 podman_compose.py:2138-2144 | **经回归对抗降级**：真实 YAML 入口下 mapping 形态在上游构造期即抛 AttributeError（上游 L1777-1778 tuple 无 `.value`），分歧仅在绕过公开构造契约手工注入 dict 时可观察（上游产出 dict、新库 AssertionError）；且 `python -O` 剥除 assert 后行为漂移。建议改回上游同构写法或补登记 |
| P2-2 | `COMPOSE_DEFAULT_LS` 双份副本 | discovery.py:14-29 与 engine.py:59-74 各一份 14 项（逐字一致） | 当前无行为差，但违反单一事实源，单边修改即静默分叉；[symbol-map.md](parity/symbol-map.md) 只登记了 discovery 落点 |
| P2-3 | D9 登记不完整：`_task_cancelled` 不仅是"提升" | logs.py:52-55 | 函数体折叠为 `bool(cancelled() or cancelling())` 并删除 3.11 版本守卫（requires-python≥3.14 下等价），D9 措辞未覆盖此重写。**处置（2026-10-10 闭环）**：F2 当时已落地代码，[differences.md D9](parity/differences.md#d9) 已补记折叠与守卫删除，本项关闭 |
| P2-4 | 版本号双轨 | pyproject 0.1.0 vs `__version__` 1.6.0+xuan.1 | `pip show` 与 `version` 命令不一致；CHANGELOG 已述"独立版本号"，[differences.md](parity/differences.md) D7 未登记元数据分裂 |
| P2-5 | logger 名与外观文本 | 固定 `xuan_compose`（上游随入口 `podman_compose`/`__main__`） | 级别/退出码无影响，未登记 |
| P2-6 | systemd unit/env 命名空间仍共享 `podman-compose@` | commands/systemd.py:93、engine.py:581 与上游逐字一致 | D4 要求模板逐字保留；与上游二进制同机并存会互相覆盖 unit，属有意对等的连带后果，建议文档点名 |
| P2-7 | bash completion 仓库资产未携带 | 上游 completion/bash/podman-compose（411 行） | 上游 wheel 本身也不安装，wheel 行为面对等；仅 tarball/发行版渠道差异，且其命令清单已落后于 24 命令 |
| P2-8 | updown 深层编排无单测 | run_container 零名称引用（20 行薄包装，经 compose_up 间接执行）；compose_up 的 attached/SIGINT/健康等待/config-hash 重建/孤儿清理分支零单测 | **继承自上游**：上游同样无单测，由 190 个需真实 podman 的集成测试承载，非重构退化；深层回归依赖集成环境，文档应明示。2026-10-10 已在 [upstream-quirks.md「T7 清单固化段」](parity/upstream-quirks.md)逐点登记 L2/L3 并与本项互链，改动通道为人工对拍上游 1.6.0 |
| P2-9 | 2 个轮询测试断言偏弱 | tests/test_dependencies_extra.py:107-129 | 仅靠 side_effect 序列不报错通过，未断言 `await_count`，存在短路假绿空间；happy_path 同类用例已有次数断言，可补齐。2026-10-10 按固化级别新口径定性为 **L1 弱断言**（判定方法见 [upstream-quirks.md 固化级别图例](parity/upstream-quirks.md)），仍开放 |
| P2-10 | Python <3.14 误用晦涩 | py3.13 下 `import xuan_compose.cli.main` 抛 `NameError: OverrideTag` | M1 移除 future annotations 后依赖 PEP 649；requires-python≥3.14 可挡住 pip 安装，契约外缺陷，PYTHONPATH 直跑/IDE 误用无友好提示 |

<a id="d-p1-fixed"></a>
## D. 验证期已修复：分发面 P1（纯 Python 包产出平台钉死 wheel）

2026-09-16 独立复验中发现并修复、经干净 PEP 517 构建实证（`py3-none-any` wheel，
940 测试基线不变）。问题、修复与实证全文见
[verification 报告 §4](verification/2026-09-16-full-parity-verification.md#section-4)。

同源缺陷提示：脚手架模板 `tools/templates/python/CMakeLists.txt` 仍会给未来纯 Python
子项目生成同类配置，建议在模板层同步移除 CMakeLists 并改用 `wheel.cmake=false`
（属 tools 区域变更，验证当次未触碰，留待决策）。

<a id="e-env"></a>
## E. 环境层面两个 P1（非产品代码缺陷，影响验证可复现性）

1. **陈旧 editable 快照劫持导入**：两个独立代理 independently 发现 py314 site-packages
   残留 D14 前的 editable 物理快照/finder，可能让"全绿"测到旧代码。复跑门禁前应重装
   editable；WSL 权威门禁使用 `PYTHONPATH=src` 不受影响；CI 干净环境不受影响。
2. **原生 Windows 34 failed**：全部为 `/` vs `\`、盘符、HOME 展开的 POSIX 路径假设差异，
   无一条逻辑失败；权威门禁在 WSL（POSIX）。发布目标平台应在 README 明示。

<a id="f-freeze-levels"></a>
## F. 固化强度分级体系与 L1→L0 加固（2026-10-10 闭环）

- **建立四级固化口径**：L0 精确断言 / L1 弱断言 / L2 注释固化 / L3 清单固化。
  图例与全部怪癖点的逐点级别登记在 [upstream-quirks.md](parity/upstream-quirks.md)；
  方法论本体（断言变异敏感度、L1 全绿假象、L3 人工对拍上游基线通道、级别可升降）
  并入主权区模式 behavior-characterization-freeze「固化强度分级」小节。
- **L1→L0 加固一处**：create_secrets 报错串第二个片段漏 f 前缀，原测试仅
  `assertRaises(ValueError)`（细节变异不红），已改为报错全文等值断言
  （[test_commands_build_updown.py:188-196](../tests/test_commands_build_updown.py#L188-L196)）；
  28 例回归通过。
- **P2-8 互链**：SIGINT 监督、config-hash 重建等深层分支在「T7 清单固化段」
  按 L2/L3 逐点登记，改动前须对拍上游 1.6.0（pinned `e3df104`）。
- **遗留**：P2-9 的两个轮询弱断言已按新口径定性为 L1，尚未升级，仍开放。
