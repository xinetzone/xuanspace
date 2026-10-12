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

按 translate-then-move 纪律，修这些会偏离上游行为、破坏对等快照，故保持原样并登记。

> **复核状态（2026-10-12 双端对拍）**：本节 13 条已对新库源码与上游 pinned 基线
> [`1f1cb09`](https://github.com/containers/podman-compose/blob/1f1cb09bfee4d79857ed23bfe2699febd7d20a53/podman_compose.py)
> 逐条复核——经 GitHub API 核实该提交即上游 main HEAD（提交日 2026-09-29）。
> 结论：**U1-U11、U13 行为在双端逐字同构，缺陷全部仍开放（0 项被上游关闭）**；
> U12 为规格允许项，非缺陷。下文上游行号均指 `podman_compose.py @ 1f1cb09`，
> 新库锚点均为 2026-10-12 实测行号。**联动规则：此后每次推进上游基线（见 §G），
> 必须逐条复核本表并回填本节状态，不得默认沿用历史结论。**

### B.1 回馈候选总览（U1-U11，按优先级排序）

优先级口径：**P1** 安全类（默认配置即可泄露）｜**P2** 普通配置即可触发的完整性/挂起｜
**P3** 仅畸形输入触发的边界加固。

| 编号 | 类别 | 优先级 | 摘要 | 上游议题踪迹（2026-10-12 检索） |
|---|---|---|---|---|
| U1 | security | **P1** | `--verbose` 下完整 argv（含 `-e` 明文密钥）入 INFO 日志 | 无对应议题 |
| U2 | integrity | **P2** | `rec_deps` 对不含起点的子环（A→B→C→D→B）剪枝失效，可 RecursionError | [#76](https://github.com/containers/podman-compose/issues/76) 为 extends 自引用场景，2019 年已修，形态不同 |
| U3 | integrity | **P2** | `include` 无 visited 去重：环包含无限追加、菱形包含重复合并 | PR [#804](https://github.com/containers/podman-compose/pull/804) 实现过环检测+单次解析，**未合并即关闭**（2025-08-04） |
| U7 | boundary | **P2** | `COMPOSE_PARALLEL_LIMIT=0` → Semaphore(0) 永久挂起 | 无对应议题 |
| U4 | boundary | P3 | 负数/0 副本产生零容器后下游 IndexError；畸形 `--scale` 取参裸 IndexError | 无 |
| U5 | fuzz | P3 | `config_hash` 的 json.dumps 无 `default=str`，YAML 日期标量 TypeError | 无 |
| U6 | boundary | P3 | `existing_containers` 对非数组 JSON/空 Names 缺防御 | 无 |
| U8 | boundary | P3 | 命令执行期 podman 二进制中途消失无 FileNotFoundError 守卫（仅版本探测有） | 无 |
| U9 | fuzz | P3 | compose 文件非法 UTF-8 抛 UnicodeDecodeError 裸栈 | 无（[#1374](https://github.com/containers/podman-compose/issues/1374) 等针对容器输出流，不同路径） |
| U10 | boundary | P3 | `PODMAN_COMPOSE_POD_ARGS` 环境变量为字符串时与 list 拼接 TypeError | 无（[#1359](https://github.com/containers/podman-compose/issues/1359) 为相邻区域，非同一缺陷） |
| U11 | integrity | P3 | `_resolve_context_dependencies` 缺失服务先 KeyError，其后 None 检查为死代码 | 无 |

> 回馈节奏建议：U1 单独 PR 最先（安全类、修复最小）；U2/U3/U7 各配最小复现，
> 并显式引用 #76/#804 说明残余形态，防止被以"已处理"关闭；P3 四项 boundary 与
> 两项 fuzz 可分别合并为 robustness PR。所有 PR 须按上游测试风格附最小回归用例。

### B.2 逐条证据、最小触发与关闭条件（冻结级别 L3）

> L3 = 仅以台账文本固化、无测试断言（四级固化口径见 §F 与 upstream-quirks.md 图例）。
> 其中 U2（含 A→B→A 控制组）、U3（自包含+菱形）、U4、U7 的关键机制已在 V 阶段
> 用包内真实代码执行复现（Python 3.14.7，2026-10-12，全部按断言复现）。

<a id="ledger-u1"></a>
- **U1（security · P1）** `--verbose` 下完整 argv 入 INFO 日志，含 `-e KEY=SECRET`
  明文环境变量；默认 WARN 级别不可见，开启 verbose 后三条命令路径全部打印。
  - 锚点：新库 [runner.py:85](../src/xuan_compose/runner.py#L85)（output）、
    [:159](../src/xuan_compose/runner.py#L159)（exec）、
    [:178](../src/xuan_compose/runner.py#L178)（run）；上游 L1891/L1963/L1982，逐字同构。
  - 最小触发：服务带 `environment: {TOKEN: 明文}` 执行 `podman-compose --verbose up`。
  - 关闭条件：上游对敏感参数脱敏（或日志降级并在文档示警）后关闭。

<a id="ledger-u2"></a>
- **U2（integrity · P2）** `rec_deps` 的环剪枝条件
  `any(start_point == x.name for x in dep_srv["_deps"])` 只覆盖含起点的二元环；
  不含起点的子环仍无限递归。
  - 锚点：新库 [dependencies.py:33-54](../src/xuan_compose/dependencies.py#L33-L54)；
    上游 L1763-L1784，逐字同构。正常工作侧（A→B→A 剪枝）已按 L0 固化，
    见 [upstream-quirks.md 依赖段](parity/upstream-quirks.md)。
  - 最小触发：`depends_on` 构造成 A→B→C→D→B（环 B-C-D 不含起点 A），
    `flat_deps` 期 RecursionError。
  - 上游历史：[#76](https://github.com/containers/podman-compose/issues/76) 报告的是
    extends 跨文件自引用（odoo extends odoo），2019 年修复；回馈时须说明本项是
    不同输入形态的残余缺陷，并附上述子环复现。
  - 关闭条件：上游引入 visited 集合或显式环报错并发布。

<a id="ledger-u3"></a>
- **U3（integrity · P2）** include 处理无 visited 去重，存在两个独立子现象：
  ① 环包含（A 包含 A，或 A→B→A）无限追加——每轮从磁盘重读的内容经 `rec_merge`
  重新带回 `include` 键，`del compose["include"]` 只删合并结果、无法阻止再追加；
  ② 菱形包含（A→B→D、A→C→D）中 D 被追加、读取、合并两次，list 字段经
  `value.extend(value2)` 重复。
  - 锚点：新库 [engine.py:466-501](../src/xuan_compose/engine.py#L466-L501)
    （del 在 :501）与 [merge.py:191-193](../src/xuan_compose/merge.py#L191-L193)；
    上游 L2904-L2937（del 在 L2937）与 L2347-L2349，逐字同构。
  - 上游历史：PR [#804](https://github.com/containers/podman-compose/pull/804)
    描述明确包含"检测环包含"与"每个文件只解析一次"，2025-08-04 关闭但**未合并**；
    其后 include 重写为 dict/`path:` 形式时未补回该两项能力（2026-10-12 经
    GitHub API 检索 main 全量源码，无 cyclic/visited 痕迹）。
  - 关闭条件：上游加入 visited 去重并对环包含显式报错。

<a id="ledger-u4"></a>
- **U4（boundary · P3）** scale 无值域与格式校验：负数/0 副本数使
  `range(1, replicas + 1)` 为空，下游首个 `[0]` 索引点裸 IndexError；
  `--scale <服务名>`（缺等号且命中同名服务）在 `split("=")` 后取 `[1]` 同样裸 IndexError。
  - 锚点：新库 [engine.py:594-612](../src/xuan_compose/engine.py#L594-L612)；
    上游 L3029-L3042；下游索引点新库
    [runexec.py:28](../src/xuan_compose/commands/runexec.py#L28)、
    [container_args.py:207](../src/xuan_compose/translate/container_args.py#L207)、
    [networks.py:158](../src/xuan_compose/translate/networks.py#L158)。
  - 最小触发：`scale: 0`（或 `deploy.replicas: 0`、`--scale svc=0`）
    后执行 run/up，下游取空列表首项越界；或对**同名已存在服务**执行
    `--scale <服务名>` 缺 `=N`，匹配分支取 `scale_args[1]` 即越界。
  - 关闭条件：replicas ≤ 0 显式报错、`--scale` 格式校验。

<a id="ledger-u5"></a>
- **U5（fuzz · P3）** `config_hash` 的 `json.dumps(jsonable_service, sort_keys=True)`
  无 `default=str`；`original_configuration` 只递归 dict，非 JSON 原生标量原样到达 dumps。
  - 锚点：新库 [engine.py:149](../src/xuan_compose/engine.py#L149)；上游 L2585。
  - 最小触发：service 配置含未加引号的 YAML 日期标量（safe_load 得 `datetime.date`）
    或残留自定义标签对象，up 重建判定计算哈希时 TypeError。
  - 关闭条件：`default=str` 或配置加载期将非标量标量规范化。

<a id="ledger-u6"></a>
- **U6（boundary · P3）** `existing_containers` 对 `podman ps --format json`
  输出无类型防御：`json.loads` 结果非数组（null/对象）、条目 `Names` 为空、
  或 `Labels` 为 null 时裸异常。
  - 锚点：新库 [runner.py:290-292](../src/xuan_compose/runner.py#L290-L292)；
    上游 L2094-L2097，逐字同构。
  - 最小触发：ps 返回 `null` → `for c in None` TypeError；条目 Names 为空 →
    `None[0]` TypeError；Labels 为 null → `None.get(...)` AttributeError。
  - 关闭条件：数组类型校验与 Names/Labels 空值防御。

<a id="ledger-u7"></a>
- **U7（boundary · P2）** `COMPOSE_PARALLEL_LIMIT=0` 时 `asyncio.Semaphore(0)`，
  所有子进程协程在 `async with self.semaphore` 处永久等待，表现为命令挂起且无输出。
  - 锚点：新库 [parser.py:168](../src/xuan_compose/cli/parser.py#L168)
    （`--parallel` 默认读环境变量）与 [main.py:47](../src/xuan_compose/cli/main.py#L47)
    （Semaphore 构造）；上游 L3332 与 L2542，逐字同构。
  - 最小触发：`COMPOSE_PARALLEL_LIMIT=0 podman-compose up`（V 阶段实证：
    env 字符串经 argparse `type=int` 转为 0，acquire 永久挂起；负值在
    Python 3.14 直接 ValueError，症状不同但同源——均无值域校验）。
  - 关闭条件：值 ≤ 0 时回退默认并发或显式报错。

<a id="ledger-u8"></a>
- **U8（boundary · P3）** FileNotFoundError 守卫仅存在于启动期版本探测；
  命令执行途中 podman 二进制消失（卸载/替换/路径失效）时，
  `create_subprocess_exec` 抛裸 FileNotFoundError。
  - 锚点：探测守卫新库 [main.py:60](../src/xuan_compose/cli/main.py#L60)
    （上游 L2551）；无守卫执行点新库 [runner.py:86](../src/xuan_compose/runner.py#L86)、
    [:189](../src/xuan_compose/runner.py#L189)、
    [:215](../src/xuan_compose/runner.py#L215)、
    [:224](../src/xuan_compose/runner.py#L224)。
  - 关闭条件：在 Podman 子进程边界统一转友好错误信息。

<a id="ledger-u9"></a>
- **U9（fuzz · P3）** compose 文件以 `open(filename, encoding="utf-8")` 读取，
  无 `errors` 参数与捕获；非法 UTF-8 字节产生 UnicodeDecodeError 裸栈。
  - 锚点：新库 [engine.py:404](../src/xuan_compose/engine.py#L404)；上游 L2840。
  - 最小触发：compose 文件写入非法 UTF-8 字节序列后执行任意命令。
  - 区分：**容器 stdout/stderr 流**非 UTF-8 的情形在上游基线代码中已由增量解码器
    处理（1f1cb09 L1915；议题 [#1374](https://github.com/containers/podman-compose/issues/1374)
    仍 open、PR [#1389](https://github.com/containers/podman-compose/pull/1389) 关闭
    未合并，修复系经其他提交落地），与本项的配置文件读取路径无关。
  - 关闭条件：读取期捕获并报"文件 X 含非法 UTF-8 编码"。

<a id="ledger-u10"></a>
- **U10（boundary · P3）** 环境变量通道将 `PODMAN_COMPOSE_*` 值原样按字符串存入
  `x_podman`；`resolve_pod_args` 只对 CLI `--pod-args` 通道做 `shlex.split`，
  env 通道命中时返回 str，调用处 `list + str` 触发 TypeError。
  - 锚点：新库环境映射 [engine.py:234-239](../src/xuan_compose/engine.py#L234-L239)、
    解析 [engine.py:195-204](../src/xuan_compose/engine.py#L195-L204)、
    拼接点 [updown.py:57](../src/xuan_compose/commands/updown.py#L57)；
    上游 L2670-L2675 与 L2631-L2640。
  - 最小触发：设置 `PODMAN_COMPOSE_POD_ARGS=--infra=false`，且无 CLI `--pod-args`
    覆盖、无 x-podman 配置时执行 up。
  - 关闭条件：env 值统一 `shlex.split` 或类型校验；相邻议题
    [#1359](https://github.com/containers/podman-compose/issues/1359) 是 compose 文件
    pod-args 不生效，非同一缺陷，回馈时勿混淆。

<a id="ledger-u11"></a>
- **U11（integrity · P3）** `services[target_service_name]` 下标在缺失服务时先抛
  KeyError，其后的 `if target_service is None` 永不可达（dict 下标不返回 None），
  友好 ValueError 文案为死代码。
  - 锚点：新库 [engine.py:726-727](../src/xuan_compose/engine.py#L726-L727)；
    上游 L3162-L3163；同函数的镜像 tag 百分号编码怪癖已按 L0 固化
    （[upstream-quirks.md 镜像上下文段](parity/upstream-quirks.md)），缺陷侧与怪癖侧互补。
  - 最小触发：`build.additional_contexts` 引用 `service:不存在的服务`。
  - 测试现状：T9 用例按真实行为断言 KeyError 路径（对等，不改写）。
  - 关闭条件：改 `.get()` 取值并让 ValueError 路径可达。

### B.3 评估为非缺陷 / 上游刻意语义，不列入回馈

<a id="ledger-u12"></a>
- **U12（design · 不回馈）** bind/secret/extends 支持绝对路径、`..`、`~`：
  compose 规格本身允许挂载宿主路径，argv 不经 shell 解释，路径逃逸边界归 podman
  与调用者责任，非本库缺陷。
  - 锚点：新库 [mounts.py:68](../src/xuan_compose/translate/mounts.py#L68)、
    [:147](../src/xuan_compose/translate/mounts.py#L147)、
    [build.py:37](../src/xuan_compose/translate/build.py#L37)、
    [secrets.py:66](../src/xuan_compose/translate/secrets.py#L66)
    （`expanduser`/`abspath`/`realpath`）。

<a id="ledger-u13"></a>
- **U13（timing · 刻意语义，暂不回馈）** runner 取消时 terminate→等待 10 秒→kill，
  吞掉 CancelledError 并返回 exit_code，up 监督循环据此把取消当正常终止。
  - 锚点：新库 [runner.py:228-239](../src/xuan_compose/runner.py#L228-L239)；
    上游 L2030-L2043，逐字同构；调用侧注释见
    [updown.py:169](../src/xuan_compose/commands/updown.py#L169)。
  - 两个子面分开定性：① 取消后不重抛 CancelledError = 上游刻意的 abort 语义，
    **不得改动**；② 不 await `out_t`/`err_t` 日志 drain 任务，取消瞬间可能丢失
    尾行日志或产生 task pending 告警——若未来回馈，仅就 ② 提最小 PR。

以上均不影响「重构相对上游行为等价」的结论。回馈一律以独立上游 PR 按 B.1
优先级进行；本库若自行修复任一条，须同步偏离登记（differences.md）并在快照/测试中
显式标注，不得在对等重构中夹带。基线推进时的复核义务见本节顶部横幅与 §G。

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
| P2-9 | 2 个轮询测试断言偏弱 | tests/test_dependencies_extra.py:107-149 | 原仅靠 side_effect 序列不报错通过，未断言调用次数；2026-10-10 按固化级别新口径定性为 **L1 弱断言**。**2026-10-10 闭环（L1→L0）**：两例改为 `call_args_list` 全序列等值断言（两次 status 轮询→wait→终态 inspect）+ `sleep.assert_awaited_once_with(0.05)`，删除已无引用的 `_instant_sleep` 桩；两次注入变异（created 短路 break / inspect 异常不重试）均被对应用例精确拦截，12 例本文件全绿 |
| P2-10 | Python <3.14 误用晦涩 | py3.13 下 `import xuan_compose.cli.main` 抛 `NameError: OverrideTag`（命中点：merge.py 类体内 4 处自引用注解 L69/81/99/103，PEP 649 前定义时求值） | **2026-10-12 闭环**：`xuan_compose/__init__.py` 顶部（先于一切第三方依赖）新增 `MINIMUM_PYTHON_VERSION=(3,14)` 与 `require_python_version()` 导入时守卫，低版本抛含最低版本/当前版本/requires-python 引导的友好 `RuntimeError`；守卫函数刻意不用 PEP 604 注解以在旧解释器可执行。tests/test_python_version_guard.py 10 例 L0 断言（阈值/边界/消息片段，先红后绿，阈值变异即红已还原）；Python 3.12.2 端到端实证 `import xuan_compose` 与 `import xuan_compose.cli.main` 均得友好 RuntimeError（环境未装 PyYAML，证明守卫先于依赖导入）；双门禁 976 passed / 覆盖率 92%。对 3.14+ 行为零影响，不改任何上游对等路径 |

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
2. **原生 Windows 34 failed（2026-10-12 闭环）**：全部为 `/` vs `\`、盘符、HOME
   展开的 POSIX 路径假设差异，无一条逻辑失败。根因是测试期望硬编码 POSIX 路径串，
   而生产代码逐行保留上游 `os.path.join`/`abspath`/`realpath`/`expanduser` 的本机
   语义（上游 CI 矩阵仅 Linux 容器，上游单测在 Windows 同样会红）。修复纪律：**不动
   src/ 生产代码、不用 skipif**，只把 9 个测试文件的期望侧改为用同一 `os.path` 原语
   表达跨平台等值断言；其中 `test_build_ssh_path_home_dir` 在 nt 额外注入
   `USERPROFILE`（`expanduser` 在 nt 读该变量），chdir 到临时根的
   `test_parse_compose_file_build.py` 在测试体内（chdir 后同一盘符）换算期望。
   已用 posixpath 语义逐式证明 22 条期望表达式在 POSIX 下与上游字面量逐字节一致，
   且断言仍为全值等值（L0，未削弱）。双门禁实证：Windows 本机 966 passed /
   79 subtests，WSL 权威门禁 `PYTHONPATH=src pytest --cov` 966 passed / 覆盖率 92%。

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
- **遗留关闭**：P2-9 的两个轮询弱断言已于 2026-10-10 升级为 L0（全调用序列
  等值 + sleep 一次断言），经两处定向变异验证"变异即红"，本条遗留清零。

<a id="g-upstream-sync"></a>
## G. 上游增量同步 e3df104 → 1f1cb09（2026-10-10）

上游第三方只读 submodule `vendor/podman-compose` 由 pinned 基线 `e3df10472e194ab6d547b5ad25542c5c79e1a5fb`
（v1.6.0）推进至 `1f1cb09bfee4d79857ed23bfe2699febd7d20a53`。区间 11 个非合并提交
（`332cfdb`、`f01d26d`、`816045e`、`03649a3`、`89a2033`、`d1088b3`、`008ef4e`、`fd64dc2`、
`d8dea76`、`29760c4`、`ee5ad43`）已全部按对等纪律移植进本库（Task 1-7），`podman_compose.py`
净 +143/−70（5541 → 5614 行）。

### 上游提交 → 新库模块对等移植映射

| 上游提交 | 新库落点 | 内容 |
|---|---|---|
| `fd64dc2` | `merge.py` + `normalize.py` | `!override` 语义链（from_yaml/to_yaml/rec_subs/rec_merge_one/normalize_service） |
| `816045e` | `translate/networks.py` | 外部网络 dict 形态 `name:` |
| `f01d26d` | `normalize.py` | `volumes: null` 显式报错 |
| `008ef4e` | `runner.py` + `commands/runexec.py` | `pass_stdin`（仅 run/exec 转发 stdin） |
| `03649a3` | `translate/run_args.py` | `exec --detach` |
| `332cfdb` | `cli/parser.py` | `up --wait` 校验与互斥 |
| `29760c4` | `translate/build.py` | `container_to_build_args` 三处 `OSError`→`PodmanComposeError` |
| `ee5ad43` | `logs.py` + `commands/updown.py` | 注释清理 |
| `d8dea76` | 不适用 | 仅 README |
| `89a2033`、`d1088b3` | 不移植 | 仅 `tests/integration/**`，按 NG2 不移植 |

### 符号面

- 上游公开符号总数仍 **142 = 函数/类 138 + 常量 4**（无符号增删）；无同名落点仍为 3
  （`PodmanCompose` L2469、`cmd_run` L3351、`cmd_parse` L3376）。
- `rec_subs` 新增 1 个 `@overload`（`OverrideTag → OverrideTag`，distinct 名称仍计 1），总账不变。
- 行号锚定：`symbol-map.md` 行号列已随本次同步统一重锚到 `1f1cb09`；`differences.md` 上游「位置」
  列中的 M3/M4/M5 等行号属登记时点证据，不随基线推进重锚。

### 怪癖条目裁决

- `upstream-quirks.md`「OverrideTag 的自定义 from_yaml 取 ScalarNode.value」条目**关闭**（上游 `fd64dc2` 修复）。
- 新增 `fd64dc2` 语义变更登记（双 `!override` 时 source 优先 / 空 mapping 得 `{}` / 标量按标记解析），
  由 `tests/test_override_environment.py`、`tests/test_merge_extra.py` 精确断言覆盖，按 L0 定性。

### 测试面

- 57 文件 965 例（+25，全部来自本次上游测试移植）；WSL 权威门禁 `PYTHONPATH=src pytest --cov`
  全绿（965 passed / 79 subtests，覆盖率 92%）。

### 遗留项与环境事实

- **P2-9** 两个轮询弱断言已于 2026-10-10 升级为 L0 并闭环（见 §C P2-9 行与 §F「遗留关闭」）。
- 原生 Windows 的 `/` vs `\` 平台差异失败（§E.2）已于 2026-10-12 经期望侧跨平台
  适配闭环（9 个测试文件、0 行生产代码改动、0 skip），Windows 与 WSL 双门禁同为
  966 passed / 79 subtests。
- 环境事实：WSL 权威门禁 venv 路径 `~/.venvs/xuan-gate`（本机由 `python3.14 -m venv` +
  `pip install pytest pytest-cov parameterized pyyaml python-dotenv` 重建；WSL 侧无 pip，需 `ensurepip`/`venv` 自带）。
- 不自动提交：本次文档回写与 Task 1-7 代码改动待用户批准后按原子提交拆分。
