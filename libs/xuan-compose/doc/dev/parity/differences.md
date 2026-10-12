# Python 3.14 兼容修复与结构性差异登记（TR-10.3）

原则：只做 3.14 必需兼容与无语义机械现代化；算法/退出码/文案逐行保留，
差异按 Task 登记，每条可回溯上游行号。

> 维护说明（2026-10-10）：本表为常青台账，条目本体随代码演进维护；
> 条目中的 Task/TR/F 编号、测试数与提交时点为**登记时点证据**，不随后续重测改写。
> 上游「位置」列中 M3/M4/M5 等行号锚定于更早的登记时点（早于 e3df104），属登记时点证据、不随后续基线推进重锚；symbol-map.md 的行号列已随 2026-10-10 同步统一锚定到 1f1cb09。

- 语法/类型现代化（无语义变化）：M1-M7
- 结构性重构（语义等价，架构必需）：D1-D15
- 有意功能扩展（上游无，Xuanspace 增量）：D16
- 上游缺陷修复（有意行为偏离，附回馈候选）：D17
- 符号总账与逐模块映射：[symbol-map.md](symbol-map.md)
- 逐行保留的上游怪癖：[upstream-quirks.md](upstream-quirks.md)

## 5.1 语法/类型现代化（无语义变化）

| # | 内容 | 影响范围 | 上游位置 | 等价理由 |
|---|---|---|---|---|
| <a id="m1"></a>M1 | 全包移除 `from __future__ import annotations`；配套在 `xuan_compose/__init__.py` 顶部加导入时版本守卫（P2-10，2026-10-12） | 8 个基础层文件（后续全包）+ 包入口 | 各文件头（上游无守卫） | PEP 649 起 3.14 注解默认惰性，future 导入冗余（ruff UP）；移除后低版本解释器会在 merge.py 类体自引用注解处抛晦涩 `NameError`，守卫把 PYTHONPATH 直跑/IDE 误用转成含当前版本与 requires-python 引导的友好 `RuntimeError`，仅在 <3.14 触发，对受支持版本与上游对等行为零影响（见 [findings-ledger P2-10](../findings-ledger.md)） |
| <a id="m2"></a>M2 | `except asyncio.TimeoutError as exc: raise TimeoutError from exc` → `except TimeoutError: raise` | runner.wait_with_timeout | 1811-1820 | 3.11+ 两者即同一内置类型（ruff UP041） |
| <a id="m3"></a>M3 | `asyncio.exceptions.IncompleteReadError/LimitOverrunError` 收敛为 `asyncio.*` 限定名 | runner._readchunk | 1020-1030 | 纯导入路径机械收敛 |
| <a id="m4"></a>M4 | `XPodmanSettingKey` 由 `PodmanCompose` 嵌套 `Enum` 提为 model.py 模块级 `StrEnum` | model.py | 2430-2480 | 嵌套枚举无法脱离单例引用；StrEnum 为 3.11+ 惯用法，键值字符串不变 |
| <a id="m5"></a>M5 | 上游 `map(...)`/zip 惰性对 → 列表推导（如 `missing = [fn for fn in files if ...]`） | engine._parse_compose_file | 2903-2917 | 等价收敛，行为一致 |
| <a id="m6"></a>M6 | 翻译层 `compose`/`cnt` 句柄统一 `Any` 鸭子类型（不反向 import engine） | translate/* | 全层 | 打破执行/翻译层反向依赖；属性访问契约与上游运行时完全一致 |
| <a id="m7"></a>M7 | 公共 API 全面类型注解：`ParserFn = Callable[[ArgumentParser], None]`、`Handler = Callable[[Any, Namespace], Awaitable[Any]]`、`cleanup_callbacks: list[Callable[[], object]]`、测试 `Union → \| None` 等 | cli/commands/translate/tests | 全局 | 3.14 完整注解，签名行为不变 |

## 5.2 结构性重构（语义等价，架构必需）

| # | 上游 | 新库 | 等价理由/证据 |
|---|---|---|---|
| <a id="d1"></a>D1 | `@cmd_run` 装饰器在单例构造期副作用注册 24 命令（含 help/desc 闭包计算） | `COMMAND_HANDLERS` 24 名显式 dict + `install_handlers(engine)`；help/desc 文本由 `COMMAND_HELP` 表 + systemd 运行时从 handler docstring 复刻装饰器切分算法（`re.sub(r"^\s+","",doc)` 无 MULTILINE） | T9 argparse 快照对拍 help/description 零差异；TR-7.1 24 名集合相等 |
| <a id="d2"></a>D2 | `@cmd_parse` 装饰器向 `handler._parse_args` append parser | `COMMAND_PARSERS: dict[str, tuple[ParserFn, ...]]` 24 键显式注册表（wait=() 空元组）+ `build_parser()` 纯构造，共享 parser 挂载顺序精确复刻 append 序 | T9 快照 11 维参数表零差异（`tests/snapshots/cli_parity.json`，上游 e3df104）——快照锚点刻意停在 e3df104（2026-10-10 推进 pinned 至 1f1cb09 的增量无 argparse 参数增删，快照内容不变，防篡改守卫见 tests/test_cli_parity.py:38） |
| <a id="d3"></a>D3 | `PodmanCompose._parse_args(self, argv)` 引擎方法（写全局 logging、回写单例） | `parse_args(engine, argv)` CLI 纯函数（engine 仅作 global_args 回写宿主，`configure_logging` 替代 basicConfig） | 回填/分流/退出码逐行一致；TR-9.1 参数表零差异 |
| <a id="d4"></a>D4 | 模块级 `script = realpath(argv[0])` 导入期副作用（L56） | `ComposeEngine.executable` 字段默认 None，`cli/main.py` 装配期单点注入 | systemd create-unit 模板渲染结果逐字一致（test_systemd*）；TR-8.3 grep 全库 argv 仅一处 |
| <a id="d5"></a>D5 | 模块级 `podman_compose = PodmanCompose()` 活单例，`run()` 操作 self | 无单例；`async_main(argv)` 显式 `ComposeEngine()` → install_handlers → parse → Podman → 分发；`main(argv=None)` 同步壳 | TR-6.4 grep 无模块级活引擎；双实例隔离测试；console script 零参调用与上游等价（argv 可选仅为测试便利） |
| <a id="d6"></a>D6 | `PodmanCompose.__init__` 内自建子进程封装、读 argv | `ComposeEngine(podman=None)` 纯状态构造；runner 由 CLI 装配注入 | TR-6.1 实例化零子进程/零 argv；T8 main 装配测试 |
| <a id="d7"></a>D7 | 容器标签版本 `io.podman.compose.version=1.6.0` | `"1.6.0+xuan.1"`（即包 `__version__`） | 唯一有意标签值差异：标识派生版本；test_container_to_args 相应标签断言已按新值移植 |
| <a id="d8"></a>D8 | `PodmanCompose.XPodmanSettingKey` 嵌套枚举 | `model.XPodmanSettingKey` 模块级 StrEnum（6 键） | 见 M4 |
| <a id="d9"></a>D9 | `compose_up` 内闭包 `_task_cancelled`（无自由变量） | 提为 `logs.py` 模块级函数；T11 F2 进一步把函数体折叠为 `bool(task.cancelled() or task.cancelling())` 并删除 `sys.version_info>=(3,11)` 死分支守卫（requires-python≥3.14，见 [review §5 F2](../review/2026-09-16-t11-adversarial-review.md#f2)） | 无自由变量，提升后行为完全一致；折叠与删守卫在 3.14 下语义等价，940 测试回归通过 |
| <a id="d10"></a>D10 | `run()`/`_parse_args()` 留在单体 | 随 CLI 层迁出；engine.commands 初始空 dict | T7 install_handlers 填充、T8 dispatch 查表 |
| <a id="d11"></a>D11 | `!override`/`!reset` YAMLObject 全局注册 | 原样保留（merge.py 导入即注册进 SafeLoader/SafeDumper），但包根 `__init__.py` 不导入 merge——`import xuan_compose` 仍无副作用 | T3 登记；行为与上游一致，包根无副作用由 TR-2.2 冒烟保证 |
| <a id="d12"></a>D12 | `prog` 随 `sys.argv[0]`（podman-compose 呈现） | argparse 默认 prog（`python -m xuan_compose`/`xuan-compose`/`xc` 呈现，见 D15） | AC-4 明确允许 prog 差异；快照序列化刻意不采集 prog |
| <a id="d13"></a>D13 | check_dep_conditions/_validate_completed_successfully 定义于 up 辅助区，T4 按 test_depends_on 归属提前迁入 dependencies.py 后，T7 在 commands/updown.py 又逐行重译了一份（重复实现） | T10 去重：删除 commands/updown.py 内的副本，改为 `from ..dependencies import check_dep_conditions` 再导出（保持符号面与测试导入路径），单一事实源 | 两份字节级一致；去重后 940 测试全绿、ruff/mypy 0 |
| <a id="d14"></a>D14 | `resolve_extends` 原置于 merge.py，但其函数体要调用 normalize 的 `rec_subs`/`normalize_service`，只能用函数内惰性 import 反向取 normalize，与 normalize 顶层 `from .merge import ...` 构成规范层内部惰性环 | T11 V 审查后下沉到 normalize.py（它本属规范化流程），内部直接用同模块函数、仅单向 `from .merge import rec_merge, load_yaml_or_die`；merge.py 删除该函数与惰性 import | 纯代码移动零行为变化；环消除后规范层依赖为 normalize→merge 单向，940 测试全绿（test_merge_extra 的 TestResolveExtends 改从 normalize 导入） |
| <a id="d15"></a>D15 | 单一 console script `podman-compose = "podman_compose:main"`（FR-3） | 双 console script 同指 `xuan_compose.cli.main:main`：规范名 `xuan-compose`（FR-3 对等）+ 日常短别名 `xc`（2026-10-10 定名，对齐 monorepo 内 `xs`/`okf` 短命令风格） | FR-3 结构对等保持（存在指向 main 的 console script）；prog 随 `argv[0]` 呈现属 D12 已允许差异，对拍快照不采集故不受影响；`tests/test_cli.py::TestConsoleScripts` 静态解析 pyproject 断言双名同指且 callable 可导入；与外部 [xcfile.dev](https://xcfile.dev/) 的 `xc` 同名风险已在 README 声明，`systemd create-unit` 建议用规范名创建（D4 路径内嵌） |

## 5.3 有意功能扩展（上游无，Xuanspace 增量）

> 与 5.2 的区别：D16 **不是语义等价重构**，而是上游不具备的新增能力。
> CLI 对拍以"声明式白名单"容纳：`tests/test_cli_parity.py::XUAN_GLOBAL_EXTENSIONS`
> 登记扩展 option，对拍前从 current 剔除（上游参数表仍逐位置零差异），
> 并有存在性 + help 文案守卫防白名单与代码漂移。

| # | 上游 | 新库 | 证据/影响 |
|---|---|---|---|
| <a id="d16"></a>D16 | 无服务端就绪概念：Windows/macOS 客户端-服务端架构下默认 podman machine（WSL2 VM）停止时，首个业务命令以 exit 125 + `Cannot connect to Podman` 失败，`runner.output` 直接抛裸 `CalledProcessError`，用户得到完整 Python traceback；启动期仅有的 `podman --version` 探活是纯本地调用，machine 停止时照样通过 | CLI 装配期**引擎就绪保障** `Podman.ensure_ready`（`runner.py`）：`podman info` 真实探活 → 不可达时 `podman machine list --format json` 仅选取 `Default=true` 机 → Stopped 则 `machine start` 并在总预算内 1s 轮询就绪；新增全局 `--no-machine-start`（env `XC_NO_MACHINE_START=1`）与就绪超时 `XC_MACHINE_READY_TIMEOUT`（默认 120s，非法值回退）；dry-run、`version`、`systemd create-unit` 豁免；无默认机/已运行仍不可达/启动失败/超时均抛 `PodmanEngineNotReadyError`，经 `main()` 以 `Error: ...` + exit 1 干净收口。**唯一子进程出口 `Podman.output` 同步分类**：业务调用以 125+连接文案失败时（`--no-machine-start` 跳过自愈、或探活后 VM 才停止的 TOCTOU 窗口）同样收口为 `PodmanEngineNotReadyError`，其余 125（容器运行时失败）保持 `CalledProcessError` 对等 | `tests/test_engine_ready.py` 23 例（连接错误双条件分类、探活、machine 枚举、已就绪零副作用、自动启动序列、Starting 等待、无默认机/非默认机/运行中不可达/启动失败/超时/JSON 损坏降级、二进制一致、output 出口连接错误收口 3 例）；`tests/test_cli.py` 装配契约 6 例（调用先于 compose 加载、三类豁免、env 开关、超时解析、无 traceback 收口）；对拍白名单守卫 2 例。真实端到端（2026-10-12，podman 5.8.1/WSL2）：停机状态 `xc up -d` 自动启动 machine 后正常拉起 notebook 栈（exit 0）；停机状态 `xc --no-machine-start up -d` 单行 `Error:` + exit 1 无 traceback。上游对等路径零行为变化：machine 已运行时仅多一次本地 `podman info` 往返（实测毫秒级），能力探测不按平台硬编码，Linux 原生 socket（无 machine）不误动作 |

## 5.4 上游缺陷修复（有意行为偏离，附回馈候选）

> 与 5.3 的区别：此处不新增能力，而是修复上游 pinned 版本自身的缺陷。
> 每条在 [findings-ledger.md](../findings-ledger.md) B 节有对应的 U 编号回馈候选，
> 修复均带 L0 回归测试；未修复的上游缺陷仍登记在台账 B 节、代码保持逐行对等。

| # | 上游 | 新库 | 证据/影响 |
|---|---|---|---|
| <a id="d17"></a>D17 | `podman-compose config --services` 必然崩溃：config 子命令的 `--services` 是 store_true **布尔**（语义"逐行打印服务名"，上游 L5491-L5493），与 build/pull/push/pause/kill/stats/logs 的位置参数 `services`（nargs="*" 列表）共用 dest；`_parse_compose_file`（上游 L2816-L2820）不校验类型直接取为 target，`_resolve_profiles`（上游 L3128）`for service in True` 抛 `TypeError: 'bool' object is not iterable`（回馈候选 [findings-ledger U14](../findings-ledger.md#ledger-u14)，P2） | target 计算仅承认 `list` 形态的服务名序列（位置参数 `nargs="*"` 的恒等产物），布尔/None 一律视为"未点名"（[engine.py:383-392](../../../src/xuan_compose/engine.py#L383-L392)）；argparse 参数表零变化（无快照偏离），仅修正 dest 值到 target 的类型边界 | `tests/test_engine_parse_paths.py::TestConfigServicesFlag` 2 例 L0 回归（无 profile 服务全量保留、带 profile 不被布尔误过滤），修复前两例均复现同一 TypeError；1011 测试全绿。对其余子命令零影响：位置参数 nargs="*" 产物恒为 list |
