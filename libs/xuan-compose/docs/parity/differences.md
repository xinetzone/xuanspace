# Python 3.14 兼容修复与结构性差异登记（TR-10.3）

原则：只做 3.14 必需兼容与无语义机械现代化；算法/退出码/文案逐行保留，
差异按 Task 登记，每条可回溯上游行号。

> 维护说明（2026-10-10）：本表为常青台账，条目本体随代码演进维护；
> 条目中的 Task/TR/F 编号、测试数与提交时点为**登记时点证据**，不随后续重测改写。

- 语法/类型现代化（无语义变化）：M1-M7
- 结构性重构（语义等价，架构必需）：D1-D14
- 符号总账与逐模块映射：[symbol-map.md](symbol-map.md)
- 逐行保留的上游怪癖：[upstream-quirks.md](upstream-quirks.md)

## 5.1 语法/类型现代化（无语义变化）

| # | 内容 | 影响范围 | 上游位置 | 等价理由 |
|---|---|---|---|---|
| <a id="m1"></a>M1 | 全包移除 `from __future__ import annotations` | 8 个基础层文件（后续全包） | 各文件头 | PEP 649 起 3.14 注解默认惰性，future 导入冗余（ruff UP） |
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
| <a id="d2"></a>D2 | `@cmd_parse` 装饰器向 `handler._parse_args` append parser | `COMMAND_PARSERS: dict[str, tuple[ParserFn, ...]]` 24 键显式注册表（wait=() 空元组）+ `build_parser()` 纯构造，共享 parser 挂载顺序精确复刻 append 序 | T9 快照 11 维参数表零差异（`tests/snapshots/cli_parity.json`，上游 e3df104） |
| <a id="d3"></a>D3 | `PodmanCompose._parse_args(self, argv)` 引擎方法（写全局 logging、回写单例） | `parse_args(engine, argv)` CLI 纯函数（engine 仅作 global_args 回写宿主，`configure_logging` 替代 basicConfig） | 回填/分流/退出码逐行一致；TR-9.1 参数表零差异 |
| <a id="d4"></a>D4 | 模块级 `script = realpath(argv[0])` 导入期副作用（L56） | `ComposeEngine.executable` 字段默认 None，`cli/main.py` 装配期单点注入 | systemd create-unit 模板渲染结果逐字一致（test_systemd*）；TR-8.3 grep 全库 argv 仅一处 |
| <a id="d5"></a>D5 | 模块级 `podman_compose = PodmanCompose()` 活单例，`run()` 操作 self | 无单例；`async_main(argv)` 显式 `ComposeEngine()` → install_handlers → parse → Podman → 分发；`main(argv=None)` 同步壳 | TR-6.4 grep 无模块级活引擎；双实例隔离测试；console script 零参调用与上游等价（argv 可选仅为测试便利） |
| <a id="d6"></a>D6 | `PodmanCompose.__init__` 内自建子进程封装、读 argv | `ComposeEngine(podman=None)` 纯状态构造；runner 由 CLI 装配注入 | TR-6.1 实例化零子进程/零 argv；T8 main 装配测试 |
| <a id="d7"></a>D7 | 容器标签版本 `io.podman.compose.version=1.6.0` | `"1.6.0+xuan.1"`（即包 `__version__`） | 唯一有意标签值差异：标识派生版本；test_container_to_args 相应标签断言已按新值移植 |
| <a id="d8"></a>D8 | `PodmanCompose.XPodmanSettingKey` 嵌套枚举 | `model.XPodmanSettingKey` 模块级 StrEnum（6 键） | 见 M4 |
| <a id="d9"></a>D9 | `compose_up` 内闭包 `_task_cancelled`（无自由变量） | 提为 `logs.py` 模块级函数；T11 F2 进一步把函数体折叠为 `bool(task.cancelled() or task.cancelling())` 并删除 `sys.version_info>=(3,11)` 死分支守卫（requires-python≥3.14，见 [review §5 F2](../review/2026-09-16-t11-adversarial-review.md#f2)） | 无自由变量，提升后行为完全一致；折叠与删守卫在 3.14 下语义等价，940 测试回归通过 |
| <a id="d10"></a>D10 | `run()`/`_parse_args()` 留在单体 | 随 CLI 层迁出；engine.commands 初始空 dict | T7 install_handlers 填充、T8 dispatch 查表 |
| <a id="d11"></a>D11 | `!override`/`!reset` YAMLObject 全局注册 | 原样保留（merge.py 导入即注册进 SafeLoader/SafeDumper），但包根 `__init__.py` 不导入 merge——`import xuan_compose` 仍无副作用 | T3 登记；行为与上游一致，包根无副作用由 TR-2.2 冒烟保证 |
| <a id="d12"></a>D12 | `prog` 随 `sys.argv[0]`（podman-compose 呈现） | argparse 默认 prog（`python -m xuan_compose`/`xuan-compose` 呈现） | AC-4 明确允许 prog 差异；快照序列化刻意不采集 prog |
| <a id="d13"></a>D13 | check_dep_conditions/_validate_completed_successfully 定义于 up 辅助区，T4 按 test_depends_on 归属提前迁入 dependencies.py 后，T7 在 commands/updown.py 又逐行重译了一份（重复实现） | T10 去重：删除 commands/updown.py 内的副本，改为 `from ..dependencies import check_dep_conditions` 再导出（保持符号面与测试导入路径），单一事实源 | 两份字节级一致；去重后 940 测试全绿、ruff/mypy 0 |
| <a id="d14"></a>D14 | `resolve_extends` 原置于 merge.py，但其函数体要调用 normalize 的 `rec_subs`/`normalize_service`，只能用函数内惰性 import 反向取 normalize，与 normalize 顶层 `from .merge import ...` 构成规范层内部惰性环 | T11 V 审查后下沉到 normalize.py（它本属规范化流程），内部直接用同模块函数、仅单向 `from .merge import rec_merge, load_yaml_or_die`；merge.py 删除该函数与惰性 import | 纯代码移动零行为变化；环消除后规范层依赖为 normalize→merge 单向，940 测试全绿（test_merge_extra 的 TestResolveExtends 改从 normalize 导入） |
