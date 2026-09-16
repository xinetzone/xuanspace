# xuan-compose 对等审计：上游符号映射表与差异登记

> 本文档是 xuan-compose 对 [containers/podman-compose](https://github.com/containers/podman-compose)
> v1.6.0 分层重构的**对等性证据**（对应 spec AC-3/AC-10、TR-10.1/10.3）。

- 上游基线：`podman_compose.py` 单文件，pinned commit
  `e3df10472e194ab6d547b5ad25542c5c79e1a5fb`
- 统计口径：AST 解析上游单文件的**顶层公开符号**（不以 `_` 开头的
  `def`/`async def`/`class`，以及全大写/`__dunder__` 模块级常量；
  `@overload` 重复定义按 distinct 名称计 1 个）
- 复现命令（子项目根）：`python scripts/symbol_inventory.py`

## 1. 总账（TR-10.1）

| 类别 | 数量 |
|---|---:|
| 上游公开函数/类（distinct） | 138 |
| 上游公开常量 | 4 |
| **上游公开符号总数** | **142** |
| 同名迁入新库 | 139 |
| 重命名迁入 | 1（`PodmanCompose` → `ComposeEngine`） |
| 删除（装饰器副作用，显式注册表替代） | 2（`cmd_run`、`cmd_parse`） |
| **映射条目合计** | **142（100%）** |

另有 2 项 AC-3 点名的结构性移除（不计入公开符号总数，单列第 3 节）：
模块级 `script = os.path.realpath(sys.argv[0])` 导入期 argv 读取、
模块级全局单例 `podman_compose = PodmanCompose()`。

## 2. 逐模块符号映射表

行号为上游 `podman_compose.py` 行号；"新库落点"为
`src/xuan_compose/` 下的模块。未标注"重命名"者名称与上游完全一致。

### 基础层

#### `compat.py`（10）

| 上游符号 | 行号 |
|---|---:|
| is_list | 61 |
| is_relative_ref | 69 |
| filteri | 79 |
| try_int | 84 |
| try_float | 99 |
| str_to_seconds | 135 |
| ver_as_list | 152 |
| strverscmp_lt | 156 |
| try_parse_bool | 565 |
| STOP_GRACE_PERIOD（常量） | 132 |

#### `errors.py`（1）、`types.py`（1）、`envfile.py`（1）、`interpolation.py`（1）

| 符号 | 落点 | 行号 |
|---|---|---:|
| PodmanComposeError | errors.py | 3276 |
| DependField（类） | types.py | 3790 |
| dotenv_to_dict | envfile.py | 2368 |
| var_interpolate | interpolation.py | 275 |

### 规范层

#### `normalize.py`（9）、`merge.py`（7）、`discovery.py`（2）

| 符号 | 落点 | 行号 |
|---|---|---:|
| rec_subs | normalize.py | 466 |
| norm_as_list | normalize.py | 500 |
| norm_as_dict | normalize.py | 517 |
| norm_ulimit | normalize.py | 538 |
| normalize_service | normalize.py | 2078 |
| normalize | normalize.py | 2186 |
| normalize_service_final | normalize.py | 2201 |
| normalize_final | normalize.py | 2214 |
| is_context_git_url | normalize.py | 3549 |
| OverrideTag（类） | merge.py | 1764 |
| ResetTag（类） | merge.py | 1793 |
| clone | merge.py | 2221 |
| rec_merge_one | merge.py | 2225 |
| rec_merge | merge.py | 2311 |
| load_yaml_or_die | merge.py | 2320 |
| resolve_extends | **normalize.py**（T11 由 merge 下沉，见 D14） | 2329 |
| find_compose_files_recursively | discovery.py | 2392 |
| COMPOSE_DEFAULT_LS（常量） | discovery.py | 2374 |

### 执行层

#### `model.py`（3）、`runner.py`（3）、`dependencies.py`（4）、`pull.py`（4）、`logs.py`（1）

| 符号 | 落点 | 行号 |
|---|---|---:|
| ServiceDependencyCondition（类） | model.py | 1625 |
| ServiceDependency（类） | model.py | 1661 |
| PullImageSettings（类） | model.py | 3992 |
| wait_with_timeout | runner.py | 1811 |
| ExistingContainer（类） | runner.py | 1834 |
| Podman（类） | runner.py | 1845 |
| rec_deps | dependencies.py | 1685 |
| calc_dependents | dependencies.py | 1709 |
| flat_deps | dependencies.py | 1719 |
| check_dep_conditions | dependencies.py（commands/updown.py 再导出） | 3863 |
| settings_to_pull_args | pull.py | 4021 |
| pull_image | pull.py | 4030 |
| pull_images | pull.py | 4039 |
| prepare_images | pull.py | 4076 |
| create_format_logs_task | logs.py | 3959 |

### 翻译层 `translate/`

| 符号 | 落点 | 行号 |
|---|---|---:|
| parse_short_mount | mounts.py | 162 |
| fix_mount_dict | mounts.py | 224 |
| assert_volume | mounts.py | 582 |
| mount_desc_to_mount_args | mounts.py | 643 |
| mount_desc_to_volume_args | mounts.py | 719 |
| get_mnt_dict | mounts.py | 759 |
| get_mount_args | mounts.py | 769 |
| get_secret_args | secrets.py | 838 |
| ulimit_to_ulimit_args | resources.py | 692 |
| container_to_ulimit_args | resources.py | 708 |
| container_to_ulimit_build_args | resources.py | 712 |
| container_to_res_args | resources.py | 957 |
| container_to_gpu_res_args | resources.py | 962 |
| container_to_cpu_res_args | resources.py | 1013 |
| default_network_name_for_project | networks.py | 551 |
| get_network_create_args | networks.py | 1110 |
| assert_cnt_nets | networks.py | 1166 |
| get_net_args_from_network_mode | networks.py | 1199 |
| get_net_args | networks.py | 1247 |
| get_net_args_from_networks | networks.py | 1255 |
| port_dict_to_str | ports.py | 1073 |
| norm_ports | ports.py | 1090 |
| container_to_args | container_args.py | 1344 |
| adjust_build_ssh_key_paths | build.py | 3570 |
| container_to_build_args | build.py | 3580 |
| is_local | run_args.py | 3391 |
| get_excluded | run_args.py | 3795 |
| deps_from_container | run_args.py | 3945 |
| get_service_info | run_args.py | 3951 |
| get_volume_names | run_args.py | 4428 |
| compose_run_update_container_from_args | run_args.py | 4594 |
| compose_cp_args | run_args.py | 4653 |
| compose_exec_args | run_args.py | 4684 |

### 引擎层 `engine.py`（1，重命名）

| 上游符号 | 行号 | 新库符号 | 等价理由 |
|---|---:|---|---|
| PodmanCompose（类） | 2419 | **ComposeEngine** | 逐行翻译其状态字段与方法；仅改名以体现"引擎而非全局 Compose 门面"语义，构造器改为 `podman: Podman \| None = None` 显式注入（见差异 D6） |

### 命令层 `commands/`（24 命令 + 3 辅助）

| 符号 | 落点 | 行号 |
|---|---|---:|
| list_running_projects（ls） | inspect.py | 3325 |
| compose_version | version.py | 3379 |
| compose_wait | lifecycle.py | 3401 |
| compose_systemd | systemd.py | 3412 |
| compose_pull | pullpush.py | 3517 |
| compose_push | pullpush.py | 3535 |
| build_one | build.py | 3688 |
| compose_build | build.py | 3712 |
| pod_exists | updown.py | 3767 |
| create_pods | updown.py | 3772 |
| create_secrets_from_environment | updown.py | 802 |
| run_container | updown.py | 3923 |
| wait_for_container_running_healthy | updown.py | 4109 |
| compose_up | updown.py | 4158 |
| compose_down | updown.py | 4445 |
| compose_ps | inspect.py | 4530 |
| compose_run | runexec.py | 4549 |
| compose_cp | runexec.py | 4637 |
| compose_exec | runexec.py | 4674 |
| transfer_service_status | lifecycle.py | 4708 |
| compose_start | lifecycle.py | 4741 |
| compose_stop | lifecycle.py | 4749 |
| compose_restart | lifecycle.py | 4754 |
| compose_logs | logs.py | 4759 |
| compose_config | inspect.py | 4796 |
| compose_port | inspect.py | 4807 |
| compose_pause | lifecycle.py | 4818 |
| compose_unpause | lifecycle.py | 4829 |
| compose_kill | lifecycle.py | 4840 |
| compose_stats | inspect.py | 4876 |
| compose_images | inspect.py | 4903 |

命令注册表：上游 `@cmd_run` 装饰器在单例构造期副作用注册 24 命令；
新库为 `commands/__init__.py` 的 `COMMAND_HANDLERS: dict[str, Handler]`
显式注册表 + `install_handlers(engine)`（见 D1）。

### CLI 层 `cli/`（25）

| 符号 | 落点 | 行号 |
|---|---|---:|
| compose_version_parse | parser.py | 4944 |
| compose_up_parse | parser.py | 4960 |
| compose_down_parse | parser.py | 5066 |
| compose_run_parse | parser.py | 5092 |
| compose_exec_parse | parser.py | 5171 |
| compose_parse_cp（上游原名如此，含命名笔误） | parser.py | 5222 |
| compose_parse_timeout | parser.py | 5251 |
| compose_logs_parse | parser.py | 5262 |
| compose_systemd_parse | parser.py | 5302 |
| compose_pull_parse | parser.py | 5313 |
| compose_push_parse | parser.py | 5324 |
| compose_ps_parse | parser.py | 5334 |
| PullPolicyAction（类） | parser.py | 5338 |
| compose_build_up_parse | parser.py | 5356 |
| compose_build_parse | parser.py | 5388 |
| compose_up_start_parse | parser.py | 5399 |
| compose_config_parse | parser.py | 5414 |
| compose_port_parse | parser.py | 5430 |
| compose_pause_unpause_parse | parser.py | 5454 |
| compose_kill_parse | parser.py | 5461 |
| compose_images_parse | parser.py | 5480 |
| compose_stats_parse | parser.py | 5485 |
| compose_format_parse | parser.py | 5508 |
| compose_ls_parse | parser.py | 5518 |
| async_main | main.py | 5528 |
| main | main.py | 5532 |
| PODMAN_CMDS（常量） | parser.py | 119 |

### 包根（1）

| 符号 | 落点 | 行号 | 备注 |
|---|---|---:|---|
| `__version__`（常量） | `__init__.py` | 54 | 值由上游裸 `"1.6.0"` 改为 `"1.6.0+xuan.1"`（PEP 440 本地基线+派生标识，D7） |

### 删除项（2，装饰器副作用，显式注册表替代）

| 上游符号 | 行号 | 处置 |
|---|---:|---|
| cmd_run（装饰器类） | 3280 | **删除**：对模块级单例的副作用注册 → `COMMAND_HANDLERS` + `install_handlers()`（D1） |
| cmd_parse（装饰器类） | 3305 | **删除**：对 handler `._parse_args` 列表的副作用 append → `COMMAND_PARSERS` + `build_parser()`（D2） |

## 3. 结构性移除（AC-3 点名，非公开符号）

| 上游构造 | 位置 | 新库处置 |
|---|---|---|
| `script = os.path.realpath(sys.argv[0])` | 上游 L56 模块级，**导入期即读 argv** | 删除；systemd unit 模板所需可执行路径改为 `ComposeEngine.executable: str \| None` 字段，由 `cli/main.py` 装配期单点注入（TR-8.3 证明全库 argv 真实读取仅该一处，D4） |
| `podman_compose = PodmanCompose()` 模块级全局单例 | 上游 L3270 | 删除；`ComposeEngine()` 可独立实例化、可注入 runner，全部装配在 `async_main(argv)` 显式完成（TR-6.4，D5） |

## 4. 新库新增符号（非上游公开符号，分层内部产物）

- `runner.CalledProcessError`：再导出 `subprocess.CalledProcessError` 作为子进程异常统一入口（TR-4.2：全包仅 runner import subprocess）。
- `model.XPodmanSettingKey`：上游为 `PodmanCompose` 嵌套 `Enum`（6 键），提层为模块级 `StrEnum`（D8）。
- `logging_utils.configure_logging`：替代上游 `_parse_args` 内联的 `logging.basicConfig`。
- 注册表/门面：`COMMAND_HANDLERS`/`install_handlers`/`Handler`（commands）、
  `COMMAND_PARSERS`/`COMMAND_HELP`/`SYSTEMD_DESC`/`build_parser`/`parse_args`/`PullPolicyAction` 之外的
  `init_global_parser`（cli）。
- `cli/__main__.py` 的 `main` 再导出；console script `xuan-compose`。

## 5. Python 3.14 兼容修复与机械现代化差异登记（TR-10.3）

原则：只做 3.14 必需兼容与无语义机械现代化；算法/退出码/文案逐行保留，
差异按 Task 登记，每条可回溯上游行号。

### 5.1 语法/类型现代化（无语义变化）

| # | 内容 | 影响范围 | 上游位置 | 等价理由 |
|---|---|---|---|---|
| M1 | 全包移除 `from __future__ import annotations` | 8 个基础层文件（后续全包） | 各文件头 | PEP 649 起 3.14 注解默认惰性，future 导入冗余（ruff UP） |
| M2 | `except asyncio.TimeoutError as exc: raise TimeoutError from exc` → `except TimeoutError: raise` | runner.wait_with_timeout | 1811-1820 | 3.11+ 两者即同一内置类型（ruff UP041） |
| M3 | `asyncio.exceptions.IncompleteReadError/LimitOverrunError` 收敛为 `asyncio.*` 限定名 | runner._readchunk | 1020-1030 | 纯导入路径机械收敛 |
| M4 | `XPodmanSettingKey` 由 `PodmanCompose` 嵌套 `Enum` 提为 model.py 模块级 `StrEnum` | model.py | 2430-2480 | 嵌套枚举无法脱离单例引用；StrEnum 为 3.11+ 惯用法，键值字符串不变 |
| M5 | 上游 `map(...)`/zip 惰性对 → 列表推导（如 `missing = [fn for fn in files if ...]`） | engine._parse_compose_file | 2903-2917 | 等价收敛，行为一致 |
| M6 | 翻译层 `compose`/`cnt` 句柄统一 `Any` 鸭子类型（不反向 import engine） | translate/* | 全层 | 打破执行/翻译层反向依赖；属性访问契约与上游运行时完全一致 |
| M7 | 公共 API 全面类型注解：`ParserFn = Callable[[ArgumentParser], None]`、`Handler = Callable[[Any, Namespace], Awaitable[Any]]`、`cleanup_callbacks: list[Callable[[], object]]`、测试 `Union → \| None` 等 | cli/commands/translate/tests | 全局 | 3.14 完整注解，签名行为不变 |

### 5.2 结构性重构（语义等价，架构必需）

| # | 上游 | 新库 | 等价理由/证据 |
|---|---|---|---|
| D1 | `@cmd_run` 装饰器在单例构造期副作用注册 24 命令（含 help/desc 闭包计算） | `COMMAND_HANDLERS` 24 名显式 dict + `install_handlers(engine)`；help/desc 文本由 `COMMAND_HELP` 表 + systemd 运行时从 handler docstring 复刻装饰器切分算法（`re.sub(r"^\s+","",doc)` 无 MULTILINE） | T9 argparse 快照对拍 help/description 零差异；TR-7.1 24 名集合相等 |
| D2 | `@cmd_parse` 装饰器向 `handler._parse_args` append parser | `COMMAND_PARSERS: dict[str, tuple[ParserFn, ...]]` 24 键显式注册表（wait=() 空元组）+ `build_parser()` 纯构造，共享 parser 挂载顺序精确复刻 append 序 | T9 快照 11 维参数表零差异（`tests/snapshots/cli_parity.json`，上游 e3df104） |
| D3 | `PodmanCompose._parse_args(self, argv)` 引擎方法（写全局 logging、回写单例） | `parse_args(engine, argv)` CLI 纯函数（engine 仅作 global_args 回写宿主，`configure_logging` 替代 basicConfig） | 回填/分流/退出码逐行一致；TR-9.1 参数表零差异 |
| D4 | 模块级 `script = realpath(argv[0])` 导入期副作用（L56） | `ComposeEngine.executable` 字段默认 None，`cli/main.py` 装配期单点注入 | systemd create-unit 模板渲染结果逐字一致（test_systemd*）；TR-8.3 grep 全库 argv 仅一处 |
| D5 | 模块级 `podman_compose = PodmanCompose()` 活单例，`run()` 操作 self | 无单例；`async_main(argv)` 显式 `ComposeEngine()` → install_handlers → parse → Podman → 分发；`main(argv=None)` 同步壳 | TR-6.4 grep 无模块级活引擎；双实例隔离测试；console script 零参调用与上游等价（argv 可选仅为测试便利） |
| D6 | `PodmanCompose.__init__` 内自建子进程封装、读 argv | `ComposeEngine(podman=None)` 纯状态构造；runner 由 CLI 装配注入 | TR-6.1 实例化零子进程/零 argv；T8 main 装配测试 |
| D7 | 容器标签版本 `io.podman.compose.version=1.6.0` | `"1.6.0+xuan.1"`（即包 `__version__`） | 唯一有意标签值差异：标识派生版本；test_container_to_args 相应标签断言已按新值移植 |
| D8 | `PodmanCompose.XPodmanSettingKey` 嵌套枚举 | `model.XPodmanSettingKey` 模块级 StrEnum（6 键） | 见 M4 |
| D9 | `compose_up` 内闭包 `_task_cancelled`（无自由变量） | 提为 `logs.py` 模块级函数 | 无自由变量，提升后行为完全一致 |
| D10 | `run()`/`_parse_args()` 留在单体 | 随 CLI 层迁出；engine.commands 初始空 dict | T7 install_handlers 填充、T8 dispatch 查表 |
| D11 | `!override`/`!reset` YAMLObject 全局注册 | 原样保留（merge.py 导入即注册进 SafeLoader/SafeDumper），但包根 `__init__.py` 不导入 merge——`import xuan_compose` 仍无副作用 | T3 登记；行为与上游一致，包根无副作用由 TR-2.2 冒烟保证 |
| D12 | `prog` 随 `sys.argv[0]`（podman-compose 呈现） | argparse 默认 prog（`python -m xuan_compose`/`xuan-compose` 呈现） | AC-4 明确允许 prog 差异；快照序列化刻意不采集 prog |
| D13 | check_dep_conditions/_validate_completed_successfully 定义于 up 辅助区，T4 按 test_depends_on 归属提前迁入 dependencies.py 后，T7 在 commands/updown.py 又逐行重译了一份（重复实现） | T10 去重：删除 commands/updown.py 内的副本，改为 `from ..dependencies import check_dep_conditions` 再导出（保持符号面与测试导入路径），单一事实源 | 两份字节级一致；去重后 940 测试全绿、ruff/mypy 0 |
| D14 | `resolve_extends` 原置于 merge.py，但其函数体要调用 normalize 的 `rec_subs`/`normalize_service`，只能用函数内惰性 import 反向取 normalize，与 normalize 顶层 `from .merge import ...` 构成规范层内部惰性环 | T11 V 审查后下沉到 normalize.py（它本属规范化流程），内部直接用同模块函数、仅单向 `from .merge import rec_merge, load_yaml_or_die`；merge.py 删除该函数与惰性 import | 纯代码移动零行为变化；环消除后规范层依赖为 normalize→merge 单向，940 测试全绿（test_merge_extra 的 TestResolveExtends 改从 normalize 导入） |

### 5.3 逐行保留的上游怪癖（未顺手"修正"，T9 测试固化）

- argparse：无参数/help 退出码 **-1**（非 1）；`--parallel` 默认 `sys.maxsize`、
  读 `COMPOSE_PARALLEL_LIMIT`；`PullPolicyAction` 对 `--pull-always false`
  静默 return 不赋值；cp 子命令 parser 函数名沿用上游笔误 `compose_parse_cp`。
- 时间解析 `str_to_seconds` 正则 `[m:]`：合法写法是 `"1:30"`/`"1m30s"`，
  `"1m:30s"` 反而解析失败返回 None；只接受 int 秒（podman 限制，原文注释保留）。
- `is_list(bytes)` 为 True（bytes 可迭代且非 str/dict）。
- 路径：normalize 对 volumes 短格式/env_file/sub_dir 的 `./` 前缀**不剥离**
  （仅 build.context 分支剥离）；对整个卷规格串做 `os.path.join`（含 `:/target`）。
- 合并：service 自身的 `extends` 声明在 `resolve_extends` 后保留在合并结果；
  `norm_as_list` 对 None 值输出裸键（`B` 而非 `B=None`）；
  `--service-ports=False` 时先删 ports 再由 `--publish` 重建（原 ports 不保留）；
  OverrideTag 的自定义 from_yaml 直接取 ScalarNode.value（无隐式类型解析，标量即字符串）。
- 依赖：`rec_deps` 对自依赖/未知依赖"保留边但不递归展开"；
  A→B→A 环按起点剪枝。
- 镜像上下文 `_resolve_context_dependencies`：`urllib.parse.quote` 会百分号编码
  镜像 tag 的冒号（`b:1` → `b%3A1`）。
- handler 怪癖原样保留清单（T7 登记）：build 拓扑 as_completed 调度、up 的
  SIGINT 监督循环与 config-hash 重建判定、down 的 `--rmi local` 删除恰为
  is_local 的镜像、kill 的 all/services 重复分支、logs `max()` 空序列
  ValueError、ls `except Exception: break` 与 json `print(list)` 非标准输出、
  create_secrets 报错串第二个片段漏 f 前缀、networks invalid network_mode
  的 `log.fatal`+sys.exit(1)。

## 6. 对等证据索引

- argparse 参数表快照（25 命令，11 维，锚定 e3df104）：
  `tests/snapshots/cli_parity.json`；对拍测试 `tests/test_cli_parity.py`、
  `tests/test_command_surface.py`；重新生成 `tests/generate_parity_snapshots.py`。
- 上游 26 个单元测试文件全量同名移植（484 例 ≥ 上游 478），见 tasks.md T9 TR-9.2。
- 覆盖率：整体 91%，核心模块 interpolation/normalize/merge/engine/translate 全 ≥90%。
