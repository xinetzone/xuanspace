# 上游符号映射表（逐模块）

> 行号为上游 `podman_compose.py` 行号；「新库落点」为 `src/xuan_compose/`
> 下的模块。未标注「重命名」者名称与上游完全一致。
> 统计口径与复现命令见 [parity/README.md](README.md)；差异编号 D*/M* 见
> [differences.md](differences.md)。

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
| resolve_extends | **normalize.py**（T11 由 merge 下沉，见 [D14](differences.md#d14)） | 2329 |
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
| PodmanCompose（类） | 2419 | **ComposeEngine** | 逐行翻译其状态字段与方法；仅改名以体现"引擎而非全局 Compose 门面"语义，构造器改为 `podman: Podman \| None = None` 显式注入（见 [D6](differences.md#d6)） |

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
显式注册表 + `install_handlers(engine)`（见 [D1](differences.md#d1)）。

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
| `__version__`（常量） | `__init__.py` | 54 | 值由上游裸 `"1.6.0"` 改为 `"1.6.0+xuan.1"`（PEP 440 本地基线+派生标识，[D7](differences.md#d7)） |

### 删除项（2，装饰器副作用，显式注册表替代）

| 上游符号 | 行号 | 处置 |
|---|---:|---|
| cmd_run（装饰器类） | 3280 | **删除**：对模块级单例的副作用注册 → `COMMAND_HANDLERS` + `install_handlers()`（[D1](differences.md#d1)） |
| cmd_parse（装饰器类） | 3305 | **删除**：对 handler `._parse_args` 列表的副作用 append → `COMMAND_PARSERS` + `build_parser()`（[D2](differences.md#d2)） |

## 3. 结构性移除（AC-3 点名，非公开符号）

| 上游构造 | 位置 | 新库处置 |
|---|---|---|
| `script = os.path.realpath(sys.argv[0])` | 上游 L56 模块级，**导入期即读 argv** | 删除；systemd unit 模板所需可执行路径改为 `ComposeEngine.executable: str \| None` 字段，由 `cli/main.py` 装配期单点注入（TR-8.3 证明全库 argv 真实读取仅该一处，[D4](differences.md#d4)） |
| `podman_compose = PodmanCompose()` 模块级全局单例 | 上游 L3270 | 删除；`ComposeEngine()` 可独立实例化、可注入 runner，全部装配在 `async_main(argv)` 显式完成（TR-6.4，[D5](differences.md#d5)） |

## 4. 新库新增符号（非上游公开符号，分层内部产物）

- `runner.CalledProcessError`：再导出 `subprocess.CalledProcessError` 作为子进程异常统一入口（TR-4.2：全包仅 runner import subprocess）。
- `model.XPodmanSettingKey`：上游为 `PodmanCompose` 嵌套 `Enum`（6 键），提层为模块级 `StrEnum`（[D8](differences.md#d8)）。
- `logging_utils.configure_logging`：替代上游 `_parse_args` 内联的 `logging.basicConfig`。
- 注册表/门面：`COMMAND_HANDLERS`/`install_handlers`/`Handler`（commands）、
  `COMMAND_PARSERS`/`COMMAND_HELP`/`SYSTEMD_DESC`/`build_parser`/`parse_args`/`PullPolicyAction` 之外的
  `init_global_parser`（cli）。
- `cli/__main__.py` 的 `main` 再导出；console script `xuan-compose`。
