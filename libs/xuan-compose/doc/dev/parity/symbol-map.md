# 上游符号映射表（逐模块）

> 行号为上游 `podman_compose.py` 行号（锚定上游 commit `1f1cb09bfee4d79857ed23bfe2699febd7d20a53`）；「新库落点」为 `src/xuan_compose/`
> 下的模块。未标注「重命名」者名称与上游完全一致。
> 统计口径与复现命令见 [parity/index.md](index.md)；差异编号 D*/M* 见
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
| try_parse_bool | 643 |
| STOP_GRACE_PERIOD（常量） | 132 |

#### `errors.py`（1）、`types.py`（1）、`envfile.py`（1）、`interpolation.py`（1）

| 符号 | 落点 | 行号 |
|---|---|---:|
| PodmanComposeError | errors.py | 3347 |
| DependField（类） | types.py | 3863 |
| dotenv_to_dict | envfile.py | 2418 |
| var_interpolate | interpolation.py | 341 |

### 规范层

#### `normalize.py`（9）、`merge.py`（7）、`discovery.py`（2）

| 符号 | 落点 | 行号 |
|---|---|---:|
| rec_subs | normalize.py | 532 |
| norm_as_list | normalize.py | 578 |
| norm_as_dict | normalize.py | 595 |
| norm_ulimit | normalize.py | 616 |
| normalize_service | normalize.py | 2113 |
| normalize | normalize.py | 2228 |
| normalize_service_final | normalize.py | 2243 |
| normalize_final | normalize.py | 2256 |
| is_context_git_url | normalize.py | 3620 |
| OverrideTag（类） | merge.py | 267 |
| ResetTag（类） | merge.py | 310 |
| clone | merge.py | 2263 |
| rec_merge_one | merge.py | 2267 |
| rec_merge | merge.py | 2361 |
| load_yaml_or_die | merge.py | 2370 |
| resolve_extends | **normalize.py**（T11 由 merge 下沉，见 [D14](differences.md#d14)） | 2379 |
| find_compose_files_recursively | discovery.py | 2442 |
| COMPOSE_DEFAULT_LS（常量） | discovery.py | 2424 |

> 注：上游 `fd64dc2` 为 `rec_subs` 新增第 4 个 `@overload`（`OverrideTag → OverrideTag`）；
> `@overload` 重复定义按 distinct 名称计 1，故 `rec_subs` 仍计 1 个、总账不变。

### 执行层

#### `model.py`（3）、`runner.py`（3）、`dependencies.py`（4）、`pull.py`（4）、`logs.py`（1）

| 符号 | 落点 | 行号 |
|---|---|---:|
| ServiceDependencyCondition（类） | model.py | 1703 |
| ServiceDependency（类） | model.py | 1739 |
| PullImageSettings（类） | model.py | 4065 |
| wait_with_timeout | runner.py | 1837 |
| ExistingContainer（类） | runner.py | 1860 |
| Podman（类） | runner.py | 1871 |
| rec_deps | dependencies.py | 1763 |
| calc_dependents | dependencies.py | 1787 |
| flat_deps | dependencies.py | 1797 |
| check_dep_conditions | dependencies.py（commands/updown.py 再导出） | 3936 |
| settings_to_pull_args | pull.py | 4094 |
| pull_image | pull.py | 4103 |
| pull_images | pull.py | 4112 |
| prepare_images | pull.py | 4149 |
| create_format_logs_task | logs.py | 4032 |

### 翻译层 `translate/`

| 符号 | 落点 | 行号 |
|---|---|---:|
| parse_short_mount | mounts.py | 162 |
| fix_mount_dict | mounts.py | 224 |
| assert_volume | mounts.py | 660 |
| mount_desc_to_mount_args | mounts.py | 721 |
| mount_desc_to_volume_args | mounts.py | 797 |
| get_mnt_dict | mounts.py | 837 |
| get_mount_args | mounts.py | 847 |
| get_secret_args | secrets.py | 916 |
| ulimit_to_ulimit_args | resources.py | 770 |
| container_to_ulimit_args | resources.py | 786 |
| container_to_ulimit_build_args | resources.py | 790 |
| container_to_res_args | resources.py | 1035 |
| container_to_gpu_res_args | resources.py | 1040 |
| container_to_cpu_res_args | resources.py | 1091 |
| default_network_name_for_project | networks.py | 629 |
| get_network_create_args | networks.py | 1188 |
| assert_cnt_nets | networks.py | 1244 |
| get_net_args_from_network_mode | networks.py | 1277 |
| get_net_args | networks.py | 1325 |
| get_net_args_from_networks | networks.py | 1333 |
| port_dict_to_str | ports.py | 1151 |
| norm_ports | ports.py | 1168 |
| container_to_args | container_args.py | 1422 |
| adjust_build_ssh_key_paths | build.py | 3641 |
| container_to_build_args | build.py | 3651 |
| is_local | run_args.py | 3462 |
| get_excluded | run_args.py | 3868 |
| deps_from_container | run_args.py | 4018 |
| get_service_info | run_args.py | 4024 |
| get_volume_names | run_args.py | 4501 |
| compose_run_update_container_from_args | run_args.py | 4667 |
| compose_cp_args | run_args.py | 4726 |
| compose_exec_args | run_args.py | 4757 |

### 引擎层 `engine.py`（1，重命名）

| 上游符号 | 行号 | 新库符号 | 等价理由 |
|---|---:|---|---|
| PodmanCompose（类） | 2469 | **ComposeEngine** | 逐行翻译其状态字段与方法；仅改名以体现"引擎而非全局 Compose 门面"语义，构造器改为 `podman: Podman \| None = None` 显式注入（见 [D6](differences.md#d6)） |

### 命令层 `commands/`（24 命令 + 3 辅助）

| 符号 | 落点 | 行号 |
|---|---|---:|
| list_running_projects（ls） | inspect.py | 3396 |
| compose_version | version.py | 3450 |
| compose_wait | lifecycle.py | 3472 |
| compose_systemd | systemd.py | 3483 |
| compose_pull | pullpush.py | 3588 |
| compose_push | pullpush.py | 3606 |
| build_one | build.py | 3761 |
| compose_build | build.py | 3785 |
| pod_exists | updown.py | 3840 |
| create_pods | updown.py | 3845 |
| create_secrets_from_environment | updown.py | 880 |
| run_container | updown.py | 3996 |
| wait_for_container_running_healthy | updown.py | 4182 |
| compose_up | updown.py | 4231 |
| compose_down | updown.py | 4518 |
| compose_ps | inspect.py | 4603 |
| compose_run | runexec.py | 4622 |
| compose_cp | runexec.py | 4710 |
| compose_exec | runexec.py | 4747 |
| transfer_service_status | lifecycle.py | 4781 |
| compose_start | lifecycle.py | 4814 |
| compose_stop | lifecycle.py | 4822 |
| compose_restart | lifecycle.py | 4827 |
| compose_logs | logs.py | 4832 |
| compose_config | inspect.py | 4869 |
| compose_port | inspect.py | 4880 |
| compose_pause | lifecycle.py | 4891 |
| compose_unpause | lifecycle.py | 4902 |
| compose_kill | lifecycle.py | 4913 |
| compose_stats | inspect.py | 4949 |
| compose_images | inspect.py | 4976 |

命令注册表：上游 `@cmd_run` 装饰器在单例构造期副作用注册 24 命令；
新库为 `commands/__init__.py` 的 `COMMAND_HANDLERS: dict[str, Handler]`
显式注册表 + `install_handlers(engine)`（见 [D1](differences.md#d1)）。

### CLI 层 `cli/`（25）

| 符号 | 落点 | 行号 |
|---|---|---:|
| compose_version_parse | parser.py | 5017 |
| compose_up_parse | parser.py | 5033 |
| compose_down_parse | parser.py | 5139 |
| compose_run_parse | parser.py | 5165 |
| compose_exec_parse | parser.py | 5244 |
| compose_parse_cp（上游原名如此，含命名笔误） | parser.py | 5295 |
| compose_parse_timeout | parser.py | 5324 |
| compose_logs_parse | parser.py | 5335 |
| compose_systemd_parse | parser.py | 5375 |
| compose_pull_parse | parser.py | 5386 |
| compose_push_parse | parser.py | 5397 |
| compose_ps_parse | parser.py | 5407 |
| PullPolicyAction（类） | parser.py | 5411 |
| compose_build_up_parse | parser.py | 5429 |
| compose_build_parse | parser.py | 5461 |
| compose_up_start_parse | parser.py | 5472 |
| compose_config_parse | parser.py | 5487 |
| compose_port_parse | parser.py | 5503 |
| compose_pause_unpause_parse | parser.py | 5527 |
| compose_kill_parse | parser.py | 5534 |
| compose_images_parse | parser.py | 5553 |
| compose_stats_parse | parser.py | 5558 |
| compose_format_parse | parser.py | 5581 |
| compose_ls_parse | parser.py | 5591 |
| async_main | main.py | 5601 |
| main | main.py | 5605 |
| PODMAN_CMDS（常量） | parser.py | 119 |

### 包根（1）

| 符号 | 落点 | 行号 | 备注 |
|---|---|---:|---|
| `__version__`（常量） | `__init__.py` | 54 | 值由上游裸 `"1.6.0"` 改为 `"1.6.0+xuan.1"`（PEP 440 本地基线+派生标识，[D7](differences.md#d7)） |

### 删除项（2，装饰器副作用，显式注册表替代）

| 上游符号 | 行号 | 处置 |
|---|---:|---|
| cmd_run（装饰器类） | 3351 | **删除**：对模块级单例的副作用注册 → `COMMAND_HANDLERS` + `install_handlers()`（[D1](differences.md#d1)） |
| cmd_parse（装饰器类） | 3376 | **删除**：对 handler `._parse_args` 列表的副作用 append → `COMMAND_PARSERS` + `build_parser()`（[D2](differences.md#d2)） |

## 3. 结构性移除（AC-3 点名，非公开符号）

| 上游构造 | 位置 | 新库处置 |
|---|---|---|
| `script = os.path.realpath(sys.argv[0])` | 上游 L56 模块级，**导入期即读 argv** | 删除；systemd unit 模板所需可执行路径改为 `ComposeEngine.executable: str \| None` 字段，由 `cli/main.py` 装配期单点注入（TR-8.3 证明全库 argv 真实读取仅该一处，[D4](differences.md#d4)） |
| `podman_compose = PodmanCompose()` 模块级全局单例 | 上游 L3341 | 删除；`ComposeEngine()` 可独立实例化、可注入 runner，全部装配在 `async_main(argv)` 显式完成（TR-6.4，[D5](differences.md#d5)） |

## 4. 新库新增符号（非上游公开符号，分层内部产物）

- `runner.CalledProcessError`：再导出 `subprocess.CalledProcessError` 作为子进程异常统一入口（TR-4.2：全包仅 runner import subprocess）。
- `model.XPodmanSettingKey`：上游为 `PodmanCompose` 嵌套 `Enum`（6 键），提层为模块级 `StrEnum`（[D8](differences.md#d8)）。
- `logging_utils.configure_logging`：替代上游 `_parse_args` 内联的 `logging.basicConfig`。
- 注册表/门面：`COMMAND_HANDLERS`/`install_handlers`/`Handler`（commands）、
  `COMMAND_PARSERS`/`COMMAND_HELP`/`SYSTEMD_DESC`/`build_parser`/`parse_args`/`PullPolicyAction` 之外的
  `init_global_parser`（cli）。
- `cli/__main__.py` 的 `main` 再导出；console script `xuan-compose`。
