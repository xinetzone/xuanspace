# 逐行保留的上游怪癖（未顺手「修正」，T9 测试固化）

> 新写测试时若实际行为与「合理预期」冲突，先假设测试错、回到上游代码核对；
> 确认是上游行为则按真实输出断言并登记本清单——为让断言符合直觉而改生产代码，
> 就是在制造与上游的不对等。本项目有 10+ 次此类纠正。

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
