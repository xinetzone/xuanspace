# 全局选项参考

适用于 `xc [全局选项] <命令> ...`，位于命令名之前。以下全表来自
`xc --help`（版本 `1.6.0+xuan.1`）。意图讲解见
[../guide/02-global-options.md](../guide/02-global-options.md)。

## 全表

| 选项 | 取值 | 说明 |
|------|------|------|
| `-h`, `--help` | — | 显示帮助并退出 |
| `-v`, `--version` | — | 显示版本并退出 |
| `--in-pod` | `true`/`false`/`<name>` | Pod 用法：`true` 使用 `pod_<项目名>`，`false` 不用 Pod，否则为自定义 Pod 名 |
| `--pod-args` | `args` | 透传给 `podman pod` 的自定义参数 |
| `--env-file` | `path` | 备用环境文件（可多次指定） |
| `-f`, `--file` | `file` | 指定 compose 文件（缺省时按候选名自动发现，见 [../guide/04-compose-processing.md](../guide/04-compose-processing.md)）；`-` 表示从 stdin 读；可多次 |
| `--profile` | `name` | 启用指定 profile（可多次） |
| `-p`, `--project-name` | `NAME` | 自定义项目名（默认目录名） |
| `--podman-path` | `PATH` | 自定义 podman 可执行路径（默认取 PATH） |
| `--podman-args` | `args` | 透传给 `podman` 的全局参数 |
| `--podman-pull-args` | `args` | 透传给 `podman pull` |
| `--podman-push-args` | `args` | 透传给 `podman push` |
| `--podman-build-args` | `args` | 透传给 `podman build` |
| `--podman-inspect-args` | `args` | 透传给 `podman inspect` |
| `--podman-run-args` | `args` | 透传给 `podman run` |
| `--podman-start-args` | `args` | 透传给 `podman start` |
| `--podman-stop-args` | `args` | 透传给 `podman stop` |
| `--podman-rm-args` | `args` | 透传给 `podman rm` |
| `--podman-volume-args` | `args` | 透传给 `podman volume` |
| `--no-ansi` | — | 不打印 ANSI 控制字符 |
| `--no-cleanup` | — | 不停止和删除既有 Pod 与容器 |
| `--dry-run` | — | 模拟执行：跳过 podman 探测、写操作只打印 argv 不执行；查询类调用仍真实发生 |
| `--parallel` | `N` | 并发 podman 调用上限（内部信号量） |
| `--no-machine-start` | — | 不自动启动停止的默认 podman machine（D16，见 [../guide/02-global-options.md](../guide/02-global-options.md#global-machine-ready)） |
| `--verbose` | — | 打印调试输出（含翻译出的 podman argv） |

## 相关环境变量

| 变量 | 作用 |
|------|------|
| `XC_NO_MACHINE_START` | `1`/`true` 等真值等同 `--no-machine-start`（D16） |
| `XC_MACHINE_READY_TIMEOUT` | 等待默认 machine 就绪的总预算秒数，默认 `120`；空值或非法正数回退默认 |</think_never_used_51bce0c785ca2f68081bfa7d91973934><seed:tool_call><function name="Edit"><parameter name="file_path" string="true">d:\AI\projects\xuanspace\libs\xuan-compose\doc\reference\global-options.md