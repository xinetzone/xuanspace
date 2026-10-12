# 01 · CLI 全景：24 个命令的分组地图

## 命令行结构

```text
xc [全局选项] <命令> [命令选项] [服务...]
```

- **全局选项**写在命令名之前（如 `xc -f compose.yaml -p demo ps`），控制文件、
  项目名、podman 透传参数与演练模式，详见 [02-global-options.md](02-global-options.md)；
- **命令选项**写在命令名之后，每个命令不同；
- 多数接受**服务名位置参数**，省略时作用于整个栈。

所有选项与上游 `podman-compose` 1.6.0 对等；权威参数表是
`tests/snapshots/cli_parity.json` 快照（除 `prog` 外零差异）。

## 帮助系统

```bash
xc --help                    # 顶层：全局选项 + 24 个命令清单
xc up --help                 # 单个命令：该命令的全部选项
xc help                      # help 伪命令，等价于顶层 --help
```

本文的命令清单即来自 `xc --help` 的实测输出；逐命令选项见
[../reference/commands.md](../reference/commands.md)。

## 24 个命令分组

### 栈生命周期

| 命令 | 用途 |
|------|------|
| `xc up` | 创建并启动整个栈或指定服务（`-d` 后台；选项最丰富） |
| `xc down` | 停止并拆除整个栈（`-v` 连带卷、`--rmi` 连带镜像） |
| `xc start` | 启动已存在但已停止的服务 |
| `xc stop` | 停止服务（不删除容器） |
| `xc restart` | 重启服务 |
| `xc wait` | 阻塞等待运行中的容器停止 |

### 服务内操作

| 命令 | 用途 |
|------|------|
| `xc run` | 基于某服务的定义创建一次性容器跑命令（`--rm`、`--service-ports`） |
| `xc exec` | 在正在运行的服务容器中执行命令（多实例用 `--index`） |
| `xc cp` | 容器与本地文件系统互拷（`SERVICE:PATH` 语法） |

### 状态与观测

| 命令 | 用途 |
|------|------|
| `xc ps` | 列出栈内容器状态（`-q` 仅 ID；`-f` JSON/Go 模板） |
| `xc logs` | 查看日志（`-f` 跟踪、`--since/--until/--tail`） |
| `xc port` | 打印某服务端口绑定的宿主端口 |
| `xc stats` | CPU/内存/IO 实时统计流（`--no-stream` 只取一次） |
| `xc images` | 列出栈使用的镜像（`-q` 仅 ID） |
| `xc ls` | 列出正在运行的 compose 项目（`-f table/json`） |

### 暂停与信号

| 命令 | 用途 |
|------|------|
| `xc pause` | 暂停所有（或指定）运行中容器 |
| `xc unpause` | 恢复暂停的容器 |
| `xc kill` | 向容器发信号（`-s SIGNAL`，默认 `KILL`；`-a` 全部） |

### 镜像

| 命令 | 用途 |
|------|------|
| `xc build` | 构建栈中带 `build:` 段的镜像（`--pull`、`--build-arg`、`--no-cache`） |
| `xc pull` | 拉取栈镜像（`--force-local` 连未带前缀的镜像也拉） |
| `xc push` | 推送栈镜像 |

### 配置与系统

| 命令 | 用途 |
|------|------|
| `xc config` | 显示解析后的 compose 文件（`--services` 仅列服务、`-q` 仅解析不打印、`--no-normalize`） |
| `xc version` | 显示版本（`--short`、`-f pretty/json`） |
| `xc systemd` | 生成 systemd unit 并注册 compose 栈（`-a register/unregister/create-unit/list`） |

## 服务名参数的作用域

接受 `[services ...]` 的命令（`up/down/start/stop/restart/logs/pull/push/build/
pause/unpause/kill/stats`）遵循同一规则：

```bash
xc up -d web                 # 只启动 web（及其依赖，除非加 --no-deps）
xc restart web redis         # 只重启两个服务
xc logs --tail 50 api        # 只看 api 的最近 50 行
```

`run` 与 `exec` 的服务名是**必填位置参数**，其后可跟要在容器内执行的命令：

```bash
xc exec web sh               # 进入 web 容器的 shell
xc run --rm web python -V    # 按 web 的定义起一次性容器跑 python -V，退出即删
```

## 命令分发的内部结构（可选了解）

CLI 解析后经 `engine.commands` 注册表查表分发到 `commands/` 下的 24 个 handler，
handler 不静态依赖引擎（以 `compose: Any` 鸭子类型接收），分层与
108 条内部 import 的逐条核对见 [../dev/index.md](../dev/index.md)
与 [../dev/parity/symbol-map.md](../dev/parity/symbol-map.md)。

## 下一步

- [02-global-options.md](02-global-options.md)：全局选项逐项说明
- [../reference/commands.md](../reference/commands.md)：每个命令的选项速查
