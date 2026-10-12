# 24 个命令选项参考

用法 `xc <命令> [选项] [服务...]`。下列各表取自 `xc <命令> --help`
实测输出（版本 `1.6.0+xuan.1`，与上游 argparse 快照零差异）。
命令分组导览见 [../guide/01-cli-overview.md](../guide/01-cli-overview.md)。

## 栈生命周期

### `xc up [选项] [services ...]`

创建并启动整个栈或部分服务。

| 选项 | 说明 |
|------|------|
| `-d`, `--detach` | 后台运行；与两个 `--abort-on-container-*` 互斥 |
| `--no-color` | 单色输出 |
| `--quiet-pull` | 拉取镜像时不打印进度 |
| `--no-deps` | 不启动链接的依赖服务 |
| `--no-attach SERVICE` | 不附加到指定服务（可多次） |
| `--force-recreate` | 配置与镜像未变也重建容器 |
| `--always-recreate-deps` | 重建依赖容器；与 `--no-recreate` 互斥 |
| `--no-recreate` | 容器已存在则不重建；与 `--force-recreate`、`-V` 互斥 |
| `--no-build` | 镜像缺失也不构建 |
| `--no-start` | 只创建不启动 |
| `--build` | 启动前构建镜像 |
| `--abort-on-container-exit` | 任一容器停止即停全部；与 `-d`、`--abort-on-container-failure` 互斥 |
| `--abort-on-container-failure` | 任一容器非零退出即停全部；与 `-d`、`--abort-on-container-exit` 互斥 |
| `-t`, `--timeout TIMEOUT` | 附着/运行中容器的关停超时秒数（默认 10） |
| `-V`, `--renew-anon-volumes` | 重建匿名卷而非复用 |
| `--remove-orphans` | 删除 Compose 文件未定义服务的容器 |
| `--scale SERVICE=NUM` | 缩放实例数（覆盖文件中的 scale） |
| `--exit-code-from SERVICE` | 返回指定服务容器的退出码；隐含 `--abort-on-container-exit` |
| `--no-hosts` | 不修改容器内 `/etc/hosts` |
| `--pull [{always,missing,never,newer}]` | 拉取策略；裸 `--pull` 视为 `newer`；默认取文件 pull_policy |
| `--pull-always [{true,false}]` | 已废弃，改用 `--pull=always` |
| `--build-arg key=val` | 构建期变量（可多次） |
| `--no-cache` | 构建不使用缓存 |
| `--wait` | 等待服务 running/healthy（隐含 detach） |
| `--wait-timeout WAIT_TIMEOUT` | 等待健康的最长秒数 |

### `xc down [选项] [services ...]`

拆除整个栈。

| 选项 | 说明 |
|------|------|
| `-v`, `--volumes` | 删除文件中声明的命名卷与容器上的匿名卷 |
| `--remove-orphans` | 删除未定义服务的容器 |
| `--rmi [{local,all}]` | 连带删除镜像：`local` 仅无自定义标签的，`all` 全部 |
| `-t`, `--timeout TIMEOUT` | 关停超时秒数 |

### `xc start [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `--wait` | 等待服务 running/healthy（隐含 detach） |
| `--wait-timeout WAIT_TIMEOUT` | 等待健康的最长秒数 |

### `xc stop [选项] [services ...]` / `xc restart [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `-t`, `--timeout TIMEOUT` | 关停超时秒数（stop/restart 同义） |

### `xc wait`

无选项。阻塞等待运行中的容器停止。

## 服务内操作

### `xc run [选项] service [command ...]`

按服务定义创建一次性容器。

| 选项 | 说明 |
|------|------|
| `--build` | 启动前构建镜像 |
| `-d`, `--detach` | 后台运行并打印新容器名 |
| `--name NAME` | 指定容器名 |
| `--entrypoint ENTRYPOINT` | 覆盖镜像入口点 |
| `-e`, `--env KEY=VAL` | 设置环境变量（可多次） |
| `-l`, `--label KEY=VAL` | 增加/覆盖标签（可多次） |
| `-u`, `--user USER` | 指定用户名或 uid 运行 |
| `--no-deps` | 不启动依赖服务 |
| `--rm` | 运行后删除容器（detach 模式下被忽略） |
| `-p`, `--publish PUBLISH` | 发布端口（可多次） |
| `--service-ports` | 启用并映射该服务自身的端口 |
| `-v`, `--volume VOLUME` | 绑定挂载卷（可多次） |
| `-T` | 禁用 TTY 分配 |
| `-w`, `--workdir WORKDIR` | 容器内工作目录 |

### `xc exec [选项] service [command ...]`

在运行中的服务容器里执行命令。

| 选项 | 说明 |
|------|------|
| `-d`, `--detach` | 后台执行 |
| `--privileged` | 赋予扩展 Linux capabilities |
| `-u`, `--user USER` | 指定用户名或 uid |
| `-T` | 禁用 TTY 分配 |
| `--index INDEX` | 多实例时指定容器序号 |
| `-e`, `--env KEY=VAL` | 环境变量（可多次） |
| `-w`, `--workdir WORKDIR` | 容器内工作目录 |

### `xc cp [选项] SOURCE DESTINATION`

容器与本地互拷；路径用 `SERVICE:PATH` 表示容器内，裸 `PATH` 表示本地。

| 选项 | 说明 |
|------|------|
| `-a`, `--archive ARCHIVE` | 按目标容器主 uid/gid chown（默认 True） |
| `--overwrite OVERWRITE` | 允许目录与非目录互相覆盖（默认 None） |

## 状态与观测

### `xc ps [选项]`

| 选项 | 说明 |
|------|------|
| `-q`, `--quiet` | 仅显示容器 ID |
| `-f`, `--format FORMAT` | JSON 或 Go 模板格式化输出 |

### `xc logs [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `-f`, `--follow` | 跟踪输出（默认 false） |
| `-l`, `--latest` | 作用于 podman 已知的最新容器 |
| `-n`, `--names` | 日志中输出容器名 |
| `--no-color` | 单色输出 |
| `--no-log-prefix` | 不打印容器标识前缀 |
| `--since SINCE` | 仅显示该时间点之后的日志 |
| `-t`, `--timestamps` | 显示时间戳 |
| `--tail TAIL` | 每容器显示末尾行数 |
| `--until UNTIL` | 显示到该时间点为止的日志 |

### `xc port [选项] service private_port`

| 选项 | 说明 |
|------|------|
| `--index INDEX` | 多实例时的容器序号 |
| `--protocol {tcp,udp}` | 协议（默认按端口绑定） |

### `xc stats [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `-i`, `--interval INTERVAL` | 报告间隔秒数（默认 5） |
| `--no-reset` | 间隔之间不清屏 |
| `--no-stream` | 只取一次结果 |
| `-f`, `--format FORMAT` | JSON 或 Go 模板格式化 |

### `xc images [选项]`

| 选项 | 说明 |
|------|------|
| `-q`, `--quiet` | 仅显示镜像 ID |

### `xc ls [选项]`

列出正在运行的 compose 项目。

| 选项 | 说明 |
|------|------|
| `-f`, `--format {table,json}` | 输出格式 |

## 暂停与信号

### `xc pause [services ...]` / `xc unpause [services ...]`

无专属选项；省略服务名时作用于全部相关容器。

### `xc kill [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `-s`, `--signal SIGNAL` | 发送的信号（默认 `KILL`） |
| `-a`, `--all` | 向所有运行中容器发信号 |

## 镜像

### `xc build [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `--pull [{always,missing,never,newer}]` | 构建拉取策略；裸 `--pull` 视为 `newer` |
| `--pull-always [{true,false}]` | 已废弃，改用 `--pull=always` |
| `--build-arg key=val` | 构建期变量（可多次） |
| `--no-cache` | 不使用构建缓存 |

### `xc pull [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `--force-local` | 对含 build 段的服务也拉取未带前缀的镜像 |

### `xc push [选项] [services ...]`

| 选项 | 说明 |
|------|------|
| `--ignore-push-failures` | 尽量推送并忽略失败（上游标注 not implemented） |

## 配置与系统

### `xc config [选项]`

| 选项 | 说明 |
|------|------|
| `--no-normalize` | 不归一化 compose 模型 |
| `--services` | 逐行打印服务名 |
| `-q`, `--quiet` | 只解析不打印 |

### `xc version [选项]`

| 选项 | 说明 |
|------|------|
| `-f`, `--format {pretty,json}` | 输出格式 |
| `--short` | 仅显示版本号 |

### `xc systemd -a {register,unregister,create-unit,list,ls}`

生成 systemd unit 并注册 compose 栈。首次安装先执行
`sudo xc systemd -a create-unit`，之后在栈目录 `xc systemd -a register`，
即可用 `systemctl --user start podman-compose@<PROJ>` 控制。

### `xc help`

伪命令，显示顶层帮助。
