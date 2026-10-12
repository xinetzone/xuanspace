# 02 · 全局选项：文件、项目、podman 透传与演练

全局选项位于命令名之前，对本次调用的所有子命令生效。以下分组讲解使用意图；
无叙述的全表见 [../reference/global-options.md](../reference/global-options.md)。

## 指定 compose 文件：`-f/--file`

```bash
xc -f compose.yaml ps
xc -f deploy/compose.override.yml config
xc -f compose.yaml -f override.yaml up -d     # 多文件：后者合并/覆盖前者
xc -f - config                                 # '-'：从标准输入读 compose 文档
```

- 未指定 `-f` 时并**不是**固定取 `docker-compose.yml`：实际是从当前目录
  **递归向上**按候选名（`compose.yaml` 最优先）自动发现，也可用
  `COMPOSE_FILE` 环境变量（或 `.env` 中的同名变量）指定，完整规则见
  {ref}`compose 处理管线·文件发现 <discovery>`（`xc --help`
  中显示的 `default: docker-compose.yml` 是 argparse 展示元数据，与自动发现
  逻辑并存）；
- `-f` 可多次出现，按 Compose 规范的合并规则叠加，管线细节见
  [04-compose-processing.md](04-compose-processing.md)。

## 项目名：`-p/--project-name`

项目名是容器、网络、Pod、卷等资源的**隔离与标签命名空间**，默认为当前目录名。

```bash
xc -p demo up -d            # 资源统一归入 demo 项目（目录名不再生效）
xc -p demo ps               # 同项目名才能查到上面的容器
xc -p demo down             # 精确拆除该项目
```

多份栈并存于同一台机器时，用不同 `-p` 隔离；省略项目名会以目录名隐式分组，
移动目录后可能「找不到」原项目的容器。

## 启用 profile：`--profile`

只激活声明了对应 `profiles` 的服务：

```bash
xc --profile debug up -d
xc --profile edge --profile canary up -d     # 可重复，启用多个
```

## 环境文件：`--env-file`

指定插值用的环境文件（可多次出现），默认读取项目目录的 `.env`：

```bash
xc --env-file .env --env-file .env.local config
```

## Pod 模式：`--in-pod` 与 `--pod-args`

```bash
xc --in-pod true up -d      # 创建/使用名为 pod_<项目名> 的 Pod（rootful 下有意义）
xc --in-pod false up -d     # 不使用 Pod
xc --in-pod mypod up -d     # 使用自定义名称的 Pod
xc --pod-args="--infra=false" up -d          # 透传给 `podman pod create`
```

rootless 单容器场景通常保持默认即可；Pod 相关的上游怪癖见
[../dev/parity/upstream-quirks.md](../dev/parity/upstream-quirks.md)。

## podman 定位与参数透传

```bash
xc --podman-path /usr/bin/podman ps          # 指定可执行文件位置
xc --podman-args="--root /var/lib/containers" ps
```

各类透传选项把自定义参数注入对应的 podman 子命令，排障与非标环境最常用：

| 选项 | 注入目标 |
|------|---------|
| `--podman-args` | `podman` 全局参数 |
| `--podman-pull-args` / `--podman-push-args` / `--podman-build-args` | 拉/推/构建 |
| `--podman-inspect-args` / `--podman-run-args` | inspect / run |
| `--podman-start-args` / `--podman-stop-args` / `--podman-rm-args` | 启动/停止/删除 |
| `--podman-volume-args` | volume 操作 |

注意 `--podman-path` 指向不存在的文件时，仅在**非演练模式**下才会直接报错退出。

(global-machine-ready)=
## 引擎就绪保障：默认 machine 自动启动（D16）

在 Windows/macOS 上 Podman 采用客户端-服务端架构：`podman` CLI 通过本机
ssh 连接到 podman machine（WSL2/Lima 虚拟机）。机器停止时（重启电脑、
WSL 回收后的常态），任何业务命令都会得到
`Cannot connect to Podman ... actively refused`（exit 125）。

xc 在执行需要服务端的命令前，会自动完成就绪保障：

1. `podman info` 真实探活（`podman --version` 是纯本地调用，证明不了服务端在线）；
2. 不可达时读取 `podman machine list --format json`，**仅**选择标记为
   `Default` 的机器；无默认机（如 Linux 原生 socket）不做任何猜测性动作；
3. 默认机停止则执行 `podman machine start`，随后每秒轮询直到就绪；
4. 机器已在运行但仍不可达、启动失败或等待超时，给出含手动命令建议的
   单行错误并以退出码 1 收口（不输出 Python traceback）。

```bash
xc up -d                      # 机器停止时自动 start，随后继续 up
xc --no-machine-start up -d   # 关闭自动启动（CI/脚本场景）
```

- 环境变量 `XC_NO_MACHINE_START=1` 等同 `--no-machine-start`；
- `XC_MACHINE_READY_TIMEOUT` 可调整就绪等待总预算（秒，默认 120，非法值回退默认）；
- `version`、`systemd create-unit` 与 `--dry-run` 不触发就绪保障；
- 该能力按能力探测工作（不按操作系统硬编码），设计细节见
  [../dev/parity/differences.md](../dev/parity/differences.md) 的 D16。

(global-dry-run)=
## 演练模式：`--dry-run`（重要边界）

```bash
xc --dry-run -f compose.yaml up
```

演练模式下：

1. **跳过启动时的 podman 版本探测**——不会因 podman 缺失而立即退出；
2. **写操作不执行**：创建/启动/停止/删除等会改变状态的 `podman` 调用只打印
   argv 不落地（翻译层照常工作，可用来审查「将要执行什么」）。

但它**不是完全离线**：查询类调用（经 `podman.output()`，如
`podman ps --filter ... --format json` 枚举现存容器、inspect 现状）仍会真实执行。
因此在没有 podman 的机器上，`xc --dry-run up` 仍可能在查询现存容器阶段失败——
需要纯离线分析 compose 文件时，请改用库 API，见
[../examples/02-parse-without-podman.md](../examples/02-parse-without-podman.md)。

## 并发与输出

```bash
xc --parallel 4 up -d       # 限制并发 podman 调用数（内部信号量）
xc --no-ansi up             # 不输出 ANSI 控制符（日志采集场景）
xc --verbose config         # 打印调试输出（含翻译出的 podman argv）
xc --no-cleanup down        # 不停止/删除既有 Pod 与容器（保留现场排障）
```

`--verbose` 是理解「compose 声明 → podman 命令」翻译过程的最直接开关。

## 下一步

- [03-library-api.md](03-library-api.md)：无 podman、无子进程的库用法
- [04-compose-processing.md](04-compose-processing.md)：文件处理管线
