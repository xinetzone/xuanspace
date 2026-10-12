# 00 · 15 分钟跑通第一个栈：nginx

> 前置：已 `pip install -e .` 且机器上可执行 `podman --version`
> （安装见 [../guide/00-introduction.md](../guide/00-introduction.md)）。

## 1. 准备目录与 compose 文件

```bash
mkdir xc-nginx && cd xc-nginx
cat > compose.yaml <<'YAML'
services:
  web:
    image: docker.io/library/nginx:stable
    ports:
      - "8080:80"
YAML
```

`services.web` 声明了一个服务：镜像 `nginx:stable`，把容器 80 端口发布到宿主 8080。

## 2. 先解析，再启动

养成「先 config 后 up」的习惯：

```bash
xc config --services         # -> web，确认服务被正确识别
xc up -d                     # 后台创建并启动
```

`-d/--detach` 让容器后台运行；首次执行会自动拉取镜像并创建项目网络等资源。
项目名默认取目录名 `xc-nginx`，所有资源都以它为命名空间打标签。

## 3. 观察

```bash
xc ps                        # 查看 web 容器状态（Up/端口映射）
xc port web 80               # 打印 80 端口映射到的宿主端口（0.0.0.0:8080）
curl -I http://localhost:8080
xc logs --tail 20 web        # 查看访问日志
```

进入容器检查（可选）：

```bash
xc exec web sh               # 在运行中的 web 容器内开 shell，exit 退出
```

## 4. 变更与重启

修改 `compose.yaml` 后，对受影响服务执行重建式重启：

```bash
xc up -d                     # xc 据配置/镜像变化决定是否重建容器
xc restart web               # 仅重启，不重建
```

也可以临时伸缩同一服务的实例数（端口由 podman 侧分配）：

```bash
xc up -d --scale web=3
xc ps
```

## 5. 停止与清理

```bash
xc stop                      # 停止但保留容器，随后 xc start 可再拉起
xc down                      # 停止并删除容器、网络（保留镜像）
xc down -v                   # 连同命名卷与匿名卷一并删除
```

只想发信号或暂停：

```bash
xc kill -s TERM web          # 发送指定信号（默认 KILL）
xc pause web                 # 暂停；xc unpause web 恢复
```

## 6. 无 podman 时如何提前演练

在没有 podman 的机器上，`xc config` 与 `xc --dry-run` 的行为边界见
{ref}`全局选项·演练模式 <global-dry-run>`；
纯离线校验 compose 文件请用库 API，见
[02-parse-without-podman.md](02-parse-without-podman.md)。

## 本例命令回顾

| 阶段 | 命令 |
|------|------|
| 校验 | `xc config --services` |
| 启动 | `xc up -d` |
| 状态 | `xc ps`、`xc port web 80`、`xc logs --tail 20 web` |
| 进入 | `xc exec web sh` |
| 变更 | `xc up -d`、`xc restart web`、`xc up -d --scale web=3` |
| 停止 | `xc stop` / `xc start` |
| 拆除 | `xc down`（`-v` 连带卷） |

下一篇：[01-web-redis.md](01-web-redis.md)——双服务、依赖与持久化。
