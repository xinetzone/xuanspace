# 01 · 双服务栈：Web + Redis

> 前置：完成 [00-nginx-quickstart.md](00-nginx-quickstart.md)，机器上有 podman。

本例引入三件新事物：**服务依赖**、**变量插值**、**命名卷持久化**，
并演示 `xc run` 的一次性任务。

## compose 文件

```bash
mkdir xc-redis && cd xc-redis
cat > compose.yaml <<'YAML'
services:
  web:
    image: docker.io/library/nginx:stable
    ports:
      - "${WEB_PORT:-8080}:80"
    depends_on:
      redis:
        condition: service_started

  redis:
    image: docker.io/library/redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  redis-data:
YAML
```

要点：

- `"${WEB_PORT:-8080}:80"` 是 bash 风格插值，未设置环境变量时取 8080，
  规则见 {ref}`compose 处理管线·插值 <interpolation-step>`；
- `depends_on` 声明 web 依赖 redis，归一化时即使写成 `[redis]` 列表也会补出
  `condition: service_started`；
- 顶层 `volumes:` 声明命名卷 `redis-data`，挂到 redis 的 `/data`，
  `xc down`（不带 `-v`）不会删除它。

## 校验插值与归一化

```bash
xc config --services                # 逐行打印：redis web（按定义/合并顺序）
WEB_PORT=9090 xc config | grep -A2 ports
xc config -q && echo "compose OK"   # 只解析，用于 CI 合法性检查
```

## 启动并观察启动顺序

```bash
xc up -d
xc ps
```

`depends_on` 使 redis 先于 web 处理。注意 `service_started` 只保证容器已启动，
**不**保证 redis 已可接受连接——应用侧仍应自带重试（这是 Compose 规范语义，
不是 xc 的限制）。

## 用 xc run 跑一次性任务

借助 redis 服务的镜像与网络，执行一次 `redis-cli` 探活：

```bash
xc run --rm redis redis-cli -h redis ping
# 期望输出：PONG
```

- `xc run` 按 `redis` 服务的定义（镜像、网络、卷）创建一个**临时容器**跑命令；
- `--rm` 退出后删除该临时容器（与常驻服务容器无关）；
- 服务名 `redis` 在项目网络内可被 DNS 解析，因此 `-h redis` 能连到常驻实例。

写入再读回，验证命名卷持久化：

```bash
xc run --rm redis redis-cli -h redis SET greeting hello
xc run --rm redis redis-cli -h redis GET greeting    # -> "hello"
xc restart redis
xc run --rm redis redis-cli -h redis GET greeting    # 重启后仍为 "hello"
```

## 日志与局部操作

```bash
xc logs --tail 50 redis            # 只看 redis
xc logs -f web redis               # 同时跟踪两个服务
xc stop redis                      # 只停 redis（web 仍在，但依赖已不满足）
xc start redis                     # 单独拉起
```

## 拆除

```bash
xc down                            # 删容器/网络，保留 redis-data 卷
xc down -v                         # 确认不再需要数据时，连带删除命名卷
```

## 本例涉及的全局选项

```bash
WEB_PORT=9090 xc up -d             # 用进程环境覆盖插值默认值
xc --env-file .env.staging up -d   # 或用额外环境文件提供变量
xc -p redisdemo up -d              # 换项目名，与其他栈并存隔离
```

## 与单服务例的差异回顾

| 主题 | 本例新增 |
|------|---------|
| 依赖 | `depends_on` 与启动顺序、`service_started` 语义 |
| 配置 | `${VAR:-default}` 插值、环境文件 |
| 持久化 | 顶层命名卷与 `down -v` 的区别 |
| 临时任务 | `xc run --rm <service> <cmd>` 复用服务定义 |

无 podman 时如何对这份文件做静态检查：
[02-parse-without-podman.md](02-parse-without-podman.md)。
