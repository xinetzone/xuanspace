# 示例索引（examples）

本目录提供可照做的完整用例，全部以 `xc` 命令讲解。

> **运行前提**：示例 00、01 需要一台装有 podman 的机器（Linux/WSL/macOS），
> `xc` 启动时会探测 `podman --version`。只有示例 02 是纯库解析，
> 无需 podman 即可在任意 Python 3.14 环境运行。

## 示例列表

| 序号 | 文档 | 内容 | 涉及命令/概念 | 难度 |
|------|------|------|--------------|------|
| 00 | [00-nginx-quickstart.md](00-nginx-quickstart.md) | 单服务 nginx 栈完整生命周期 | up/ps/logs/port/exec/down、项目命名空间 | ⭐ 入门 |
| 01 | [01-web-redis.md](01-web-redis.md) | Web + Redis 双服务 | depends_on、插值、网络、命名卷、run 一次性任务 | ⭐⭐ 进阶 |
| 02 | [02-parse-without-podman.md](02-parse-without-podman.md) | 无 podman 环境的纯解析演练 | ComposeEngine、parse_args、翻译函数、config 模型 | ⭐ 入门 |

## 通用循环

任何栈的日常操作都是同一组命令：

```bash
xc up -d            # 后台启动
xc ps               # 看状态
xc logs -f          # 跟日志
xc down             # 拆除（-v 连带删除命名卷与匿名卷）
```

## 学习路径建议

1. **零起点**：先在有 podman 的机器做 [00](00-nginx-quickstart.md)；
2. **写多服务文件**：再做 [01](01-web-redis.md)，配合
   [../guide/04-compose-processing.md](../guide/04-compose-processing.md) 理解归一化；
3. **无 podman/做 CI**：直接看 [02](02-parse-without-podman.md)。

```{toctree}
:hidden:
:maxdepth: 1

00-nginx-quickstart
01-web-redis
02-parse-without-podman
```
