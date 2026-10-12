# 使用指南索引（guide）

本目录回答「**xc 是什么、怎么用**」。面向 xuan-compose 的使用者，
按学习路径排列；命令的完整参数表见 [../reference/index.md](../reference/index.md)，
可运行的完整用例见 [../examples/index.md](../examples/index.md)。

## 指南列表

| 序号 | 文档 | 回答的问题 | 前置 |
|------|------|-----------|------|
| 00 | [00-introduction.md](00-introduction.md) | xc 从哪里来、如何安装、三个等价入口、命名注意 | 无 |
| 01 | [01-cli-overview.md](01-cli-overview.md) | 命令行结构、24 个命令如何分组、如何查帮助 | 00 |
| 02 | [02-global-options.md](02-global-options.md) | 全局选项如何控制文件、项目名、podman 与演练 | 01 |
| 03 | [03-library-api.md](03-library-api.md) | 不启动子进程，如何把 xuan_compose 当库用 | 00 |
| 04 | [04-compose-processing.md](04-compose-processing.md) | compose.yaml 如何被发现、插值、合并与归一化 | 01 |

## 推荐路径

- **只想跑容器**：00 → 01 → 做一遍 [examples/00](../examples/00-nginx-quickstart.md)
- **要写 compose 文件**：00 → 01 → 04 → [examples/01](../examples/01-web-redis.md)
- **要在 Python 中嵌入/测试**：00 → 03 → [examples/02](../examples/02-parse-without-podman.md)

> 本目录讲「怎么用」；「为什么这样设计、与上游逐行对等的证据」属于工程证据，
> 见 [../dev/index.md](../dev/index.md)，两者互不重复。

```{toctree}
:hidden:
:maxdepth: 1

00-introduction
01-cli-overview
02-global-options
03-library-api
04-compose-processing
```
