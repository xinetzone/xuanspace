# 参考索引（reference）

本目录是无叙述的参数信源，内容取自 `xc --help` 与各命令 `xc <cmd> --help`
的实测输出（版本 `1.6.0+xuan.1`），与上游 podman-compose 1.6.0 的
argparse 快照零差异（除 `prog`）。

| 文档 | 内容 |
|------|------|
| [global-options.md](global-options.md) | 命令名之前的全部全局选项全表 |
| [commands.md](commands.md) | 24 个命令的用途与逐命令选项表 |

使用指南与实战：[../guide/index.md](../guide/index.md)、
[../examples/index.md](../examples/index.md)。
工程证据（对等性/差异台账）：[../dev/index.md](../dev/index.md)。

```{toctree}
:hidden:
:maxdepth: 1

global-options
commands
```
