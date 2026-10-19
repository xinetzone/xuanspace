# 03 · 库 API：无 podman、无子进程的用法

xuan-compose 首先是一个**库**，`xc` 只是它的 CLI 装配。库包导入零副作用
（不读 argv、不起子进程、不要求 podman），因此解析、插值、归一化、翻译
都可以在普通 Python 进程与单元测试中直接使用。

## 纯解析：ComposeEngine + parse_args

```python
from xuan_compose.engine import ComposeEngine
from xuan_compose.cli.parser import parse_args

engine = ComposeEngine()  # 纯状态对象
args = parse_args(engine, ["-f", "compose.yaml", "config", "--quiet"])
print(args.command, args.file, args.quiet)
# -> config ['compose.yaml'] True
```

`parse_args` 是纯函数式解析，没有 compose 文件也能调用；解析结果同时回写到
`engine.global_args`。

## 加载 compose 文件

```python
engine._parse_compose_file()  # 发现 → 插值 → 合并 → 归一化
print(engine.project_name)  # -> xc_demo（默认取目录名）
print(sorted(engine.all_services))  # -> ['redis', 'web']
```

以下实测来自一份包含 `nginx` 与 `redis` 两个服务、位于 `xc_demo/` 目录的
`compose.yaml`（Windows 原生 Python 3.14，无 podman）：

```text
command = config | files = ['compose.yaml'] | quiet = True
project = xc_demo
services = ['redis', 'web']
```

这一入口适合：CI 中校验 compose 文件、提取服务清单、做配置静态分析。
完整可运行演练见 [../examples/02-parse-without-podman.md](../examples/02-parse-without-podman.md)。

(translate-functions)=
## 纯翻译函数：不依赖引擎实例

`translate/` 层把声明式片段翻译为 podman argv 片段，函数可独立调用：

```python
from xuan_compose.translate.mounts import parse_short_mount

m = parse_short_mount("./data:/var/lib/mysql:Z", basedir="/home/u/proj")
# {
#   'type': 'bind',
#   'source': '/home/u/proj/data',
#   'target': '/var/lib/mysql',
#   'bind': {'propagation': 'Z'},
# }

v = parse_short_mount("dbdata:/var/lib/postgresql/data:ro", basedir=".")
# v['source'] == 'dbdata'；v['target'] == '/var/lib/postgresql/data'
# v['read_only'] is True；v['type'] == 'volume'
```

短语法 `源:目标[:选项]` 的判定规则：

- 源是相对路径（`./data`、`data`）→ `bind` 挂载，`source` 相对 `basedir` 解析为绝对路径；
- 源是名字（`dbdata`）→ 命名卷 `volume`；
- 选项 `ro` → `read_only=True`，`Z/z` → SELinux 传播属性。

> Windows 实测时 `basedir` 按本机路径语义解析（上述 `/home/u/proj` 示例在
> Windows 上会得到盘符形态），这与上游在 Windows 上的 `os.path` 行为对等，
> 不是逻辑差异（见 [../dev/findings-ledger.md](../dev/findings-ledger.md)）。

其他可独立使用的翻译模块：`container_args`（容器 argv 主翻译）、`networks`、
`ports`、`build`、`run_args`、`resources`、`secrets`。

## 完整运行时：async_main

需要装配 podman runner、版本探测与命令分发时，使用 CLI 的异步装配链：

```python
import asyncio
from xuan_compose.cli.main import async_main

asyncio.run(async_main(["--dry-run", "-f", "compose.yaml", "up"]))
```

这等价于在命令行执行 `xc --dry-run -f compose.yaml up`，同样遵守
{ref}`全局选项·演练模式 <global-dry-run>` 一节中
`--dry-run` 的边界：跳过版本探测与写操作，但查询类 podman 调用仍真实发生。

## 分层边界（对嵌入者有用的保证）

- `sys.argv` 的真实读取只存在于 `xuan_compose.cli.main` 单点；
- `import subprocess` 只存在于 `runner.py` 单一边界模块，其他层只消费它
  再导出的 `CalledProcessError`；
- `commands` 层不 import 引擎，handler 以 `compose: Any` 鸭子类型接收引擎；
- 依赖严格单向无环（108 条内部 import 经独立审查核对），架构图见
  [../../../README.md](../../../README.md)「架构分层」。

## 下一步

- [04-compose-processing.md](04-compose-processing.md)：`_parse_compose_file()` 内部四步
- [../examples/02-parse-without-podman.md](../examples/02-parse-without-podman.md)：离线演练完整脚本
