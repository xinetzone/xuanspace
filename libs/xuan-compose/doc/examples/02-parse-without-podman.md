# 02 · 无 podman 环境：纯解析演练（Windows/Linux 均可）

`xc` CLI 启动时会探测 podman，没有 podman 的机器（如开发笔记本、CI 的静态
检查阶段）无法执行 `xc config/up`。但**库路径**不读 argv、不起子进程、
不要求 podman——插值、合并、归一化、翻译都能照常运行。

本例全部命令在 Windows 原生 Python 3.14 下实测通过，Linux/macOS 同理。

## 准备

```bash
# 于 libs/xuan-compose 安装（editable），或仅设置 PYTHONPATH=src
pip install -e .
python --version            # 必须 >= 3.14
```

建立演练目录：

```bash
mkdir xc_demo && cd xc_demo
```

```yaml
# compose.yaml
services:
  web:
    image: docker.io/library/nginx:stable
    ports: ["${WEB_PORT:-8080}:80"]
    depends_on: [redis]
    environment:
      TZ: Asia/Shanghai
  redis:
    image: docker.io/library/redis:7-alpine
    volumes: ["redis-data:/data"]
volumes:
  redis-data:
```

## 脚本

保存为 `demo_inspect.py`：

```python
"""不依赖 podman 的 compose 静态检查示例。"""

import json
import os

from xuan_compose.engine import ComposeEngine
from xuan_compose.cli.parser import parse_args

engine = ComposeEngine()

# 1) 纯解析 argv：等价于 `xc -f compose.yaml config --quiet`
args = parse_args(engine, ["-f", "compose.yaml", "config", "--quiet"])
print("command =", args.command, "| files =", args.file, "| quiet =", args.quiet)

# 2) 发现 → 插值 → 合并 → 归一化（不需要 podman）
engine._parse_compose_file()
print("project =", engine.project_name)
print("services =", sorted(engine.all_services))

# 3) 读取归一化后的服务模型
web = engine.services["web"]
print("web ports =", web["ports"])  # 插值已生效
print("depends_on =", json.dumps(web["depends_on"], ensure_ascii=False))
print("environment =", web["environment"])
print("volumes =", sorted(engine.vols))
```

## 运行与预期输出

```bash
# Windows PowerShell
$env:WEB_PORT="9090"; python demo_inspect.py
# Linux/macOS
WEB_PORT=9090 python demo_inspect.py
```

```text
command = config | files = ['compose.yaml'] | quiet = True
project = xc_demo
services = ['redis', 'web']
web ports = ['9090:80']
depends_on = {"redis": {"condition": "service_started"}}
environment = {'TZ': 'Asia/Shanghai'}
volumes = ['redis-data']
```

观察到的三件事，正好对应 [../guide/04-compose-processing.md](../guide/04-compose-processing.md)：

1. **项目名**取目录名 `xc_demo`（与 `-p` 等价的隐式命名空间）；
2. **插值**把 `${WEB_PORT:-8080}` 替换为环境值 `9090`；
3. **归一化**把列表简写 `depends_on: [redis]` 展开为带默认条件的字典。

`engine.merged_yaml` 还持有合并归一化后的完整 YAML 文本（服务按名字典序输出），
等价于 `xc config` 的产物，可直接落盘做差异审计：

```python
with open("resolved.compose.yaml", "w", encoding="utf-8") as f:
    f.write(engine.merged_yaml)
```

## 再进一步：纯翻译函数

不构造引擎，也能验证短挂载语法如何翻译（规则见
{ref}`库 API·纯翻译函数 <translate-functions>`）：

```python
from xuan_compose.translate.mounts import parse_short_mount

print(parse_short_mount("./data:/var/lib/mysql:Z", basedir="."))
# {'type': 'bind', 'source': '<basedir>/data', 'target': '/var/lib/mysql',
#  'bind': {'propagation': 'Z'}}
print(parse_short_mount("redis-data:/data:ro", basedir=".")["read_only"])
# True
```

## 适用场景与边界

- ✅ CI 校验 compose 文件可解析、服务清单正确、插值与归一化结果符合预期；
- ✅ 自定义工具提取服务/端口/卷信息，或审计「最终生效配置」；
- ✅ 单元测试中断言翻译产物（本库测试即全部 fake 子进程边界，零真实 podman）；
- ❌ 要真正创建容器仍需 podman——库路径不替代运行时，回到
  [00-nginx-quickstart.md](00-nginx-quickstart.md) 的完整生命周期。

## 相关

- 库 API 总览：[../guide/03-library-api.md](../guide/03-library-api.md)
- `--dry-run` 的能力与查询类调用边界：
  [../guide/02-global-options.md](../guide/02-global-options.md)
