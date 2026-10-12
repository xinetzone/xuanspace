# 00 · 认识 xc：安装与三个入口

## xc 是什么

`xc` 是 **xuan-compose** 的日常短命令。xuan-compose 是一个 [Compose 规范](https://compose-spec.io/)
声明式容器编排引擎：读取 `compose.yaml`，把声明式的服务定义翻译成
[podman](https://podman.io/) 命令并执行，与上游
[containers/podman-compose](https://github.com/containers/podman-compose) v1.6.0
的 CLI 与解析行为逐行对等（argparse 参数表快照零差异）。

- **规范名**：`xuan-compose`——与上游 `podman-compose` 一一对应，用于对拍、跨环境分发；
- **短别名**：`xc`——同一入口的短名，与 monorepo 内 `xs`/`okf` 短命令风格一致，
  本文档所有示例默认使用它；
- **模块入口**：`python -m xuan_compose`。

三个入口同指 `xuan_compose.cli.main:main`，行为完全一致，
仅 `--help` 中的 `prog` 随调用名呈现（差异 D12，见 [../dev/parity/differences.md](../dev/parity/differences.md)）。

> **命名提示**：短别名 `xc` 与外部任务运行器 [xc（xcfile.dev）](https://xcfile.dev/) 重名，
> 同一环境的安装顺序决定 `xc` 指向。别名面向 Xuanspace 环境使用；
> 跨环境分发或创建 systemd unit 时请使用规范名 `xuan-compose`。

## 环境要求

| 项 | 要求 |
|----|------|
| Python | ≥ 3.14（`requires-python` 硬门槛，导入期即守卫） |
| podman | 已安装且 `podman --version` 可执行（除 `version` 等极少数命令外，启动即探测） |
| 依赖 | python-dotenv ≥ 1.0、PyYAML ≥ 6.0 |

注意：`xc` **不是**容器运行时，它只是 podman 的编排前端——真正的容器由 podman 创建。
在未安装 podman 的机器上，除显式的库用法（见 [03-library-api.md](03-library-api.md)）外，
CLI 会在启动探测时给出如下错误并以退出码 1 终止：

```text
ERROR ... failed to check if podman is installed: [WinError 2] ...
CRITICAL ... It seems that you either do not have `podman` installed ...
```

## 安装

于库目录（`libs/xuan-compose`）以可编辑方式安装：

```bash
pip install -e .
# 或连同测试与 lint 可选依赖
pip install -e ".[test,lint]"
```

安装后同时获得 `xc`、`xuan-compose` 两个 console script 与 `xuan_compose` 库包。
构建后端为 scikit-build-core（纯 Python 包，wheel 中 `cmake = false`，产出
`py3-none-any` 通用 wheel）。

## 验证安装

```bash
xc --version        # 打印 1.6.0+xuan.1（对等上游 1.6.0 的本库版本号）
xc --help           # 查看 24 个命令与全部全局选项
xc help             # help 伪命令，效果同上
```

在无 podman 的机器上，查看帮助不需要运行时——`--help` 由 argparse 直接处理。

## 最小工作循环

在装有 podman 的环境中，于任意目录编写 `compose.yaml`：

```yaml
services:
  web:
    image: docker.io/library/nginx:stable
    ports:
      - "8080:80"
```

随后：

```bash
xc up -d            # 后台创建并启动整个栈
xc ps               # 查看容器状态
xc logs -f web      # 跟踪 web 服务日志
xc down             # 停止并清理整个栈
```

完整走一遍见 [../examples/00-nginx-quickstart.md](../examples/00-nginx-quickstart.md)。

## 库优先：import 无副作用

与许多「导入即读 argv、导入即起子进程」的工具不同，`xuan_compose` 库包
**仅 import 不产生任何副作用**：不读 `sys.argv`、不启动子进程、不要求 podman 存在。
`sys.argv` 的真实读取只存在于 `cli/main.py` 单点，`subprocess` 的导入只存在于
`runner.py` 单一边界模块。嵌入与测试用法见 [03-library-api.md](03-library-api.md)。

## 下一步

- [01-cli-overview.md](01-cli-overview.md)：24 个命令的分组地图
- [../examples/00-nginx-quickstart.md](../examples/00-nginx-quickstart.md)：15 分钟跑通第一个栈
- 来源、许可证（GPL-2.0-only）与完整安装说明：[../../../README.md](../../../README.md)
