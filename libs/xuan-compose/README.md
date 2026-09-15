# xuan-compose

> Compose 规范声明式容器编排引擎，Xuanspace（玄境）monorepo 纯 Python 库。

## 来源与许可

本库衍生自 [containers/podman-compose](https://github.com/containers/podman-compose) v1.6.0：

- 上游 pinned commit：`e3df10472e194ab6d547b5ad25542c5c79e1a5fb`
- 重构方式：将上游 5541 行单体 `podman_compose.py` 翻译式搬迁为分层包，行为与 1.6.0 对等
- 许可证：**GPL-2.0-only**（继承上游，不得重新许可），完整许可证文本见 [LICENSE](LICENSE)；全部源文件携带 `SPDX-License-Identifier: GPL-2.0-only`

## 安装

```bash
# 于本目录（libs/xuan-compose）
pip install -e .
# 或带测试依赖
pip install -e ".[test]"
```

要求 Python >= 3.14。

> 构建说明：scikit-build-core 的 CMake 产物全部落在 `build/{wheel_tag}/`（已被
> xuanspace 根 `.gitignore` 忽略），源码根目录保持干净。因此本库**不使用**
> `editable.mode = "inplace"`（该模式会强制在源码目录内 configure CMake）；
> 测试经 `pytest` 的 `pythonpath=["src"]` 始终直连 src，无需重装。

## 用法

> 详细 API 文档随重构切片逐步补全（T10 收尾）。

```python
import xuan_compose

print(xuan_compose.__version__)
```

命令行兼容入口（不注册 console script，遵循 libs 准入规则）：

```bash
python -m xuan_compose --help
```

## 测试

```bash
pytest --cov=src/xuan_compose --cov-report=term-missing
ruff check src tests
mypy src
```

**门禁平台（与上游 CI 一致的 POSIX 语义）**：路径拼接类用例的断言按 POSIX 分隔符书写（上游同构），
权威门禁在 WSL 的 Python 3.14 中执行：

```bash
# WSL 内一次性准备：python3.14 -m venv ~/.venvs/xuan-gate
# ~/.venvs/xuan-gate/bin/pip install pytest pytest-cov parameterized pyyaml python-dotenv
cd /mnt/d/spaces/SpecWeave/projects/xuanspace/libs/xuan-compose
PYTHONPATH=src ~/.venvs/xuan-gate/bin/pytest -q
```

Windows 原生 py3.14 可跑除路径分隔符敏感用例外的全部测试（差异仅为 `\`/`/` 与盘符根，
非逻辑差异；`normalize` 原样使用 `os.path.join`，上游在 Windows 上行为相同）。

需要真实 podman 守护进程的上游集成测试不在本库门禁内；可在 WSL（podman-machine-default）中手工验证。
