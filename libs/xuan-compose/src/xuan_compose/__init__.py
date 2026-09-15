# SPDX-License-Identifier: GPL-2.0-only
"""
xuan_compose - Compose 规范声明式容器编排引擎。

由 containers/podman-compose v1.6.0（GPL-2.0-only）分层重构而来，
行为与上游 1.6.0 对等，包结构按执行层/规范层/翻译层/引擎层/命令层/CLI 层拆分。

库的使用方式是显式构造引擎，导入本包不产生任何副作用（不读取 sys.argv、
不创建全局引擎实例、不发起子进程或网络访问）。
"""

__version__ = "1.6.0+xuan.1"
