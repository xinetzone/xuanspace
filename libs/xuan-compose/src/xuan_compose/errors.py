# SPDX-License-Identifier: GPL-2.0-only
"""xuan_compose 异常类型。

迁移自 podman_compose.PodmanComposeError（上游 1.6.0 第 3276-3277 行），
保留类名以维持错误消息与捕获语义的逐行对等。
"""


class PodmanComposeError(Exception):
    """用户可恢复的 compose 处理错误（如必填变量缺失）。"""
