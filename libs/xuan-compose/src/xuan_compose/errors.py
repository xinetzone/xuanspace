# SPDX-License-Identifier: GPL-2.0-only
"""xuan_compose 异常类型。

迁移自 podman_compose.PodmanComposeError（上游 1.6.0 第 3276-3277 行），
保留类名以维持错误消息与捕获语义的逐行对等。
"""


class PodmanComposeError(Exception):
    """用户可恢复的 compose 处理错误（如必填变量缺失）。"""


class PodmanEngineNotReadyError(PodmanComposeError):
    """Podman 服务端不可达，且引擎就绪保障未能恢复（如 machine 停止且自动
    启动失败/被禁用）。

    继承 :class:`PodmanComposeError`，使 ``main()`` 统一以
    ``Error: <msg>`` + 退出码 1 收口，向用户呈现可操作信息而非 traceback。
    """
