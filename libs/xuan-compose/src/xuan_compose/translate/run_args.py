# SPDX-License-Identifier: GPL-2.0-only
"""运行参数翻译（T4 仅迁入 ``is_local``，翻译自 podman_compose.py 第 3391-3397 行；
其余 run_args 系列函数在 T5 补齐）。"""

from typing import Any

__all__ = [
    "is_local",
]


def is_local(container: dict[str, Any]) -> bool:
    """Test if a container is local, i.e. if it is
    * prefixed with localhost/
    * has a build section and is not prefixed
    """
    image = container.get("image", "")
    return image.startswith("localhost/") or ("build" in container and "/" not in image)
