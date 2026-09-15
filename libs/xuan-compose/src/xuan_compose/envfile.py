# SPDX-License-Identifier: GPL-2.0-only
""".env 文件读取。

迁移自上游 1.6.0 第 2368-2371 行 ``dotenv_to_dict``。
"""

import os

from dotenv import dotenv_values


def dotenv_to_dict(dotenv_path: str) -> dict[str, str | None]:
    if not os.path.isfile(dotenv_path):
        return {}
    return dotenv_values(dotenv_path)
