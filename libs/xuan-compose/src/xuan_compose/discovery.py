# SPDX-License-Identifier: GPL-2.0-only
"""compose 文件发现。

迁移自上游 1.6.0 第 2374-2416 行：默认候选文件名与
``find_compose_files_recursively``（自起始目录向上最多 10 层查找）。
"""

import os

from .logging_utils import log

__all__ = ["COMPOSE_DEFAULT_LS", "find_compose_files_recursively"]

COMPOSE_DEFAULT_LS = [
    "compose.yaml",
    "compose.yml",
    "compose.override.yaml",
    "compose.override.yml",
    "podman-compose.yaml",
    "podman-compose.yml",
    "docker-compose.yml",
    "docker-compose.yaml",
    "docker-compose.override.yml",
    "docker-compose.override.yaml",
    "container-compose.yml",
    "container-compose.yaml",
    "container-compose.override.yml",
    "container-compose.override.yaml",
]


def find_compose_files_recursively(
    start_dir: str, compose_files: list[str], max_depth: int = 10
) -> tuple[list[str], str] | None:
    current_dir = os.path.abspath(start_dir)

    for _ in range(max_depth):
        found_files = []
        for compose_file in compose_files:
            file_path = os.path.join(current_dir, compose_file)
            if os.path.exists(file_path):
                found_files.append(file_path)

        if found_files:
            log.debug("Found compose files in %s: %s", current_dir, found_files)
            return found_files, current_dir

        parent_dir = os.path.dirname(current_dir)

        # If we've reached the root directory, stop searching
        if parent_dir == current_dir:
            break

        current_dir = parent_dir

    return None
