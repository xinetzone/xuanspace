# SPDX-License-Identifier: GPL-2.0-only
"""``find_compose_files_recursively`` 行为测试（上游无对应单测，按上游
第 2374-2416 行真实分支自建：命中返回、向上爬升、到根停止、超深度 None）。"""

import os

from xuan_compose.discovery import COMPOSE_DEFAULT_LS, find_compose_files_recursively


def test_finds_files_in_start_dir(tmp_path) -> None:
    compose = tmp_path / "compose.yaml"
    override = tmp_path / "compose.override.yaml"
    compose.write_text("services: {}", encoding="utf-8")
    override.write_text("services: {}", encoding="utf-8")

    result = find_compose_files_recursively(str(tmp_path), COMPOSE_DEFAULT_LS)

    assert result is not None
    files, base = result
    assert base == os.path.abspath(str(tmp_path))
    assert files == [
        os.path.join(base, "compose.yaml"),
        os.path.join(base, "compose.override.yaml"),
    ]


def test_climbs_to_parent_directory(tmp_path) -> None:
    # 起始于深层子目录，compose 文件在祖父目录；候选顺序按 COMPOSE_DEFAULT_LS
    parent = tmp_path / "proj"
    deep = parent / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (parent / "compose.yaml").write_text("x: 1", encoding="utf-8")

    result = find_compose_files_recursively(str(deep), ["compose.yaml"])

    assert result is not None
    files, base = result
    assert base == os.path.abspath(str(parent))
    assert files == [os.path.join(base, "compose.yaml")]


def test_respects_custom_candidate_order_and_names(tmp_path) -> None:
    (tmp_path / "docker-compose.yml").write_text("x: 1", encoding="utf-8")
    (tmp_path / "compose.yaml").write_text("x: 1", encoding="utf-8")

    result = find_compose_files_recursively(
        str(tmp_path), ["docker-compose.yml", "compose.yaml"]
    )

    assert result is not None
    assert [os.path.basename(p) for p in result[0]] == [
        "docker-compose.yml",
        "compose.yaml",
    ]


def test_returns_none_when_no_file_within_max_depth(tmp_path) -> None:
    # 在足够深的空树中给很小的 max_depth，爬升耗尽即 None
    deep = tmp_path
    for part in ("a", "b", "c", "d", "e"):
        deep = deep / part
        deep.mkdir()

    assert find_compose_files_recursively(str(deep), ["compose.yaml"], max_depth=2) is None


def test_stops_at_filesystem_root(tmp_path) -> None:
    # max_depth 极大时也必须在根目录停止（parent == current 分支），不无限循环
    result = find_compose_files_recursively(
        str(tmp_path), ["definitely-not-a-real-compose-file-xyz.yaml"], max_depth=100
    )
    assert result is None
