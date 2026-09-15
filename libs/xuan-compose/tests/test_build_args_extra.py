# SPDX-License-Identifier: GPL-2.0-only
"""translate.build 分支补测（移植 20 例覆盖 dockerfile 探测/ssh 路径/
build_arg 主路径；这里补 inline 临时文件与冲突、字符串 build、
custom dockerfile 缺失、labels dict/additional_contexts/target/tags/
cache/no-cache/pull 等选项分支——上游第 3549-3685 行）。"""

import argparse
import os
from unittest import mock

import pytest

from xuan_compose.translate.build import (
    adjust_build_ssh_key_paths,
    container_to_build_args,
)


def _args(**kw: object) -> argparse.Namespace:
    defaults = {"build_arg": [], "no_cache": False, "pull": None}
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _compose() -> mock.Mock:
    c = mock.Mock()
    c.dirname = "/base"
    c.declared_secrets = {}
    c.project_name = "proj"
    return c


class TestAdjustSsh:
    def test_plain_agent_id_returned_as_is(self) -> None:
        assert adjust_build_ssh_key_paths(_compose(), "default") == "default"

    def test_key_equals_path_joined_to_dirname(self) -> None:
        out = adjust_build_ssh_key_paths(_compose(), "mykey=./keys/id_rsa")
        assert out == "mykey=" + os.path.join("/base", "./keys/id_rsa")


class TestContainerToBuildArgs:
    def test_inline_dockerfile_written_and_callback_registered(self) -> None:
        # inline 写临时文件后走自定义 dockerfile 分支，path_exists 必须能识别
        # 刚创建的绝对路径（os.path.join(ctx, 绝对路径) 仍是该绝对路径）
        callbacks = []
        out = container_to_build_args(
            _compose(),
            {"build": {"context": ".", "dockerfile_inline": "FROM scratch\n"}, "image": "img"},
            _args(),
            path_exists=os.path.exists,
            cleanup_callbacks=callbacks,
        )
        assert len(callbacks) == 1
        # -f 指向生成的 .containerfile 临时文件
        f_index = out.index("-f")
        temp_path = out[f_index + 1]
        assert temp_path.endswith(".containerfile")
        assert os.path.exists(temp_path)
        callbacks[0]()  # 清理回调删除临时文件
        assert not os.path.exists(temp_path)

    def test_inline_conflicts_with_dockerfile(self) -> None:
        with pytest.raises(OSError, match="can't be used simultaneously"):
            container_to_build_args(
                _compose(),
                {"build": {"context": ".", "dockerfile": "DF", "dockerfile_inline": "x"},
                 "image": "img"},
                _args(),
                path_exists=lambda p: True,
            )

    def test_custom_dockerfile_missing_raises(self) -> None:
        with pytest.raises(OSError, match="Dockerfile not found"):
            container_to_build_args(
                _compose(),
                {"build": {"context": "./ctx", "dockerfile": "Custom.df"}, "image": "img"},
                _args(),
                path_exists=lambda p: False,
            )

    def test_git_context_with_dockerfile_passes_through(self) -> None:
        url = "https://example.com/repo.git#branch:dir"
        out = container_to_build_args(
            _compose(),
            {"build": {"context": url, "dockerfile": "Alt.df"}, "image": "img"},
            _args(),
            path_exists=lambda p: False,
        )
        # git 上下文不探测本地文件，-f 原样透传
        assert out[0:2] == ["-f", "Alt.df"]
        assert out[-1] == url

    def test_full_option_set(self) -> None:
        cnt = {
            "image": "img",
            "platform": "linux/amd64",
            "build": {
                "context": ".",
                "extra_hosts": ["host:1.2.3.4"],
                "tags": ["img:v1", "img:latest"],
                "labels": {"vcs": "git", "empty": ""},
                "additional_contexts": ["shared=./shared"],
                "target": "builder",
                "ssh": ["default=./key"],
                "args": {"A": "1"},
                "cache_from": ["prev:latest"],
                "cache_to": ["prev:latest"],
            },
        }
        out = container_to_build_args(_compose(), cnt, _args(no_cache=True, pull="always"),
                                      path_exists=lambda p: True)
        assert ["--platform", "linux/amd64"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["--add-host", "host:1.2.3.4"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["--build-context=shared=./shared"] in [[x] for x in out]
        assert ["--target", "builder"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["--ssh", "default=" + os.path.join("/base", "./key")] in [
            out[i : i + 2] for i in range(len(out) - 1)
        ]
        assert ["--build-arg", "A=1"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["--cache-from", "prev:latest"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert ["--cache-to", "prev:latest"] in [out[i : i + 2] for i in range(len(out) - 1)]
        assert "--no-cache" in out
        assert "--pull=always" in out
        # 主 tag + 两个附加 tag
        assert out.count("-t") == 3
        assert ["--label", "vcs=git"] in [out[i : i + 2] for i in range(len(out) - 1)]
