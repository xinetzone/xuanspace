# SPDX-License-Identifier: GPL-2.0-only
"""translate.mounts 分支补测（上游 test_volumes 仅 1 例 parse_short_mount；
本文件按上游第 160-257、582-799 行补齐短语法全形态、fix_mount_dict
命名规则、assert_volume 的 bind 创建/卷 inspect-create、五种 mount
类型的参数串与 get_mount_args 分流——podman 调用全部 AsyncMock）。"""

import asyncio
import os
from unittest import mock

import pytest

from xuan_compose.errors import PodmanComposeError
from xuan_compose.runner import CalledProcessError
from xuan_compose.translate.mounts import (
    assert_volume,
    fix_mount_dict,
    get_mnt_dict,
    get_mount_args,
    mount_desc_to_mount_args,
    mount_desc_to_volume_args,
    parse_short_mount,
)


class TestParseShortMount:
    def test_anonymous_single_segment(self) -> None:
        m = parse_short_mount("/var/lib/mysql", "/base")
        assert m["source"] is None
        assert m["target"] == "/var/lib/mysql"
        assert m["type"] == "volume"

    def test_two_segment_option_form(self) -> None:
        # /data:rw：第二段不以 / 开头 → 选项形态，source 匿名
        m = parse_short_mount("/data:rw", "/base")
        assert m["source"] is None
        assert m["target"] == "/data"
        assert m["read_only"] is False

    def test_three_segments(self) -> None:
        m = parse_short_mount("named:/data:ro", "/base")
        assert m["source"] == "named"
        assert m["target"] == "/data"
        assert m["read_only"] is True
        assert m["type"] == "volume"

    def test_too_many_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="could not parse mount"):
            parse_short_mount("a:b:c:d", "/base")

    def test_relative_bind_source_joined_to_basedir(self) -> None:
        m = parse_short_mount("./cache:/tmp/cache", "/base")
        assert m["type"] == "bind"
        assert m["source"] == os.path.abspath(os.path.join("/base", "./cache"))

    def test_ro_rw_consistency_and_propagation(self) -> None:
        m = parse_short_mount("named:/data:ro,cached,Z", "/base")
        assert m["read_only"] is True
        assert m["consistency"] == "cached"
        assert m["bind"] == {"propagation": "Z"}
        m2 = parse_short_mount("named:/data:rw,rshared", "/base")
        assert m2["read_only"] is False
        assert m2["bind"] == {"propagation": "rshared"}

    def test_unknown_option_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mount option bogus"):
            parse_short_mount("named:/data:bogus", "/base")


class TestFixMountDict:
    @staticmethod
    def _compose(vols=None):
        c = mock.Mock()
        c.project_name = "proj"
        c.vols = vols if vols is not None else {}
        c.format_name.side_effect = lambda *parts: "proj_" + "_".join(parts)[:12]
        return c

    def test_already_fixed_is_returned(self) -> None:
        mnt = {"type": "bind", "_vol": {"x": 1}}
        assert fix_mount_dict(self._compose(), mnt, "srv") is mnt

    def test_anonymous_volume_gets_generated_name(self) -> None:
        mnt = {"type": "volume", "source": None, "target": "/data"}
        out = fix_mount_dict(self._compose(), mnt, "web")
        assert out["_vol"]["name"].startswith("proj_web_")

    def test_named_volume_default_project_prefix(self) -> None:
        c = self._compose(vols={"data": {}})
        out = fix_mount_dict(c, {"type": "volume", "source": "data", "target": "/d"}, "web")
        assert out["_vol"]["name"] == "proj_data"

    def test_external_true_keeps_bare_source_name(self) -> None:
        c = self._compose(vols={"data": {"external": True}})
        out = fix_mount_dict(c, {"type": "volume", "source": "data", "target": "/d"}, "web")
        assert out["_vol"]["name"] == "data"

    def test_external_dict_can_override_name(self) -> None:
        c = self._compose(vols={"data": {"external": {"name": "real-ext"}}})
        out = fix_mount_dict(c, {"type": "volume", "source": "data", "target": "/d"}, "web")
        assert out["_vol"]["name"] == "real-ext"

    def test_external_dict_without_name_falls_back_to_source(self) -> None:
        c = self._compose(vols={"data": {"external": {}}})
        out = fix_mount_dict(c, {"type": "volume", "source": "data", "target": "/d"}, "web")
        assert out["_vol"]["name"] == "data"

    def test_bind_passes_through_without_vol(self) -> None:
        out = fix_mount_dict(self._compose(), {"type": "bind", "source": "/h", "target": "/c"}, "w")
        assert "_vol" not in out


class TestAssertVolume:
    def test_bind_existing_source_rewritten(self, tmp_path) -> None:
        c = mock.Mock()
        c.dirname = str(tmp_path)
        mnt = {"type": "bind", "source": "./host", "target": "/c", "bind": {}}
        (tmp_path / "host").mkdir()
        asyncio.run(assert_volume(c, mnt))
        assert mnt["source"] == os.path.realpath(str(tmp_path / "host"))
        c.podman.output.assert_not_called()

    def test_bind_missing_source_created_by_default(self, tmp_path) -> None:
        c = mock.Mock()
        c.dirname = str(tmp_path)
        mnt = {"type": "bind", "source": "auto-create", "target": "/c", "bind": {}}
        asyncio.run(assert_volume(c, mnt))
        assert (tmp_path / "auto-create").is_dir()

    def test_bind_missing_source_create_host_path_false_raises(self, tmp_path) -> None:
        c = mock.Mock()
        c.dirname = str(tmp_path)
        mnt = {
            "type": "bind",
            "source": "nope",
            "target": "/c",
            "bind": {"create_host_path": False},
        }
        with pytest.raises(ValueError, match="bind source path does not exist"):
            asyncio.run(assert_volume(c, mnt))

    def test_non_volume_non_bind_returns_early(self) -> None:
        c = mock.Mock()
        mnt = {"type": "tmpfs", "target": "/t"}
        asyncio.run(assert_volume(c, mnt))
        c.podman.output.assert_not_called()

    def test_existing_volume_inspected_no_create(self) -> None:
        c = mock.Mock()
        c.podman.output = mock.AsyncMock(return_value=b"mountpoint\n")
        mnt = {"type": "volume", "_vol": {"name": "data"}, "target": "/d"}
        asyncio.run(assert_volume(c, mnt))
        c.podman.output.assert_awaited_once_with([], "volume", ["inspect", "data"])

    def test_missing_external_volume_raises(self) -> None:
        c = mock.Mock()
        c.podman.output = mock.AsyncMock(side_effect=CalledProcessError(1, []))
        mnt = {"type": "volume", "_vol": {"name": "ext", "external": True}, "target": "/d"}
        with pytest.raises(PodmanComposeError, match="External volume \\[ext\\] does not exist"):
            asyncio.run(assert_volume(c, mnt))

    def test_missing_volume_created_with_labels_driver_opts(self) -> None:
        c = mock.Mock()
        c.project_name = "proj"
        # 调用序列：首次 inspect 失败 → create 成功 → 再次 inspect 成功
        c.podman.output = mock.AsyncMock(
            side_effect=[CalledProcessError(1, []), b"", b""]
        )
        mnt = {
            "type": "volume",
            "_vol": {
                "name": "data",
                "labels": ["a=b"],
                "driver": "local",
                "driver_opts": {"type": "nfs"},
            },
            "target": "/d",
        }
        asyncio.run(assert_volume(c, mnt))
        create_call = c.podman.output.await_args_list[1]
        args = create_call.args[2]
        assert args[0] == "create"
        assert "--label" in args and "io.podman.compose.project=proj" in args
        assert "a=b" in args
        assert args[-3:] == ["--opt", "type=nfs", "data"]
        # 创建后再次 inspect
        assert c.podman.output.await_args_list[2].args[2] == ["inspect", "data"]

    def test_missing_volume_minimal_create(self) -> None:
        c = mock.Mock()
        c.project_name = "proj"
        c.podman.output = mock.AsyncMock(
            side_effect=[CalledProcessError(1, []), b"", b""]
        )
        mnt = {"type": "volume", "_vol": {"name": "v"}, "target": "/d"}
        asyncio.run(assert_volume(c, mnt))
        args = c.podman.output.await_args_list[1].args[2]
        assert args == [
            "create",
            "--label",
            "io.podman.compose.project=proj",
            "--label",
            "com.docker.compose.project=proj",
            "v",
        ]


class TestMountArgStrings:
    def test_volume_ro_propagation_and_subpath(self) -> None:
        s = mount_desc_to_mount_args(
            {
                "type": "volume",
                "source": "data",
                "target": "/d",
                "read_only": True,
                "volume": {"propagation": "Z", "subpath": "sp"},
                "_vol": {"name": "real-data"},
            }
        )
        assert s == "type=volume,source=real-data,destination=/d,volume-propagation=Z,ro,subpath=sp"

    def test_tmpfs_size_and_mode(self) -> None:
        s = mount_desc_to_mount_args(
            {
                "type": "tmpfs",
                "target": "/t",
                "tmpfs": {"size": "100m", "mode": "1777"},
            }
        )
        assert s == "type=tmpfs,destination=/t,tmpfs-size=100m,tmpfs-mode=1777"

    def test_bind_selinux(self) -> None:
        s = mount_desc_to_mount_args(
            {"type": "bind", "source": "/h", "target": "/c", "bind": {"selinux": "Z"}}
        )
        assert s == "type=bind,source=/h,destination=/c,Z"

    def test_glob_and_image_types(self) -> None:
        assert mount_desc_to_mount_args(
            {"type": "glob", "source": "./d*", "target": "/c"}
        ) == "type=glob,source=./d*,destination=/c"
        assert mount_desc_to_mount_args(
            {"type": "image", "source": "img", "target": "/c", "image": {"subpath": "x"}}
        ) == "type=image,source=img,destination=/c,subpath=x"

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mount type"):
            mount_desc_to_mount_args({"type": "bogus", "target": "/c"})

    def test_volume_args_unknown_type_and_missing_source(self) -> None:
        with pytest.raises(ValueError, match="unknown mount type"):
            mount_desc_to_volume_args({"type": "tmpfs", "target": "/t"}, "web")
        with pytest.raises(ValueError, match="missing mount source"):
            mount_desc_to_volume_args({"type": "volume", "source": None, "target": "/t"}, "web")

    def test_volume_args_propagation_merge_and_rw_selinux(self) -> None:
        s = mount_desc_to_volume_args(
            {
                "type": "volume",
                "source": "data",
                "target": "/d",
                "read_only": False,
                "volume": {"propagation": "shared"},
                "bind": {"propagation": "Z", "selinux": "z"},
            },
            "web",
        )
        # 非 bind：volume 与 bind.propagation 合并；selinux 仅 bind 类型生效
        assert s.startswith("data:/d:")
        opts = set(s.split(":", 2)[2].split(","))
        assert opts == {"rw", "shared", "Z"}

    def test_volume_args_bind_selinux_appended(self) -> None:
        # propagation 取 bind.propagation（shared），与 read_only/selinux 拼接
        s = mount_desc_to_volume_args(
            {
                "type": "bind",
                "source": "/h",
                "target": "/c",
                "read_only": True,
                "bind": {"propagation": "shared", "selinux": "Z"},
            },
            "web",
        )
        assert s == "/h:/c:shared,ro,Z"


class TestGetMntDictAndGetMountArgs:
    def test_get_mnt_dict_string_and_dict_inputs(self) -> None:
        compose = mock.Mock()
        compose.dirname = "/base"
        compose.project_name = "proj"
        compose.vols = {}
        mnt = get_mnt_dict(compose, {"_service": "web"}, "named:/data")
        assert mnt["type"] == "volume"
        assert mnt["source"] == "named"

    def test_prefer_volume_uses_dash_v_for_bind(self) -> None:
        compose = mock.Mock()
        compose.prefer_volume_over_mount = True
        compose.dirname = "/base"
        compose.project_name = "proj"
        compose.vols = {}
        compose.podman.output = mock.AsyncMock()
        args = asyncio.run(
            get_mount_args(
                compose, {"_service": "web"}, {"type": "bind", "source": "/h", "target": "/c"}
            )
        )
        assert args == ["-v", "/h:/c"]

    def test_prefer_volume_tmpfs_form(self) -> None:
        compose = mock.Mock()
        compose.prefer_volume_over_mount = True
        compose.podman.output = mock.AsyncMock()
        mnt = {"type": "tmpfs", "target": "/t", "tmpfs": {"size": "100m", "mode": "1777"}}
        args = asyncio.run(get_mount_args(compose, {"_service": "web"}, mnt))
        assert args == ["--tmpfs", "/t:size=100m,mode=1777"]

    def test_default_uses_mount_form(self) -> None:
        compose = mock.Mock()
        compose.prefer_volume_over_mount = False
        compose.podman.output = mock.AsyncMock()
        mnt = {"type": "tmpfs", "target": "/t"}
        args = asyncio.run(get_mount_args(compose, {"_service": "web"}, mnt))
        assert args == ["--mount", "type=tmpfs,destination=/t"]
