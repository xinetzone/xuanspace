# SPDX-License-Identifier: GPL-2.0-only
"""translate.secrets 分支补测（移植的 25 例覆盖 run/build 主路径；
这里补 build file secret 三 target 分支、run file secret 的目标/relabel
分支、uid 警告、external 透传选项与同名校验、unparsable 兜底——
上游第 838-954 行真实分支）。"""

import os
from unittest import mock

import pytest

from xuan_compose.translate.secrets import get_secret_args


def _compose(declared: dict, project_name: str = "proj") -> mock.Mock:
    c = mock.Mock()
    c.declared_secrets = declared
    c.project_name = project_name
    c.dirname = "/base"
    return c


class TestUndeclaredAndEnvironment:
    def test_undeclared_secret_raises(self) -> None:
        c = _compose({})
        with pytest.raises(ValueError, match="undeclared secret"):
            get_secret_args(c, {"_service": "web"}, "ghost")

    def test_environment_secret_at_build_time_with_target(self) -> None:
        c = _compose({"token": {"environment": "TOKEN_VAR"}})
        out = get_secret_args(
            c, {"_service": "web"}, {"source": "token", "target": "build-token"},
            podman_is_building=True,
        )
        assert out == ["--secret", "id=build-token,env=TOKEN_VAR"]

    def test_environment_secret_run_mount(self) -> None:
        c = _compose({"token": {"environment": "TOKEN_VAR"}})
        out = get_secret_args(c, {"_service": "web"}, "token")
        assert out == ["--secret", "proj_token"]


class TestBuildFileSecrets:
    def test_no_target_uses_secret_name_as_id(self) -> None:
        c = _compose({"cfg": {"file": "./secrets/cfg.txt"}})
        out = get_secret_args(c, {"_service": "web"}, "cfg", podman_is_building=True)
        src = os.path.realpath("/base/secrets/cfg.txt")
        assert out == ["--secret", f"id=cfg,src={src}"]

    def test_plain_target_becomes_id(self) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt"}})
        out = get_secret_args(
            c, {"_service": "web"}, {"source": "cfg", "target": "renamed.txt"},
            podman_is_building=True,
        )
        assert out[1].startswith("id=renamed.txt,src=")

    def test_target_with_slash_rejected(self) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt"}})
        with pytest.raises(ValueError, match="invalid target"):
            get_secret_args(
                c, {"_service": "web"}, {"source": "cfg", "target": "dir/cfg.txt"},
                podman_is_building=True,
            )


class TestRunFileSecrets:
    def test_default_target_under_run_secrets(self) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt"}})
        out = get_secret_args(c, {"_service": "web"}, "cfg")
        src = os.path.realpath("/base/cfg.txt")
        assert out == ["--volume", f"{src}:/run/secrets/cfg:ro,rprivate,rbind"]

    def test_relative_target(self) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt"}})
        out = get_secret_args(
            c, {"_service": "web"}, {"source": "cfg", "target": "my.cfg"}
        )
        assert out[1].endswith(":/run/secrets/my.cfg:ro,rprivate,rbind")

    def test_absolute_target_used_verbatim(self) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt"}})
        out = get_secret_args(
            c, {"_service": "web"}, {"source": "cfg", "target": "/etc/my.cfg"}
        )
        assert out[1].endswith(":/etc/my.cfg:ro,rprivate,rbind")

    @pytest.mark.parametrize("relabel,expected", [("z", ",z"), ("Z", ",Z")])
    def test_selinux_relabel_suffix(self, relabel, expected) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt", "x-podman.relabel": relabel}})
        out = get_secret_args(c, {"_service": "web"}, "cfg")
        assert out[1].endswith(f"rprivate,rbind{expected}")

    def test_invalid_relabel_raises(self) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt", "x-podman.relabel": "bad"}})
        with pytest.raises(ValueError, match="invalid .relabel. option"):
            get_secret_args(c, {"_service": "web"}, "cfg")

    def test_uid_gid_mode_emits_warning_but_mounts(self, caplog) -> None:
        c = _compose({"cfg": {"file": "./cfg.txt"}})
        with caplog.at_level("WARNING"):
            out = get_secret_args(
                c, {"_service": "web"},
                {"source": "cfg", "uid": "1000", "gid": "1000", "mode": 0o400},
            )
        assert "not supported" in caplog.text
        assert out[0] == "--volume"


class TestExternalSecrets:
    def test_external_with_all_options(self) -> None:
        c = _compose({"ext": {"external": True}})
        out = get_secret_args(
            c, {"_service": "web"},
            {"source": "ext", "uid": "1000", "gid": "1000", "mode": "0400",
             "type": "a", "target": "/t"},
        )
        assert out == [
            "--secret",
            "ext,uid=1000,gid=1000,mode=0400,type=a,target=/t",
        ]

    def test_named_secret_without_options(self) -> None:
        c = _compose({"ext": {"name": "ext"}})
        assert get_secret_args(c, {"_service": "web"}, "ext") == ["--secret", "ext"]

    def test_custom_name_mismatch_raises(self) -> None:
        c = _compose({"ext": {"name": "real-ext"}})
        with pytest.raises(ValueError, match="not supported"):
            get_secret_args(c, {"_service": "web"}, "ext")

    def test_unparsable_secret_raises(self) -> None:
        # 声明存在但既无 environment/file/external/name
        c = _compose({"broken": {}})
        with pytest.raises(ValueError, match="unparsable secret"):
            get_secret_args(c, {"_service": "web"}, "broken")
