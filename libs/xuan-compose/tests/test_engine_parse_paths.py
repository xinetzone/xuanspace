# SPDX-License-Identifier: GPL-2.0-only
"""``ComposeEngine._parse_compose_file`` 的错误/边界/命名分支补测。

T6 的 test_parse_compose_file_build/test_include/test_can_merge_build
已覆盖多文件合并、include 成功与四类 include 错误、build.context
规范化主路径；本文件只补未覆盖的真实分支：文件缺失退出码、env 文件
加载与 PODMAN_ 变量导出、COMPOSE_PROJECT_DIR、顶层非对象、项目名
归一失败、网络/卷缺失与默认网络选择、scale/replicas 副本命名、
container_name、--no-normalize、args.env 与 extends.file 绝对化。
"""

import os

import pytest
import yaml

from xuan_compose.cli.parser import parse_args
from xuan_compose.engine import ComposeEngine

_ENV_KEYS = (
    "COMPOSE_PROJECT_DIR",
    "COMPOSE_FILE",
    "COMPOSE_PATH_SEPARATOR",
    "COMPOSE_PROJECT_NAME",
    "COMPOSE_PROFILES",
    "COMPOSE_ENV_FILES",
    "COMPOSE_PARALLEL_LIMIT",
    "PODMAN_FOO",
)


@pytest.fixture(autouse=True)
def _isolated_cwd_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield tmp_path, monkeypatch


def _write(path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _parse(argv: list[str]) -> ComposeEngine:
    engine = ComposeEngine()
    parse_args(engine, argv)
    engine._parse_compose_file()
    return engine


class TestFileDiscoveryErrors:
    def test_no_compose_file_exits_minus_one(self) -> None:
        with pytest.raises(SystemExit) as ctx:
            _parse(["ps"])
        assert ctx.value.code == -1

    def test_explicit_missing_file_exits_1(self) -> None:
        with pytest.raises(SystemExit) as ctx:
            _parse(["-f", "nope.yaml", "ps"])
        assert ctx.value.code == 1

    def test_missing_env_file_exits_1(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(tmp_path / "compose.yaml", "services: {}\n")
        with pytest.raises(SystemExit) as ctx:
            _parse(["--env-file", "missing.env", "-f", "compose.yaml", "ps"])
        assert ctx.value.code == 1

    def test_non_object_top_level_exits_1(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(tmp_path / "compose.yaml", "- a\n- b\n")
        with pytest.raises(SystemExit) as ctx:
            _parse(["-f", "compose.yaml", "ps"])
        assert ctx.value.code == 1


class TestEnvLoading:
    def test_env_file_exports_podman_vars_and_activates_profiles(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n"
            "  web:\n    image: img\n"
            "  db:\n    image: db\n    profiles: [p1]\n",
        )
        _write(tmp_path / "cli.env", "PODMAN_FOO=bar\nCOMPOSE_PROFILES=p1\n")
        engine = _parse(["--env-file", "cli.env", "-f", "compose.yaml", "ps"])
        assert os.environ["PODMAN_FOO"] == "bar"
        assert set(engine.services) == {"web", "db"}

    def test_compose_project_dir_changes_search_root(self, _isolated_cwd_env) -> None:
        tmp_path, monkeypatch = _isolated_cwd_env
        work = tmp_path / "work"
        work.mkdir()
        _write(work / "compose.yaml", "name: fromdir\nservices: {}\n")
        monkeypatch.setenv("COMPOSE_PROJECT_DIR", str(work))
        engine = _parse(["ps"])
        assert engine.dirname == os.path.realpath(str(work))
        assert engine.project_name == "fromdir"


class TestProjectName:
    def test_normalized_empty_name_raises(self, _isolated_cwd_env) -> None:
        tmp_path, monkeypatch = _isolated_cwd_env
        _write(tmp_path / "compose.yaml", "services:\n  web:\n    image: img\n")
        monkeypatch.setenv("COMPOSE_PROJECT_NAME", "!!!")
        with pytest.raises(RuntimeError, match="normalized to empty"):
            _parse(["-f", "compose.yaml", "ps"])

    def test_obsolete_version_attribute_ignored(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            'version: "3"\nname: proj\nservices:\n  web:\n    image: img\n',
        )
        engine = _parse(["-f", "compose.yaml", "ps"])
        # 顶层 version 属性在解析末期被弹出（501 行），仅告警不报错
        assert "version" not in yaml.safe_load(engine.merged_yaml)


class TestNetworks:
    def test_missing_network_raises(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n  web:\n    image: img\n    networks: [missing-net]\n",
        )
        with pytest.raises(RuntimeError, match="missing networks: missing-net"):
            _parse(["-f", "compose.yaml", "ps"])

    def test_unused_network_warns_but_passes(self, _isolated_cwd_env, caplog) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n"
            "  web:\n    image: img\n    networks: [used]\n"
            "networks:\n  used: null\n  spare: null\n",
        )
        with caplog.at_level("WARNING"):
            engine = _parse(["-f", "compose.yaml", "ps"])
        assert "unused networks: spare" in caplog.text
        assert set(engine.networks) == {"used", "spare"}

    def test_single_network_becomes_default(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n  web:\n    image: img\n"
            "networks:\n  solo: null\n",
        )
        engine = _parse(["-f", "compose.yaml", "ps"])
        assert engine.default_net == "solo"

    def test_multiple_networks_without_default_yields_none(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n  web:\n    image: img\n"
            "networks:\n  a: null\n  b: null\n",
        )
        engine = _parse(["-f", "compose.yaml", "ps"])
        assert engine.default_net is None

    def test_default_named_network_selected(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n  web:\n    image: img\n"
            "networks:\n  default: null\n  extra: null\n",
        )
        engine = _parse(["-f", "compose.yaml", "ps"])
        assert engine.default_net == "default"


class TestVolumesAndNaming:
    def test_undefined_named_volume_raises(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n  web:\n    image: img\n    volumes: [\"named:/data\"]\n",
        )
        with pytest.raises(RuntimeError, match=r"volume \[named\] not defined"):
            _parse(["-f", "compose.yaml", "ps"])

    def test_cli_scale_creates_three_container_names(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(tmp_path / "compose.yaml", "services:\n  web:\n    image: img\n")
        engine = _parse(["-p", "proj", "-f", "compose.yaml", "up", "--scale", "web=3"])
        assert engine.container_names_by_service["web"] == [
            "proj_web_1",
            "proj_web_2",
            "proj_web_3",
        ]

    def test_compose_scale_and_deploy_replicas(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n"
            "  web:\n    image: img\n    scale: 2\n"
            "  worker:\n    image: w\n    deploy:\n      mode: replicated\n      replicas: 3\n",
        )
        engine = _parse(["-p", "proj", "-f", "compose.yaml", "ps"])
        assert len(engine.container_names_by_service["web"]) == 2
        assert len(engine.container_names_by_service["worker"]) == 3

    def test_container_name_sets_name_and_prefix(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n  web:\n    image: img\n    container_name: myweb\n",
        )
        engine = _parse(["-p", "proj", "-f", "compose.yaml", "ps"])
        cnt = engine.container_by_name["myweb"]
        assert cnt["log_prefix"] == "myweb"
        assert cnt["num"] == 1

    def test_replica_second_ignores_container_name(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(
            tmp_path / "compose.yaml",
            "services:\n  web:\n    image: img\n    container_name: myweb\n    scale: 2\n",
        )
        engine = _parse(["-p", "proj", "-f", "compose.yaml", "ps"])
        # 副本 1 用自定义名，副本 2 起回退到编号名（上游规则）
        assert engine.container_names_by_service["web"] == ["myweb", "proj_web_2"]


class TestNoNormalizeAndEnvAndExtends:
    def test_no_normalize_keeps_relative_build_context(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(tmp_path / "compose.yaml", "services:\n  web:\n    build: ./ctx\n")
        engine = _parse(["-f", "compose.yaml", "config", "--no-normalize"])
        # normalize（加载期）已把字符串 build 归一成 mapping，但 final 阶段
        # 的 normpath(project_dir) 拼接被 --no-normalize 跳过
        assert engine.services["web"]["build"]["context"] == "./ctx"

    def test_run_env_merged_into_engine_environ(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        _write(tmp_path / "compose.yaml", "services:\n  web:\n    image: img\n")
        engine = _parse(["-f", "compose.yaml", "run", "--env", "FOO=bar", "web"])
        assert engine.environ["FOO"] == "bar"

    def test_extends_file_rewritten_against_compose_file_dir(self, _isolated_cwd_env) -> None:
        tmp_path, _ = _isolated_cwd_env
        proj = tmp_path / "proj"
        proj.mkdir()
        _write(
            proj / "compose.yaml",
            "services:\n"
            "  web:\n    extends:\n      file: base.yaml\n      service: base\n",
        )
        _write(proj / "base.yaml", "services:\n  base:\n    image: base:1\n")
        # CWD 在父目录、以相对 -f 指向子目录里的 compose；若无 extends.file
        # 绝对化（上游第 451-456 行），resolve_extends 将打不开 base.yaml
        engine = _parse(["-f", "proj/compose.yaml", "ps"])
        assert engine.services["web"]["image"] == "base:1"
