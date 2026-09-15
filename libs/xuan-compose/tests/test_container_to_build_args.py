# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_container_to_build_args.py（20 例，断言零削弱）。

适配点：
- ``podman_compose.container_to_build_args`` →
  ``xuan_compose.translate.build.container_to_build_args``。
- 本文件保留上游本地 create_compose_mock（不与 test_container_to_args 共享），
  其中 ``dirname="test_dirname"`` 为相对路径：build ssh 路径拼接只做
  ``os.path.join`` 不做 realpath，期望保持上游原样（如
  ``id1=test_dirname/id1/test1``），故不可改为绝对夹具根。
"""

import os
import unittest
from typing import Any
from unittest import mock

from xuan_compose.translate.build import container_to_build_args


def create_compose_mock(project_name: str = "test_project_name") -> Any:
    compose = mock.Mock()
    compose.project_name = project_name
    compose.dirname = "test_dirname"
    compose.container_names_by_service.get = mock.Mock(return_value=None)
    compose.prefer_volume_over_mount = False
    compose.default_net = None
    compose.networks = {}
    compose.x_podman = {}
    return compose


def get_minimal_container() -> dict[str, Any]:
    return {
        "name": "project_name_service_name1",
        "service_name": "service_name",
        "image": "new-image",
        "build": {},
    }


def get_minimal_args() -> Any:
    args = mock.Mock()
    args.build_arg = []
    args.pull = None
    return args


class TestContainerToBuildArgs(unittest.TestCase):
    def test_minimal(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--no-cache",
                ".",
            ],
        )

    def test_platform(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["platform"] = "linux/amd64"
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--platform",
                "linux/amd64",
                "--no-cache",
                ".",
            ],
        )

    def test_tags(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["tags"] = ["some-tag1", "some-tag2:2"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "-t",
                "some-tag1",
                "-t",
                "some-tag2:2",
                "--no-cache",
                ".",
            ],
        )

    def test_labels(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["labels"] = ["some-label1", "some-label2.2"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--label",
                "some-label1",
                "--label",
                "some-label2.2",
                "--no-cache",
                ".",
            ],
        )

    def test_caches(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["cache_from"] = ["registry/image1", "registry/image2"]
        cnt["build"]["cache_to"] = ["registry/image3", "registry/image4"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--no-cache",
                "--cache-from",
                "registry/image1",
                "--cache-from",
                "registry/image2",
                "--cache-to",
                "registry/image3",
                "--cache-to",
                "registry/image4",
                ".",
            ],
        )

    def test_dockerfile_inline(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["dockerfile_inline"] = "FROM busybox\nRUN echo 'hello world'"
        args = get_minimal_args()

        cleanup_callbacks = []
        args = container_to_build_args(
            c, cnt, args, lambda path: True, cleanup_callbacks=cleanup_callbacks
        )

        temp_dockerfile = args[args.index("-f") + 1]
        self.assertTrue(os.path.exists(temp_dockerfile))

        with open(temp_dockerfile) as file:
            contents = file.read()
            self.assertEqual(contents, "FROM busybox\n" + "RUN echo 'hello world'")

        for cb in cleanup_callbacks:
            cb()
        self.assertFalse(os.path.exists(temp_dockerfile))

    def test_context_git_url(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["context"] = "https://github.com/test_repo.git"
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: False)
        self.assertEqual(
            args,
            [
                "-t",
                "new-image",
                "--no-cache",
                "https://github.com/test_repo.git",
            ],
        )

    def test_context_git_url_with_dockerfile(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["context"] = "https://github.com/test_repo.git"
        cnt["build"]["dockerfile"] = "Dockerfile.custom"
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: False)
        self.assertEqual(
            args,
            [
                "-f",
                "Dockerfile.custom",
                "-t",
                "new-image",
                "--no-cache",
                "https://github.com/test_repo.git",
            ],
        )

    def test_context_invalid_git_url_git_is_not_prefix(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["context"] = "not_prefix://github.com/test_repo"
        args = get_minimal_args()

        with self.assertRaises(OSError):
            container_to_build_args(c, cnt, args, lambda path: False)

    def test_build_ssh_absolute_path(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["ssh"] = ["id1=/test1"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--ssh",
                "id1=/test1",
                "--no-cache",
                ".",
            ],
        )

    def test_build_ssh_relative_path(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["ssh"] = ["id1=id1/test1"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--ssh",
                "id1=test_dirname/id1/test1",
                "--no-cache",
                ".",
            ],
        )

    def test_build_ssh_working_dir(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["ssh"] = ["id1=./test1"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--ssh",
                "id1=test_dirname/./test1",
                "--no-cache",
                ".",
            ],
        )

    @mock.patch.dict(os.environ, {"HOME": "/home/user"}, clear=True)
    def test_build_ssh_path_home_dir(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["ssh"] = ["id1=~/test1"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--ssh",
                "id1=/home/user/test1",
                "--no-cache",
                ".",
            ],
        )

    def test_build_ssh_map(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["ssh"] = {"id1": "test1", "id2": "test2"}
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--ssh",
                "id1=test_dirname/test1",
                "--ssh",
                "id2=test_dirname/test2",
                "--no-cache",
                ".",
            ],
        )

    def test_build_ssh_array(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["ssh"] = ["id1=test1", "id2=test2"]
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--ssh",
                "id1=test_dirname/test1",
                "--ssh",
                "id2=test_dirname/test2",
                "--no-cache",
                ".",
            ],
        )

    def test_pull_always(self) -> None:
        c = create_compose_mock()
        cnt = get_minimal_container()
        args = get_minimal_args()
        args.pull = "always"

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--no-cache",
                "--pull=always",
                ".",
            ],
        )

    def test_containerfile_in_context(self) -> None:
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["build"]["context"] = "./subdir"
        args = get_minimal_args()
        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "subdir/Containerfile",
                "-t",
                "new-image",
                "--no-cache",
                "./subdir",
            ],
        )

    def test_build_environment_secret_no_target(self) -> None:
        c = create_compose_mock()
        c.declared_secrets = {"my_secret": {"environment": "MY_VAR"}}
        cnt = get_minimal_container()
        cnt["build"]["secrets"] = ["my_secret"]
        args = get_minimal_args()
        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--secret",
                "id=my_secret,env=MY_VAR",
                "--no-cache",
                ".",
            ],
        )

    def test_build_environment_secret_with_target(self) -> None:
        c = create_compose_mock()
        c.declared_secrets = {"my_secret": {"environment": "MY_VAR"}}
        cnt = get_minimal_container()
        cnt["build"]["secrets"] = [{"source": "my_secret", "target": "custom_id"}]
        args = get_minimal_args()
        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--secret",
                "id=custom_id,env=MY_VAR",
                "--no-cache",
                ".",
            ],
        )

    def test_pass_no_ipc_to_build(self) -> None:
        """Do not pass --ipc to podman build"""
        c = create_compose_mock()

        cnt = get_minimal_container()
        cnt["ipc"] = "host"
        args = get_minimal_args()

        args = container_to_build_args(c, cnt, args, lambda path: True)
        self.assertEqual(
            args,
            [
                "-f",
                "Containerfile",
                "-t",
                "new-image",
                "--no-cache",
                ".",
            ],
        )
