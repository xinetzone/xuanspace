# SPDX-License-Identifier: GPL-2.0-only
"""逐用例移植自上游 tests/unit/test_compose_run_log_format.py（11 例，断言零削弱）。

适配点：``podman_compose.Podman`` → ``xuan_compose.runner.Podman``。
``_format_stream`` 生产实现随 T4 运行器层提前落地，本文件在 T5 随翻译层
测试集一并交付（上游归属 tests/unit，属 T5 测试清单 14 文件之一）。
类型注解现代化：``Union[...]`` → ``... | None``（T10 差异登记）。
"""

# pylint: disable=protected-access

import io
import unittest

from xuan_compose.runner import Podman


class DummyReader:
    def __init__(self, data: list[bytes] | None = None):
        self.data = data or []

    async def readuntil(self, _: str) -> bytes:
        return self.data.pop(0)

    def at_eof(self) -> bool:
        return len(self.data) == 0


class TestComposeRunLogFormat(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.p = get_minimal_podman()
        self.buffer = io.StringIO()
        self.unicode_sample = b"\xe3\x81\x82\xe3\x81\x84\n"  # result of 'あい\n'.encode('utf-8')

    async def test_single_line_single_chunk(self) -> None:
        reader = DummyReader([b"hello, world\n"])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: hello, world\n")

    async def test_empty(self) -> None:
        reader = DummyReader([])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "")

    async def test_empty2(self) -> None:
        reader = DummyReader([b""])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "")

    async def test_empty_line(self) -> None:
        reader = DummyReader([b"\n"])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: \n")

    async def test_line_split(self) -> None:
        reader = DummyReader([b"hello,", b" world\n"])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: hello, world\n")

    async def test_two_lines_in_one_chunk(self) -> None:
        reader = DummyReader([b"hello\nbye\n"])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: hello\nLL: bye\n")

    async def test_double_blank(self) -> None:
        reader = DummyReader([b"hello\n\n\nbye\n"])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: hello\nLL: \nLL: \nLL: bye\n")

    async def test_no_new_line_at_end(self) -> None:
        reader = DummyReader([b"hello\nbye"])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: hello\nLL: bye\n")

    async def test_split_multibyte(self) -> None:
        string = self.unicode_sample
        mid = 4
        reader = DummyReader([string[:mid], string[mid:]])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: あい\n")

    async def test_incomplete_multibyte_at_end(self) -> None:
        string = self.unicode_sample
        mid = 4
        reader = DummyReader([string[:mid]])
        await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]
        self.assertEqual(self.buffer.getvalue(), "LL: あ\n")

    async def test_incomplete_multibyte_at_beginning(self) -> None:
        string = self.unicode_sample
        mid = 4
        reader = DummyReader([string[mid:]])
        with self.assertRaises(UnicodeDecodeError):
            await self.p._format_stream(reader, self.buffer, "LL:")  # type: ignore[arg-type]


def get_minimal_podman() -> Podman:
    return Podman(None)  # type: ignore[arg-type]
