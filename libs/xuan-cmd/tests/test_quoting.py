"""参数引用与命令拼接的行为。"""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from xuan_cmd import format_command, quote_arg


def test_quote_arg_plain_token_is_unchanged():
    assert quote_arg("-n") == "-n"


def test_quote_arg_quotes_space_by_platform():
    expected = '"a b"' if os.name == "nt" else "'a b'"
    assert quote_arg("a b") == expected


def test_quote_arg_accepts_path():
    expected = '"a b"' if os.name == "nt" else "'a b'"
    assert quote_arg(Path("a b")) == expected


def test_quote_arg_roundtrip_with_shlex():
    token = "it's a path"
    assert shlex.split(quote_arg(token)) == [token]


def test_quote_arg_windows_uses_list2cmdline(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert quote_arg("a b") == '"a b"'


def test_format_command_joins_parts():
    command = format_command(["cmake", "-S", Path("src dir"), "-B", "build"])
    assert shlex.split(command) == ["cmake", "-S", "src dir", "-B", "build"]


def test_format_command_empty():
    assert format_command([]) == ""


@pytest.mark.parametrize("token", ["--define", "A=B", "", "a'b"])
def test_format_command_single_token(token):
    """空串也被保留为单个空参数，不会被 shell 规则吞掉。"""
    command = format_command([token])
    if os.name == "nt":
        assert command == subprocess.list2cmdline([token])
    else:
        assert shlex.split(command) == [token]
