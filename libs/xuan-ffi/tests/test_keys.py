"""能力键语法：``xuan:ffi/<name>#api``。"""

import pytest
from xuan_plugin.manifest import parse_manifest

from xuan_ffi import FFIKeyError, ffi_key, ffi_name


def test_key_roundtrip() -> None:
    key = ffi_key("caffe")
    assert key == "xuan:ffi/caffe#api"
    assert ffi_name(key) == "caffe"


def test_key_allows_digits_and_separators() -> None:
    assert ffi_key("npu2-vta_pre") == "xuan:ffi/npu2-vta_pre#api"
    assert ffi_name("xuan:ffi/npu2-vta_pre#api") == "npu2-vta_pre"


@pytest.mark.parametrize("name", ["", "Caffe", "1caffe", "caf fe", "caffe/extra", "caffe#api"])
def test_invalid_names_are_rejected(name: str) -> None:
    with pytest.raises(FFIKeyError):
        ffi_key(name)


@pytest.mark.parametrize(
    "key",
    ["", "xuan:build/backend#dispatch", "xuan:ffi/caffe", "xuan:ffi/#api", "xuan:ffi/Bad#api", None],
)
def test_non_ffi_keys_return_none(key: object) -> None:
    assert ffi_name(key) is None  # type: ignore[arg-type]


def test_key_passes_manifest_validation() -> None:
    """能力键必须能被 xuan-plugin 的 K2 校验器接受，避免出现第三套语法。"""
    manifest = parse_manifest(
        {"id": "demo.ffi", "version": "0.1.0", "provide": [{"key": ffi_key("caffe")}]},
    )
    assert manifest.provide_keys() == ("xuan:ffi/caffe#api",)
