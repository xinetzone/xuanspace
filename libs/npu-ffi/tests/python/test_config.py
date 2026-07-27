"""Tests for VTA hardware configuration module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from npu_ffi.vta.config import (
    VTAConfig,
    DEFAULT_CONFIGS,
    get_default_config,
    validate_config,
)


class TestVTAConfigCreation:
    """Test VTAConfig dataclass creation and defaults."""

    def test_default_config_creation(self):
        config = get_default_config("vta")
        assert config.target == "pynq"
        assert config.hw_version == "v1.0.0"
        assert config.log_bus_width == 3
        assert config.log_inp_width == 3
        assert config.log_wgt_width == 3
        assert config.log_acc_width == 5
        assert config.log_out_width == 3
        assert config.log_batch == 0
        assert config.log_block_in == 4
        assert config.log_block_out == 4
        assert config.log_uop_buff_size == 11
        assert config.log_inp_buff_size == 15
        assert config.log_wgt_buff_size == 15
        assert config.log_acc_buff_size == 13
        assert config.ins_width == 7
        assert config.uop_width == 5

    def test_all_default_configs_exist(self):
        assert "vta" in DEFAULT_CONFIGS
        assert "vta_v3" in DEFAULT_CONFIGS
        assert "vta_v4" in DEFAULT_CONFIGS

    def test_v3_config(self):
        config = get_default_config("vta_v3")
        assert config.target == "ultra96"
        assert config.hw_version == "v3.0.0"
        assert config.log_bus_width == 4
        assert config.log_inp_buff_size == 16

    def test_v4_config(self):
        config = get_default_config("vta_v4")
        assert config.target == "zcu104"
        assert config.hw_version == "v4.0.0"
        assert config.log_bus_width == 5
        assert config.log_block_in == 5
        assert config.log_block_out == 5
        assert config.log_batch == 1

    def test_unknown_config_raises(self):
        with pytest.raises(KeyError):
            get_default_config("nonexistent")

    def test_custom_config_creation(self):
        config = VTAConfig(
            target="custom",
            hw_version="v0.1.0",
            log_bus_width=4,
        )
        assert config.target == "custom"
        assert config.log_bus_width == 4
        assert config.log_block_in == 4

    def test_config_is_frozen(self):
        config = get_default_config()
        with pytest.raises(Exception):
            config.log_bus_width = 5


class TestVTAConfigProperties:
    """Test computed property accessors."""

    def test_bus_width(self):
        config = get_default_config()
        assert config.bus_width == 8

    def test_data_widths(self):
        config = get_default_config()
        assert config.inp_width == 8
        assert config.wgt_width == 8
        assert config.acc_width == 32
        assert config.out_width == 8

    def test_block_dimensions(self):
        config = get_default_config()
        assert config.batch == 1
        assert config.block_in == 16
        assert config.block_out == 16

    def test_buffer_sizes(self):
        config = get_default_config()
        assert config.uop_buff_size == 2048
        assert config.inp_buff_size == 32768
        assert config.wgt_buff_size == 32768
        assert config.acc_buff_size == 8192

    def test_instruction_widths(self):
        config = get_default_config()
        assert config.ins_width_bits == 128
        assert config.uop_width_bits == 32

    def test_acc_buff_bytes(self):
        config = get_default_config()
        assert config.acc_buff_bytes == 8192 * 4


class TestVTAConfigValidation:
    """Test configuration validation."""

    def test_valid_default_configs(self):
        for name in DEFAULT_CONFIGS:
            config = get_default_config(name)
            validate_config(config)

    def test_invalid_bus_width(self):
        with pytest.raises(ValueError, match="log_bus_width"):
            VTAConfig(log_bus_width=100)

    def test_invalid_acc_width_too_narrow(self):
        with pytest.raises(ValueError, match="Accumulator width"):
            VTAConfig(log_acc_width=3, log_block_in=4, log_batch=0, log_inp_width=3, log_wgt_width=3)

    def test_replace_creates_new_config(self):
        config = get_default_config()
        new_config = config.replace(target="custom_target", log_bus_width=4)
        assert new_config.target == "custom_target"
        assert new_config.log_bus_width == 4
        assert config.target == "pynq"
        assert config.log_bus_width == 3


class TestVTAConfigSerialization:
    """Test dict-based serialization."""

    def test_to_dict(self):
        config = get_default_config()
        d = config.to_dict()
        assert d["target"] == "pynq"
        assert d["log_bus_width"] == 3
        assert len(d) == 16

    def test_from_dict(self):
        d = {"target": "test", "log_bus_width": 4, "extra_key": "ignored"}
        config = VTAConfig.from_dict(d)
        assert config.target == "test"
        assert config.log_bus_width == 4


class TestProtobufIO:
    """Test protobuf serialization/deserialization roundtrip."""

    @pytest.fixture
    def require_protobuf(self):
        from npu_ffi.vta import proto_io
        if not proto_io.is_protobuf_available():
            pytest.skip("protobuf not available")

    def test_protobuf_available_flag(self, require_protobuf):
        from npu_ffi.vta import proto_io
        assert proto_io.is_protobuf_available()

    def test_binary_roundtrip(self, require_protobuf):
        from npu_ffi.vta import proto_io
        config = get_default_config()
        data = proto_io.serialize(config)
        assert isinstance(data, bytes)
        assert len(data) > 0

        loaded = proto_io.deserialize(data)
        assert loaded.target == config.target
        assert loaded.hw_version == config.hw_version
        assert loaded.log_bus_width == config.log_bus_width
        assert loaded.log_block_in == config.log_block_in
        assert loaded.log_acc_width == config.log_acc_width
        assert loaded.ins_width == config.ins_width
        assert loaded.uop_width == config.uop_width

    def test_save_load_binary_file(self, require_protobuf, tmp_path):
        from npu_ffi.vta import proto_io
        config = get_default_config("vta_v3")
        path = tmp_path / "config.bin"

        proto_io.save_config(config, path)
        assert path.exists()

        loaded = proto_io.load_config(path)
        assert loaded.target == "ultra96"
        assert loaded.log_bus_width == 4

    def test_json_roundtrip(self, require_protobuf, tmp_path):
        from npu_ffi.vta import proto_io
        config = get_default_config()
        path = tmp_path / "config.json"

        proto_io.save_json(config, path)
        assert path.exists()
        content = path.read_text()
        assert "target" in content

        loaded = proto_io.load_json(path)
        assert loaded.target == config.target
        assert loaded.log_block_in == config.log_block_in

    def test_text_roundtrip(self, require_protobuf, tmp_path):
        from npu_ffi.vta import proto_io
        config = get_default_config("vta_v4")
        path = tmp_path / "config.pbtxt"

        proto_io.save_text(config, path)
        assert path.exists()

        loaded = proto_io.load_text(path)
        assert loaded.target == "zcu104"
        assert loaded.log_block_out == 5

    def test_config_to_proto_and_back(self, require_protobuf):
        from npu_ffi.vta import proto_io
        config = get_default_config()
        pb = proto_io.config_to_proto(config)
        assert pb.target == "pynq"
        assert pb.log_block_in == 4
        assert pb.log_bus_width == 3

        back = proto_io.proto_to_config(pb)
        assert back == config
