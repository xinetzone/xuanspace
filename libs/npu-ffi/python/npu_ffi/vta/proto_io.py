"""Protobuf serialization/deserialization for VTAConfig.

This module provides optional protobuf support for VTAConfig.
If protobuf (google.protobuf) is not installed, serialization functions
will raise ImportError with a helpful message.

Usage:
    from npu_ffi.vta.config import get_default_config
    from npu_ffi.vta.proto_io import save_config, load_config

    config = get_default_config()
    save_config(config, "vta_config.bin")
    loaded = load_config("vta_config.bin")
"""

import os
from pathlib import Path
from typing import Union

from .config import VTAConfig

try:
    from google.protobuf import message as _message
    from . import vta_config_pb2 as _pb2

    _PROTOBUF_AVAILABLE = True
    _ProtobufVTAConfig = _pb2.VTAConfig
    _ProtobufDefaultConfigs = _pb2.DefaultConfigs
except ImportError:
    _PROTOBUF_AVAILABLE = False
    _pb2 = None
    _ProtobufVTAConfig = None
    _ProtobufDefaultConfigs = None


def is_protobuf_available() -> bool:
    """Check if protobuf library is available for use."""
    return _PROTOBUF_AVAILABLE


def _require_protobuf() -> None:
    """Raise ImportError if protobuf is not available."""
    if not _PROTOBUF_AVAILABLE:
        raise ImportError(
            "protobuf serialization requires google.protobuf package. "
            "Install it with: pip install 'protobuf>=7.0.0'"
        )


def config_to_proto(config: VTAConfig) -> "_ProtobufVTAConfig":
    """Convert a pure Python VTAConfig to a protobuf message.

    Args:
        config: VTAConfig dataclass instance.

    Returns:
        Protobuf VTAConfig message.
    """
    _require_protobuf()
    pb = _pb2.VTAConfig()
    pb.target = config.target
    pb.hw_version = config.hw_version
    pb.log_bus_width = config.log_bus_width
    pb.log_inp_width = config.log_inp_width
    pb.log_wgt_width = config.log_wgt_width
    pb.log_acc_width = config.log_acc_width
    pb.log_out_width = config.log_out_width
    pb.log_batch = config.log_batch
    pb.log_block_in = config.log_block_in
    pb.log_block_out = config.log_block_out
    pb.log_uop_buff_size = config.log_uop_buff_size
    pb.log_inp_buff_size = config.log_inp_buff_size
    pb.log_wgt_buff_size = config.log_wgt_buff_size
    pb.log_acc_buff_size = config.log_acc_buff_size
    pb.ins_width = config.ins_width
    pb.uop_width = config.uop_width
    return pb


def proto_to_config(pb: "_ProtobufVTAConfig") -> VTAConfig:
    """Convert a protobuf VTAConfig message to a pure Python VTAConfig.

    Args:
        pb: Protobuf VTAConfig message.

    Returns:
        VTAConfig dataclass instance.
    """
    _require_protobuf()
    return VTAConfig(
        target=pb.target,
        hw_version=pb.hw_version,
        log_bus_width=pb.log_bus_width,
        log_inp_width=pb.log_inp_width,
        log_wgt_width=pb.log_wgt_width,
        log_acc_width=pb.log_acc_width,
        log_out_width=pb.log_out_width,
        log_batch=pb.log_batch,
        log_block_in=pb.log_block_in,
        log_block_out=pb.log_block_out,
        log_uop_buff_size=pb.log_uop_buff_size,
        log_inp_buff_size=pb.log_inp_buff_size,
        log_wgt_buff_size=pb.log_wgt_buff_size,
        log_acc_buff_size=pb.log_acc_buff_size,
        ins_width=pb.ins_width,
        uop_width=pb.uop_width,
    )


def serialize(config: VTAConfig) -> bytes:
    """Serialize VTAConfig to binary protobuf format.

    Args:
        config: VTAConfig dataclass instance.

    Returns:
        Serialized binary bytes.
    """
    pb = config_to_proto(config)
    return pb.SerializeToString()


def deserialize(data: bytes) -> VTAConfig:
    """Deserialize VTAConfig from binary protobuf format.

    Args:
        data: Serialized binary bytes.

    Returns:
        VTAConfig dataclass instance.
    """
    _require_protobuf()
    pb = _pb2.VTAConfig()
    pb.ParseFromString(data)
    return proto_to_config(pb)


def save_config(config: VTAConfig, path: Union[str, os.PathLike]) -> None:
    """Save VTAConfig to a binary protobuf file.

    Args:
        config: VTAConfig dataclass instance.
        path: Output file path (.bin or .pb extension recommended).
    """
    _require_protobuf()
    data = serialize(config)
    Path(path).write_bytes(data)


def load_config(path: Union[str, os.PathLike]) -> VTAConfig:
    """Load VTAConfig from a binary protobuf file.

    Args:
        path: Input file path.

    Returns:
        VTAConfig dataclass instance.
    """
    _require_protobuf()
    data = Path(path).read_bytes()
    return deserialize(data)


def save_json(config: VTAConfig, path: Union[str, os.PathLike]) -> None:
    """Save VTAConfig to a JSON text file (using protobuf JSON format).

    Args:
        config: VTAConfig dataclass instance.
        path: Output file path (.json extension).
    """
    _require_protobuf()
    from google.protobuf import json_format

    pb = config_to_proto(config)
    json_str = json_format.MessageToJson(pb, indent=2, preserving_proto_field_name=True)
    Path(path).write_text(json_str, encoding="utf-8")


def load_json(path: Union[str, os.PathLike]) -> VTAConfig:
    """Load VTAConfig from a JSON text file (using protobuf JSON format).

    Args:
        path: Input file path (.json).

    Returns:
        VTAConfig dataclass instance.
    """
    _require_protobuf()
    from google.protobuf import json_format

    json_str = Path(path).read_text(encoding="utf-8")
    pb = _pb2.VTAConfig()
    json_format.Parse(json_str, pb)
    return proto_to_config(pb)


def save_text(config: VTAConfig, path: Union[str, os.PathLike]) -> None:
    """Save VTAConfig to a protobuf text format file.

    Args:
        config: VTAConfig dataclass instance.
        path: Output file path (.textproto or .pbtxt extension).
    """
    _require_protobuf()
    from google.protobuf import text_format

    pb = config_to_proto(config)
    text_str = text_format.MessageToString(pb)
    Path(path).write_text(text_str, encoding="utf-8")


def load_text(path: Union[str, os.PathLike]) -> VTAConfig:
    """Load VTAConfig from a protobuf text format file.

    Args:
        path: Input file path (.textproto or .pbtxt).

    Returns:
        VTAConfig dataclass instance.
    """
    _require_protobuf()
    from google.protobuf import text_format

    text_str = Path(path).read_text(encoding="utf-8")
    pb = _pb2.VTAConfig()
    text_format.Parse(text_str, pb)
    return proto_to_config(pb)
