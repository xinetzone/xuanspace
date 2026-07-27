"""VTA hardware configuration - pure Python dataclass implementation.

This module provides VTAConfig dataclass for hardware parameter configuration
without requiring protobuf at runtime. For protobuf serialization, see proto_io.py.

All size parameters are in log2 domain: actual_value = 2 ** log_value
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class VTAConfig:
    """VTA hardware configuration parameters.

    All log_* parameters represent log2 values; the actual hardware value
    is computed as 2 ** log_value. This ensures hardware-friendly power-of-two
    dimensions for the accelerator.
    """

    target: str = "pynq"
    hw_version: str = "v1.0.0"

    # Bus/interface
    log_bus_width: int = 3

    # Data type widths (log2 bits)
    log_inp_width: int = 3
    log_wgt_width: int = 3
    log_acc_width: int = 5
    log_out_width: int = 3

    # Block/tile dimensions (log2)
    log_batch: int = 0
    log_block_in: int = 4
    log_block_out: int = 4

    # On-chip SRAM buffer sizes (log2 entries/bytes)
    log_uop_buff_size: int = 11
    log_inp_buff_size: int = 15
    log_wgt_buff_size: int = 15
    log_acc_buff_size: int = 13

    # Instruction format (log2 bits)
    ins_width: int = 7
    uop_width: int = 5

    def __post_init__(self) -> None:
        validate_config(self)

    @property
    def bus_width(self) -> int:
        """Actual bus width in bytes."""
        return 1 << self.log_bus_width

    @property
    def inp_width(self) -> int:
        """Input type width in bits."""
        return 1 << self.log_inp_width

    @property
    def wgt_width(self) -> int:
        """Weight type width in bits."""
        return 1 << self.log_wgt_width

    @property
    def acc_width(self) -> int:
        """Accumulator width in bits."""
        return 1 << self.log_acc_width

    @property
    def out_width(self) -> int:
        """Output type width in bits."""
        return 1 << self.log_out_width

    @property
    def batch(self) -> int:
        """Batch dimension."""
        return 1 << self.log_batch

    @property
    def block_in(self) -> int:
        """Input block dimension (B)."""
        return 1 << self.log_block_in

    @property
    def block_out(self) -> int:
        """Output block dimension (C)."""
        return 1 << self.log_block_out

    @property
    def uop_buff_size(self) -> int:
        """UOP buffer size in entries."""
        return 1 << self.log_uop_buff_size

    @property
    def inp_buff_size(self) -> int:
        """Input buffer size in bytes."""
        return 1 << self.log_inp_buff_size

    @property
    def wgt_buff_size(self) -> int:
        """Weight buffer size in bytes."""
        return 1 << self.log_wgt_buff_size

    @property
    def acc_buff_size(self) -> int:
        """Accumulator buffer size in acc-width elements."""
        return 1 << self.log_acc_buff_size

    @property
    def ins_width_bits(self) -> int:
        """Instruction width in bits."""
        return 1 << self.ins_width

    @property
    def uop_width_bits(self) -> int:
        """UOP width in bits."""
        return 1 << self.uop_width

    @property
    def acc_buff_bytes(self) -> int:
        """Accumulator buffer size in bytes."""
        return self.acc_buff_size * (self.acc_width // 8)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VTAConfig:
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def replace(self, **changes: Any) -> VTAConfig:
        """Create a new config with specified fields replaced."""
        return dataclasses.replace(self, **changes)


def validate_config(config: VTAConfig) -> None:
    """Validate VTA configuration parameters.

    Args:
        config: The VTAConfig to validate.

    Raises:
        ValueError: If any parameter is invalid.
    """
    errors: list[str] = []

    if config.log_bus_width < 2 or config.log_bus_width > 8:
        errors.append(
            f"log_bus_width={config.log_bus_width} out of range [2, 8] "
            "(4 bytes to 256 bytes bus width)"
        )

    for name, val in [
        ("log_inp_width", config.log_inp_width),
        ("log_wgt_width", config.log_wgt_width),
        ("log_out_width", config.log_out_width),
    ]:
        if val < 2 or val > 5:
            errors.append(f"{name}={val} out of range [2, 5] (4 bits to 32 bits)")

    if config.log_acc_width < 4 or config.log_acc_width > 6:
        errors.append(
            f"log_acc_width={config.log_acc_width} out of range [4, 6] "
            "(16 bits to 64 bits accumulator)"
        )

    if config.acc_width < config.inp_width + config.wgt_width + config.log_block_in + config.log_batch:
        errors.append(
            f"Accumulator width ({config.acc_width} bits) may be insufficient "
            f"for GEMM: need at least {config.inp_width + config.wgt_width + config.log_block_in + config.log_batch} bits"
        )

    for name, val in [
        ("log_batch", config.log_batch),
        ("log_block_in", config.log_block_in),
        ("log_block_out", config.log_block_out),
    ]:
        if val < 0 or val > 8:
            errors.append(f"{name}={val} out of range [0, 8]")

    for name, val, min_val, max_val in [
        ("log_uop_buff_size", config.log_uop_buff_size, 8, 16),
        ("log_inp_buff_size", config.log_inp_buff_size, 10, 20),
        ("log_wgt_buff_size", config.log_wgt_buff_size, 10, 20),
        ("log_acc_buff_size", config.log_acc_buff_size, 9, 18),
    ]:
        if val < min_val or val > max_val:
            errors.append(f"{name}={val} out of range [{min_val}, {max_val}]")

    if config.ins_width < 5 or config.ins_width > 8:
        errors.append(
            f"ins_width={config.ins_width} out of range [5, 8] "
            "(32 bits to 256 bits instruction)"
        )

    if config.uop_width < 4 or config.uop_width > 6:
        errors.append(
            f"uop_width={config.uop_width} out of range [4, 6] "
            "(16 bits to 64 bits UOP)"
        )

    if errors:
        raise ValueError("Invalid VTAConfig:\n  " + "\n  ".join(errors))


_VTA_STANDARD = VTAConfig(
    target="pynq",
    hw_version="v1.0.0",
    log_bus_width=3,
    log_inp_width=3,
    log_wgt_width=3,
    log_acc_width=5,
    log_out_width=3,
    log_batch=0,
    log_block_in=4,
    log_block_out=4,
    log_uop_buff_size=11,
    log_inp_buff_size=15,
    log_wgt_buff_size=15,
    log_acc_buff_size=13,
    ins_width=7,
    uop_width=5,
)

_VTA_V3 = VTAConfig(
    target="ultra96",
    hw_version="v3.0.0",
    log_bus_width=4,
    log_inp_width=3,
    log_wgt_width=3,
    log_acc_width=5,
    log_out_width=3,
    log_batch=0,
    log_block_in=4,
    log_block_out=4,
    log_uop_buff_size=12,
    log_inp_buff_size=16,
    log_wgt_buff_size=16,
    log_acc_buff_size=14,
    ins_width=7,
    uop_width=5,
)

_VTA_V4 = VTAConfig(
    target="zcu104",
    hw_version="v4.0.0",
    log_bus_width=5,
    log_inp_width=3,
    log_wgt_width=3,
    log_acc_width=5,
    log_out_width=3,
    log_batch=1,
    log_block_in=5,
    log_block_out=5,
    log_uop_buff_size=13,
    log_inp_buff_size=17,
    log_wgt_buff_size=17,
    log_acc_buff_size=15,
    ins_width=7,
    uop_width=5,
)

DEFAULT_CONFIGS: Dict[str, VTAConfig] = {
    "vta": _VTA_STANDARD,
    "vta_v3": _VTA_V3,
    "vta_v4": _VTA_V4,
}


def get_default_config(name: str = "vta") -> VTAConfig:
    """Get a default VTA configuration by name.

    Args:
        name: Configuration name ("vta", "vta_v3", or "vta_v4").

    Returns:
        A VTAConfig instance with default values.

    Raises:
        KeyError: If name is not a known default configuration.
    """
    if name not in DEFAULT_CONFIGS:
        raise KeyError(
            f"Unknown default config '{name}'. Available: {list(DEFAULT_CONFIGS.keys())}"
        )
    return DEFAULT_CONFIGS[name]
