"""Tests for VTA enum values - verify consistency with C++ definitions."""
from __future__ import annotations

import pytest

from npu_ffi import vta


class TestDebugFlag:
    """Tests for DebugFlag IntFlag enumeration."""

    def test_dump_insn_value(self):
        assert int(vta.DebugFlag.DUMP_INSN) == (1 << 1)

    def test_dump_uop_value(self):
        assert int(vta.DebugFlag.DUMP_UOP) == (1 << 2)

    def test_skip_read_barrier_value(self):
        assert int(vta.DebugFlag.SKIP_READ_BARRIER) == (1 << 3)

    def test_skip_write_barrier_value(self):
        assert int(vta.DebugFlag.SKIP_WRITE_BARRIER) == (1 << 4)

    def test_force_serial_value(self):
        assert int(vta.DebugFlag.FORCE_SERIAL) == (1 << 5)

    def test_flags_can_be_combined(self):
        combined = vta.DebugFlag.DUMP_INSN | vta.DebugFlag.DUMP_UOP
        assert int(combined) == (1 << 1) | (1 << 2)
        assert (combined & vta.DebugFlag.DUMP_INSN) != 0
        assert (combined & vta.DebugFlag.DUMP_UOP) != 0


class TestMemcpyKind:
    """Tests for MemcpyKind IntEnum enumeration."""

    def test_h2d_value(self):
        assert int(vta.MemcpyKind.H2D) == 1

    def test_d2h_value(self):
        assert int(vta.MemcpyKind.D2H) == 2

    def test_d2d_value(self):
        assert int(vta.MemcpyKind.D2D) == 3

    def test_all_values_unique(self):
        values = {int(k) for k in vta.MemcpyKind}
        assert len(values) == 3


class TestMemoryType:
    """Tests for MemoryType IntEnum enumeration."""

    def test_dram_value(self):
        assert int(vta.MemoryType.DRAM) == 0

    def test_sram_value(self):
        assert int(vta.MemoryType.SRAM) == 1

    def test_uop_value(self):
        assert int(vta.MemoryType.UOP) == 2

    def test_inp_value(self):
        assert int(vta.MemoryType.INP) == 3

    def test_wgt_value(self):
        assert int(vta.MemoryType.WGT) == 4

    def test_acc_value(self):
        assert int(vta.MemoryType.ACC) == 5

    def test_out_value(self):
        assert int(vta.MemoryType.OUT) == 6

    def test_all_values_unique(self):
        values = {int(k) for k in vta.MemoryType}
        assert len(values) == 7


class TestALUOpcode:
    """Tests for ALUOpcode IntEnum enumeration."""

    def test_add_value(self):
        assert int(vta.ALUOpcode.ADD) == 0

    def test_sub_value(self):
        assert int(vta.ALUOpcode.SUB) == 1

    def test_mul_value(self):
        assert int(vta.ALUOpcode.MUL) == 2

    def test_min_value(self):
        assert int(vta.ALUOpcode.MIN) == 3

    def test_max_value(self):
        assert int(vta.ALUOpcode.MAX) == 4

    def test_shr_value(self):
        assert int(vta.ALUOpcode.SHR) == 5

    def test_shl_value(self):
        assert int(vta.ALUOpcode.SHL) == 6

    def test_all_values_unique(self):
        values = {int(k) for k in vta.ALUOpcode}
        assert len(values) == 7
