"""Tests for VTA CommandContext and command handle management."""

import gc
import pytest

from npu_ffi import vta
from npu_ffi.vta import Buffer, CommandContext, command_handle, get_command_handle


class TestCommandContextBasic:
    """Tests for basic CommandContext functionality."""

    def test_context_manager_returns_handle(self):
        with CommandContext() as cmd:
            assert isinstance(cmd, int)
            assert cmd != 0

    def test_command_handle_helper(self):
        cmd = command_handle()
        assert isinstance(cmd, int)
        assert cmd != 0

    def test_get_command_handle_alias(self):
        cmd1 = command_handle()
        cmd2 = get_command_handle()
        assert isinstance(cmd2, int)
        assert cmd2 != 0
        assert cmd2 >= cmd1

    def test_handle_property_when_entered(self):
        ctx = CommandContext()
        with ctx as cmd:
            assert ctx.handle == cmd
            assert isinstance(ctx.handle, int)

    def test_handle_property_raises_when_not_entered(self):
        ctx = CommandContext()
        with pytest.raises(RuntimeError, match="not entered"):
            _ = ctx.handle

    def test_command_context_repr_inactive(self):
        ctx = CommandContext()
        repr_str = repr(ctx)
        assert "CommandContext" in repr_str
        assert "inactive" in repr_str
        assert "cmd=None" in repr_str

    def test_command_context_repr_active(self):
        ctx = CommandContext()
        with ctx as cmd:
            repr_str = repr(ctx)
            assert "active" in repr_str
            assert "wait_cycles=0" in repr_str

    def test_explicit_synchronize(self):
        ctx = CommandContext()
        with ctx as cmd:
            assert ctx.handle == cmd
            ctx.synchronize()
        assert ctx._cmd is None

    def test_double_synchronize_safe(self):
        ctx = CommandContext()
        with ctx as cmd:
            ctx.synchronize()
            ctx.synchronize()
        assert ctx._cmd is None

    def test_synchronize_before_exit(self):
        ctx = CommandContext()
        with ctx as cmd:
            ctx.synchronize()
            assert ctx._cmd is None


class TestCommandContextWorkflow:
    """Tests for complete command workflow with buffers and ops."""

    def test_synchronize_on_exit(self):
        with CommandContext() as cmd:
            assert cmd != 0

    def test_context_with_wait_cycles(self):
        with CommandContext(wait_cycles=100) as cmd:
            assert cmd != 0

    def test_full_command_pipeline(self):
        with CommandContext() as cmd:
            inp = Buffer(1024)
            wgt = Buffer(1024)
            acc = Buffer(1024)
            assert inp.data != 0
            assert wgt.data != 0
            assert acc.data != 0

            vta.load_buffer_2d(
                cmd, inp.data, 0, 16, 8, 32,
                0, 0, 0, 0, 0, int(vta.MemoryType.INP)
            )
            vta.load_buffer_2d(
                cmd, wgt.data, 0, 16, 8, 32,
                0, 0, 0, 0, 0, int(vta.MemoryType.WGT)
            )

            vta.uop_loop_begin(1, 0, 0, 0)
            vta.uop_push(0, 1, 0, 0, 0, int(vta.ALUOpcode.ADD), 0, 0)
            vta.uop_loop_end()

            vta.push_gemm_op()

            vta.store_buffer_2d(
                cmd, 0, int(vta.MemoryType.ACC),
                acc.data, 0, 16, 8, 32
            )

            inp.write_barrier(cmd, 32, 0, 8)
            acc.read_barrier(cmd, 32, 0, 8)


class TestMultipleContexts:
    """Tests for sequential CommandContext usage."""

    def test_sequential_contexts(self):
        with CommandContext() as cmd1:
            assert cmd1 != 0
        with CommandContext() as cmd2:
            assert cmd2 != 0
            assert cmd2 != cmd1

    def test_command_handle_increments(self):
        cmd1 = command_handle()
        cmd2 = command_handle()
        assert cmd2 > cmd1

    def test_many_sequential_contexts(self):
        handles = []
        for _ in range(10):
            with CommandContext() as cmd:
                handles.append(cmd)
        for i in range(1, len(handles)):
            assert handles[i] != handles[i - 1]


class TestCommandContextWithDebug:
    """Tests for CommandContext with debug mode."""

    def test_set_debug_in_context(self):
        with CommandContext() as cmd:
            vta.set_debug_mode(cmd, int(vta.DebugFlag.DUMP_INSN))

    def test_debug_mode_combined_flags(self):
        with CommandContext() as cmd:
            flags = (
                int(vta.DebugFlag.DUMP_INSN) |
                int(vta.DebugFlag.DUMP_UOP)
            )
            vta.set_debug_mode(cmd, flags)


class TestCommandContextWithDependencies:
    """Tests for dependency operations within context."""

    def test_dep_push_pop_in_context(self):
        with CommandContext() as cmd:
            assert vta.dep_push(cmd, 0, 1) == 0
            assert vta.dep_pop(cmd, 0, 1) == 0

    def test_dep_chain(self):
        with CommandContext() as cmd:
            for i in range(3):
                assert vta.dep_push(cmd, i, i + 1) == 0
            for i in range(3):
                assert vta.dep_pop(cmd, i, i + 1) == 0


class TestCommandContextPrepareCall:
    """Tests for prepare_call_func within context."""

    def test_prepare_call_in_context(self):
        with CommandContext() as cmd:
            vta.prepare_call_func(cmd, "test_kernel")
