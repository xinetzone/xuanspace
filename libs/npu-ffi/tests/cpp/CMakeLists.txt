# NPU FFI C++ Tests
#
# Tests are organized by instruction category following the five-layer FFI pattern:
# - test_buffer:      Lifecycle management (RAII, move semantics, CommandContext)
# - test_memory_ops:  Memory operations (copy, load/store, barriers, cpu_ptr)
# - test_compute_ops: Compute operations (GEMM/ALU uop pushes, opcodes)
# - test_control_flow: Control flow (loops, dependencies, sync, debug flags)
#
# When adding tests for new NPU instructions, create a test_<category>.cc
# file following the pattern in test_memory_ops.cc, then add it below.
#
# Note: npu_ffi_vta propagates include directories via PUBLIC target_include_directories,
# so test executables automatically get the correct include paths by linking.

add_executable(test_buffer test_buffer.cc)
target_link_libraries(test_buffer PRIVATE npu_ffi_vta)
add_test(NAME test_buffer COMMAND test_buffer)

add_executable(test_memory_ops test_memory_ops.cc)
target_link_libraries(test_memory_ops PRIVATE npu_ffi_vta)
add_test(NAME test_memory_ops COMMAND test_memory_ops)

add_executable(test_compute_ops test_compute_ops.cc)
target_link_libraries(test_compute_ops PRIVATE npu_ffi_vta)
add_test(NAME test_compute_ops COMMAND test_compute_ops)

add_executable(test_control_flow test_control_flow.cc)
target_link_libraries(test_control_flow PRIVATE npu_ffi_vta)
add_test(NAME test_control_flow COMMAND test_control_flow)
