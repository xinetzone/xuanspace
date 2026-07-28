add_executable(test_buffer test_buffer.cc)
target_link_libraries(test_buffer PRIVATE npu_ffi_vta)

add_test(NAME test_buffer COMMAND test_buffer)
