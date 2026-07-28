if(NPU_FFI_BUILD_EXAMPLES)
  add_executable(basic_usage basic_usage.cc)
  target_link_libraries(basic_usage PRIVATE npu_ffi_vta)
endif()
