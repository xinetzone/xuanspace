set(VTA_SOURCES
  command_context.cc
  ffi_registry.cc
  runtime_common.cc
)

if(NPU_FFI_VTA_USE_STUB)
  message(STATUS "Using stub VTA runtime implementation")
  list(APPEND VTA_SOURCES stub_rt.cc)
else()
  message(STATUS "Using real VTA runtime implementation")
  list(APPEND VTA_SOURCES real_rt.cc)
endif()

add_library(npu_ffi_vta SHARED ${VTA_SOURCES})

target_compile_definitions(npu_ffi_vta PRIVATE NPU_FFI_EXPORTS)

target_include_directories(npu_ffi_vta PUBLIC ${NPU_FFI_INCLUDE_DIR})

if(NOT NPU_FFI_VTA_USE_STUB)
  target_link_libraries(npu_ffi_vta PRIVATE VTA::VTA)
endif()

tvm_ffi_configure_target(npu_ffi_vta
  STUB_DIR "${CMAKE_SOURCE_DIR}/python"
  STUB_INIT OFF
)

install(TARGETS npu_ffi_vta DESTINATION .)
tvm_ffi_install(npu_ffi_vta DESTINATION .)
