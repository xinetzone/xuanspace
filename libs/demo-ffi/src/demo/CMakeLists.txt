set(DEMO_SOURCES
  ffi_registry.cc
  math_ops.cc
)

if(DEMO_FFI_USE_STUB)
  message(STATUS "Using stub demo runtime implementation")
  list(APPEND DEMO_SOURCES stub_rt.cc)
else()
  message(STATUS "Using real demo runtime implementation")
  list(APPEND DEMO_SOURCES real_rt.cc)
endif()

add_library(demo_ffi_demo SHARED ${DEMO_SOURCES})

target_include_directories(demo_ffi_demo PUBLIC ${DEMO_FFI_INCLUDE_DIR})

if(NOT DEMO_FFI_USE_STUB)
  target_link_libraries(demo_ffi_demo PRIVATE DEMO::DEMO)
endif()

tvm_ffi_configure_target(demo_ffi_demo
  STUB_DIR "${CMAKE_SOURCE_DIR}/python"
  STUB_INIT OFF
)

install(TARGETS demo_ffi_demo DESTINATION .)
tvm_ffi_install(demo_ffi_demo DESTINATION .)
