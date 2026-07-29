# Tests.cmake - C++ 单元测试目标配置
enable_testing()

file(GLOB CAFFE_FFI_CPP_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/*.cpp"
)

add_executable(caffe_ffi_tests
  ${CAFFE_FFI_CPP_TEST_SRCS}
)

caffe_ffi_configure_target(caffe_ffi_tests VISIBILITY PRIVATE)

target_include_directories(caffe_ffi_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp"
)

target_link_libraries(caffe_ffi_tests PRIVATE
  _caffe_ffi
  tvm_ffi::shared
)

if(MSVC)
  caffe_ffi_copy_runtime_dlls(caffe_ffi_tests)
  caffe_ffi_copy_target_dll(caffe_ffi_tests _caffe_ffi)
endif()

add_test(NAME caffe_ffi_cpp_tests COMMAND caffe_ffi_tests)
