# Tests.cmake - C++ 单元测试 + Python 回归测试 + CMake 模块测试
enable_testing()

# ── CMake 模块单元测试 ──

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/test_detect_openblas.cmake")
  add_test(
    NAME detect_openblas_module
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/test_detect_openblas.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  set_tests_properties(detect_openblas_module PROPERTIES
    LABELS "cmake;module;blas"
    TIMEOUT 30
  )
endif()

# ── C++ 单元测试 ──

file(GLOB CAFFE_FFI_CPP_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/*.cpp"
)
list(LENGTH CAFFE_FFI_CPP_TEST_SRCS _cpp_test_count)
message(STATUS "[caffe_ffi] C++ test source count: ${_cpp_test_count}")
message(STATUS "[caffe_ffi] C++ test sources: ${CAFFE_FFI_CPP_TEST_SRCS}")

add_executable(caffe_ffi_tests
  ${CAFFE_FFI_CPP_TEST_SRCS}
)
message(STATUS "[caffe_ffi] caffe_ffi_tests target created (executable)")

caffe_ffi_configure_target(caffe_ffi_tests VISIBILITY PRIVATE)

target_include_directories(caffe_ffi_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp"
)

target_link_libraries(caffe_ffi_tests PRIVATE
  _caffe_ffi
  tvm_ffi::shared
)
message(STATUS "[caffe_ffi] caffe_ffi_tests links: _caffe_ffi (PRIVATE), tvm_ffi::shared (PRIVATE)")

if(MSVC)
  caffe_ffi_copy_runtime_dlls(caffe_ffi_tests)
  caffe_ffi_copy_target_dll(caffe_ffi_tests _caffe_ffi)
  message(STATUS "[caffe_ffi] caffe_ffi_tests: runtime DLLs + _caffe_ffi.dll copy configured")
endif()

add_test(NAME caffe_ffi_cpp_tests COMMAND caffe_ffi_tests)

# ── Python 测试 ──

find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_Interpreter_FOUND)
  message(STATUS "[caffe_ffi] Python3 interpreter: ${Python3_EXECUTABLE}")
  # ── P2-B 回归测试套件 ──
  # 包含: Split 拓扑正确性 + 性能扩展 + 极端边界 + 内存稳定性
  set(P2B_REGRESSION_TEST "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_p2b_regression.py")

  if(EXISTS "${P2B_REGRESSION_TEST}")
    add_test(
      NAME caffe_ffi_python_p2b_regression
      COMMAND ${Python3_EXECUTABLE} -m pytest "${P2B_REGRESSION_TEST}" -v --tb=short
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(caffe_ffi_python_p2b_regression PROPERTIES
      LABELS "python;p2b;regression"
      TIMEOUT 300
      ENVIRONMENT "KMP_DUPLICATE_LIB_OK=TRUE"
    )
  endif()

  # ── P2-B 性能测试（仅性能埋点，含 SPLIT-PERF 日志输出）──
  if(EXISTS "${P2B_REGRESSION_TEST}")
    add_test(
      NAME caffe_ffi_python_p2b_performance
      COMMAND ${Python3_EXECUTABLE} -m pytest "${P2B_REGRESSION_TEST}::TestSplitPerformanceScaling" -v -s
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(caffe_ffi_python_p2b_performance PROPERTIES
      LABELS "python;p2b;performance"
      TIMEOUT 600
      ENVIRONMENT "KMP_DUPLICATE_LIB_OK=TRUE;CAFFE_FFI_LOG_LEVEL=WARN"
    )
  endif()

  # ── 全量 Python 测试 ──
  file(GLOB CAFFE_FFI_PYTHON_TESTS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_*.py"
  )
  if(CAFFE_FFI_PYTHON_TESTS)
    list(LENGTH CAFFE_FFI_PYTHON_TESTS _py_test_count)
    message(STATUS "[caffe_ffi] Python test files: ${_py_test_count}")
    add_test(
      NAME caffe_ffi_python_all
      COMMAND ${Python3_EXECUTABLE} -m pytest tests/python/ -v --tb=short
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(caffe_ffi_python_all PROPERTIES
      LABELS "python;all"
      TIMEOUT 600
      ENVIRONMENT "KMP_DUPLICATE_LIB_OK=TRUE"
    )
  endif()

  # ── 聚合自定义目标 ──

  # p2b-regression: 仅运行 P2-B 回归（C++ + Python P2-B）
  add_custom_target(p2b-regression
    COMMAND ${CMAKE_CTEST_COMMAND} -R "caffe_ffi_cpp_tests|caffe_ffi_python_p2b_regression" --output-on-failure
    DEPENDS caffe_ffi_tests _caffe_ffi
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Running P2-B regression tests (C++ + Python)"
  )

  # p2b-performance: 仅运行性能埋点测试
  add_custom_target(p2b-performance
    COMMAND ${CMAKE_CTEST_COMMAND} -R "caffe_ffi_python_p2b_performance" --output-on-failure -V
    DEPENDS _caffe_ffi
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Running P2-B performance scaling tests with SPLIT-PERF logging"
  )

  # check-all: 运行所有测试（C++ + 全量 Python）
  add_custom_target(check-all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS caffe_ffi_tests _caffe_ffi
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Running all tests (C++ + Python)"
  )
else()
  message(STATUS "Python3 not found; Python tests will not be registered.")
endif()
