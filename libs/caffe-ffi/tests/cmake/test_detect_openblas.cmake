# test_detect_openblas.cmake — DetectOpenBLAS 模块单元测试
# ============================================================
# 使用方式:
#   cmake -P tests/cmake/test_detect_openblas.cmake
# 或通过 CTest:
#   ctest -R detect_openblas -V
#
# 测试覆盖:
#   T1 - 函数可加载性（模块语法正确）
#   T2 - 基本检测流程（不崩溃、输出变量定义）
#   T3 - USE_BLAS=OFF 显式禁用
#   T4 - EXTRA_HINTS 自定义路径（mock 环境）
#   T5 - 空 EXTRA_HINTS 不报错
#   T6 - 输出变量类型正确（BLAS_FOUND 为布尔语义）

cmake_minimum_required(VERSION 3.26)

set(_test_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_module_dir "${_test_dir}/../../cmake")
set(_passed 0)
set(_failed 0)
set(_skipped 0)

# ── 辅助宏 ──

macro(test_pass _name)
  message(STATUS "[PASS] ${_name}")
  math(EXPR _passed "${_passed} + 1")
endmacro()

macro(test_fail _name _reason)
  message(STATUS "[FAIL] ${_name}: ${_reason}")
  math(EXPR _failed "${_failed} + 1")
endmacro()

macro(test_skip _name _reason)
  message(STATUS "[SKIP] ${_name}: ${_reason}")
  math(EXPR _skipped "${_skipped} + 1")
endmacro()

# ── T1: 模块可加载性 ──

include("${_module_dir}/DetectOpenBLAS.cmake")

if(COMMAND detect_openblas)
  test_pass("T1 - Module loads, detect_openblas function defined")
else()
  test_fail("T1 - Module loads" "detect_openblas not defined as a function")
endif()

# ── T2: 基本检测流程 ──

# 重置内部缓存变量（模拟干净的调用环境）
unset(OPENBLAS_INCLUDE_DIR CACHE)
unset(OPENBLAS_LIBRARY CACHE)

detect_openblas()

if(DEFINED BLAS_FOUND)
  test_pass("T2 - Basic detection: BLAS_FOUND defined (${BLAS_FOUND})")
else()
  test_fail("T2 - Basic detection" "BLAS_FOUND not defined")
endif()

if(DEFINED BLAS_LIBRARIES)
  test_pass("T2 - Basic detection: BLAS_LIBRARIES defined")
else()
  test_fail("T2 - Basic detection" "BLAS_LIBRARIES not defined")
endif()

if(DEFINED BLAS_INCLUDE_DIRS)
  test_pass("T2 - Basic detection: BLAS_INCLUDE_DIRS defined")
else()
  test_fail("T2 - Basic detection" "BLAS_INCLUDE_DIRS not defined")
endif()

# ── T3: USE_BLAS=OFF 显式禁用 ──

# 重置
unset(OPENBLAS_INCLUDE_DIR CACHE)
unset(OPENBLAS_LIBRARY CACHE)

set(USE_BLAS OFF)
detect_openblas()

if(NOT BLAS_FOUND)
  test_pass("T3 - USE_BLAS=OFF: BLAS_FOUND=OFF correctly")
else()
  test_fail("T3 - USE_BLAS=OFF" "BLAS_FOUND should be OFF but got ${BLAS_FOUND}")
endif()

# ── T4: EXTRA_HINTS 自定义路径（mock OpenBLAS 环境） ──

# 创建 mock 目录结构模拟 OpenBLAS 安装
set(_mock_root "${CMAKE_CURRENT_BINARY_DIR}/_mock_openblas")
file(MAKE_DIRECTORY "${_mock_root}/include/openblas")
file(MAKE_DIRECTORY "${_mock_root}/lib")

# 创建 mock 头文件
file(WRITE "${_mock_root}/include/openblas/cblas.h" "/* mock cblas.h */")
file(WRITE "${_mock_root}/include/openblas/openblas_config.h" "/* mock openblas_config.h */")

# 创建 mock 库文件
if(WIN32)
  file(WRITE "${_mock_root}/lib/libopenblas.lib" "mock lib")
else()
  file(WRITE "${_mock_root}/lib/libopenblas.a" "mock lib")
endif()

# 重置缓存
unset(OPENBLAS_INCLUDE_DIR CACHE)
unset(OPENBLAS_LIBRARY CACHE)
set(USE_BLAS ON)

detect_openblas(EXTRA_HINTS "${_mock_root}")

if(BLAS_FOUND AND OPENBLAS_INCLUDE_DIR)
  test_pass("T4 - EXTRA_HINTS: mock OpenBLAS detected at ${OPENBLAS_INCLUDE_DIR}")
else()
  test_fail("T4 - EXTRA_HINTS" "mock OpenBLAS not found (include: ${OPENBLAS_INCLUDE_DIR}, lib: ${OPENBLAS_LIBRARY})")
endif()

# ── T5: 空 EXTRA_HINTS 不报错 ──

unset(OPENBLAS_INCLUDE_DIR CACHE)
unset(OPENBLAS_LIBRARY CACHE)
set(USE_BLAS ON)

detect_openblas(EXTRA_HINTS)

test_pass("T5 - Empty EXTRA_HINTS: no crash, BLAS_FOUND=${BLAS_FOUND}")

# ── T6: 输出变量布尔语义 ──

unset(OPENBLAS_INCLUDE_DIR CACHE)
unset(OPENBLAS_LIBRARY CACHE)
set(USE_BLAS ON)

detect_openblas(EXTRA_HINTS "${_mock_root}")

if(BLAS_FOUND)
  if(BLAS_LIBRARIES AND BLAS_INCLUDE_DIRS)
    test_pass("T6 - Boolean semantics: BLAS_FOUND=ON with valid libraries and includes")
  else()
    test_fail("T6 - Boolean semantics" "BLAS_FOUND=ON but libraries/includes missing")
  endif()
else()
  test_skip("T6 - Boolean semantics" "mock OpenBLAS detection failed, skipping")
endif()

# ── 清理 mock 目录 ──
file(REMOVE_RECURSE "${_mock_root}")

# ── 结果汇总 ──

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  DetectOpenBLAS Unit Test Results")
message(STATUS "  Passed:  ${_passed}")
message(STATUS "  Failed:  ${_failed}")
message(STATUS "  Skipped: ${_skipped}")
message(STATUS "========================================")

if(_failed GREATER 0)
  message(FATAL_ERROR "${_failed} test(s) failed")
endif()