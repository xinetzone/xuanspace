# Dependencies.cmake - 第三方依赖查找与配置
# TVM FFI、Protobuf、Threads 依赖在此查找；BLAS 检测委托给 DetectBLAS.cmake
#
# tvm-ffi 依赖查找支持三种模式：
# 1. 发布模式（默认）：通过 find_package(tvm_ffi CONFIG REQUIRED) 查找已安装的 tvm-ffi
# 2. 自动开发模式：当 CAFFE_FFI_TVM_FFI_DIR 未设置时，自动检测 ../../vendor/tvm-ffi 是否存在
#    CMakeLists.txt，若存在则自动使用本地源码构建（从libs/caffe-ffi视角：caffe-ffi→libs→xuanspace→vendor/tvm-ffi）
# 3. 显式开发模式：通过 -DCAFFE_FFI_TVM_FFI_DIR=/path/to/tvm-ffi 显式指定源码路径

set(CAFFE_FFI_TVM_FFI_DIR "" CACHE PATH "Path to tvm-ffi source directory for local development build")

if(CAFFE_FFI_TVM_FFI_DIR)
  set(_tvm_ffi_src_dir "${CAFFE_FFI_TVM_FFI_DIR}")
  message(STATUS "Using tvm-ffi from explicit source directory: ${_tvm_ffi_src_dir}")
else()
  set(_auto_tvm_ffi_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/tvm-ffi")
  if(EXISTS "${_auto_tvm_ffi_dir}/CMakeLists.txt")
    set(_tvm_ffi_src_dir "${_auto_tvm_ffi_dir}")
    message(STATUS "Auto-detected local tvm-ffi source: ${_tvm_ffi_src_dir}")
  endif()
endif()

if(DEFINED _tvm_ffi_src_dir)
  set(TVM_FFI_USE_LIBBACKTRACE OFF CACHE BOOL "Disable libbacktrace" FORCE)
  set(TVM_FFI_BACKTRACE_ON_SEGFAULT OFF CACHE BOOL "Disable segfault backtrace" FORCE)
  add_subdirectory("${_tvm_ffi_src_dir}" tvm-ffi EXCLUDE_FROM_ALL)
  if(NOT TARGET tvm_ffi::shared)
    add_library(tvm_ffi::shared ALIAS tvm_ffi_shared)
  endif()
else()
  # 当 tvm-ffi 已通过 pip 安装（如 apache-tvm-ffi）时，
  # 使用 python -m tvm_ffi.config --cmakedir 获取 CMake 配置目录
  find_package(Python COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND OR Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE _tvm_ffi_cmakedir
      ERROR_QUIET
      RESULT_VARIABLE _tvm_ffi_config_result
    )
    if(_tvm_ffi_config_result EQUAL 0 AND _tvm_ffi_cmakedir)
      message(STATUS "Found tvm-ffi CMake config via Python: ${_tvm_ffi_cmakedir}")
      set(tvm_ffi_ROOT "${_tvm_ffi_cmakedir}" CACHE PATH "Path to tvm-ffi CMake config directory" FORCE)
    endif()
  endif()
  find_package(tvm_ffi CONFIG REQUIRED)
endif()

set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "Use module-compatible protobuf variables" FORCE)
find_package(Protobuf CONFIG REQUIRED)
if(Protobuf_VERSION VERSION_LESS "7.0.0")
  message(FATAL_ERROR "Protobuf >= 7.0.0 is required, found ${Protobuf_VERSION}")
endif()
message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
find_package(Threads REQUIRED)

include(DetectBLAS)

find_package(Python COMPONENTS Interpreter QUIET)
