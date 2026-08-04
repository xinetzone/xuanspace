# Dependencies.cmake - 第三方依赖查找与配置
# TVM FFI、Protobuf、Threads 依赖在此查找；BLAS 检测委托给 DetectBLAS.cmake
#
# tvm-ffi 依赖查找优先级：
# 1. 显式源码模式：通过 -DCAFFE_FFI_TVM_FFI_DIR=/path/to/tvm-ffi 显式指定源码路径（最高优先级）
# 2. 系统包优先模式（默认）：当 CAFFE_FFI_PREFER_SYSTEM_TVM_FFI=ON 时，先查找已安装的 tvm-ffi
# 3. 本地源码自动检测：检测 ../../vendor/tvm-ffi 是否存在，若存在则使用本地源码构建
# 4. 最终回退：强制 find_package(tvm_ffi CONFIG REQUIRED)，找不到则报错

# ── 辅助宏：打印导入目标的库文件路径 ──
macro(_caffe_ffi_log_imported_lib target_name label)
  if(TARGET ${target_name})
    get_target_property(_lib_type ${target_name} TYPE)
    if(_lib_type STREQUAL "INTERFACE_LIBRARY")
      get_target_property(_iface_libs ${target_name} INTERFACE_LINK_LIBRARIES)
      message(STATUS "  [DEP] ${label}: INTERFACE library, links: ${_iface_libs}")
    else()
      get_target_property(_lib_loc ${target_name} IMPORTED_LOCATION)
      if(NOT _lib_loc)
        get_target_property(_lib_loc ${target_name} IMPORTED_LOCATION_RELEASE)
      endif()
      if(NOT _lib_loc)
        get_target_property(_lib_loc ${target_name} IMPORTED_LOCATION_DEBUG)
      endif()
      if(_lib_loc)
        message(STATUS "  [DEP] ${label}: ${_lib_loc}")
      else()
        message(STATUS "  [DEP] ${label}: target exists (built locally, no IMPORTED_LOCATION)")
      endif()
      # Also log interface include dirs if available
      get_target_property(_iface_includes ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
      if(_iface_includes)
        message(STATUS "  [DEP] ${label} include: ${_iface_includes}")
      endif()
    endif()
  else()
    message(STATUS "  [DEP] ${label}: ${target_name} (raw, not a CMake target)")
  endif()
endmacro()

set(CAFFE_FFI_TVM_FFI_DIR "" CACHE PATH "Path to tvm-ffi source directory for local development build")
option(CAFFE_FFI_PREFER_SYSTEM_TVM_FFI "Prefer using system-installed tvm-ffi over local vendor source" ON)

set(_tvm_ffi_use_find_package FALSE)
set(_tvm_ffi_src_dir "")

if(CAFFE_FFI_TVM_FFI_DIR)
  set(_tvm_ffi_src_dir "${CAFFE_FFI_TVM_FFI_DIR}")
  message(STATUS "Using tvm-ffi from explicit source directory: ${_tvm_ffi_src_dir}")
else()
  if(CAFFE_FFI_PREFER_SYSTEM_TVM_FFI)
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
        set(_tvm_ffi_use_find_package TRUE)
      endif()
    endif()
  endif()

  if(NOT _tvm_ffi_use_find_package)
    set(_auto_tvm_ffi_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/tvm-ffi")
    if(EXISTS "${_auto_tvm_ffi_dir}/CMakeLists.txt")
      set(_tvm_ffi_src_dir "${_auto_tvm_ffi_dir}")
      message(STATUS "Auto-detected local tvm-ffi source, using add_subdirectory: ${_tvm_ffi_src_dir}")
    endif()
  endif()
endif()

if(_tvm_ffi_src_dir)
  set(TVM_FFI_USE_LIBBACKTRACE OFF CACHE BOOL "Disable libbacktrace" FORCE)
  set(TVM_FFI_BACKTRACE_ON_SEGFAULT OFF CACHE BOOL "Disable segfault backtrace" FORCE)
  add_subdirectory("${_tvm_ffi_src_dir}" tvm-ffi EXCLUDE_FROM_ALL)
  if(NOT TARGET tvm_ffi::shared)
    add_library(tvm_ffi::shared ALIAS tvm_ffi_shared)
  endif()
  message(STATUS "[DEP] tvm-ffi: built from local source (${_tvm_ffi_src_dir})")
  _caffe_ffi_log_imported_lib(tvm_ffi::shared "tvm_ffi::shared")
  _caffe_ffi_log_imported_lib(tvm_ffi::header "tvm_ffi::header")
else()
  find_package(tvm_ffi CONFIG REQUIRED)
  message(STATUS "[DEP] tvm-ffi: system package (tvm_ffi_ROOT=${tvm_ffi_ROOT})")
  _caffe_ffi_log_imported_lib(tvm_ffi::shared "tvm_ffi::shared")
  _caffe_ffi_log_imported_lib(tvm_ffi::header "tvm_ffi::header")
endif()

set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "Use module-compatible protobuf variables" FORCE)
find_package(Protobuf CONFIG REQUIRED)
if(Protobuf_VERSION VERSION_LESS "7.0.0")
  message(FATAL_ERROR "Protobuf >= 7.0.0 is required, found ${Protobuf_VERSION}")
endif()
message(STATUS "[DEP] Protobuf: v${Protobuf_VERSION} (protoc: ${Protobuf_PROTOC_EXECUTABLE})")
message(STATUS "[DEP] Protobuf include: ${Protobuf_INCLUDE_DIRS}")
_caffe_ffi_log_imported_lib(protobuf::libprotobuf "protobuf::libprotobuf")

find_package(Threads REQUIRED)
message(STATUS "[DEP] Threads: CMAKE_THREAD_LIBS_INIT=${CMAKE_THREAD_LIBS_INIT}")
_caffe_ffi_log_imported_lib(Threads::Threads "Threads::Threads")

include(DetectBLAS)
message(STATUS "[DEP] BLAS: ${BLAS_VENDOR} (libraries: ${BLAS_LIBRARIES})")
if(BLAS_INCLUDE_DIRS)
  message(STATUS "[DEP] BLAS include: ${BLAS_INCLUDE_DIRS}")
endif()

# ── OpenMP 检测（CAFFE_USE_OPENMP=ON 时试图启用）──
# 结果变量 CAFFE_USE_OPENMP_FOUND 供 CompilerConfig 在目标上追加
# -fopenmp / /openmp 编译选项、OpenMP::OpenMP_CXX 链接目标与 CAFFE_USE_OPENMP 宏。
# 若编译器不支持 OpenMP（如部分 CI 工具链），自动回退为串行执行，不影响构建。
set(CAFFE_USE_OPENMP_FOUND FALSE)
if(CAFFE_USE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    set(CAFFE_USE_OPENMP_FOUND TRUE)
    message(STATUS "[DEP] OpenMP: FOUND (flags: ${OpenMP_CXX_FLAGS})")
  else()
    message(STATUS "[DEP] OpenMP: NOT FOUND - falling back to serial execution")
  endif()
endif()

find_package(Python COMPONENTS Interpreter QUIET)
