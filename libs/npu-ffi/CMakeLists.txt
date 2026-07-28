cmake_minimum_required(VERSION 3.26)
project(npu_ffi LANGUAGES CXX C)

option(NPU_FFI_VTA_USE_STUB "Use stub VTA runtime implementation" ON)
option(NPU_FFI_FROM_SOURCE "Build tvm_ffi from source" OFF)
option(NPU_FFI_ENABLE_PROTOBUF "Enable Protobuf for config serialization" OFF)
option(NPU_FFI_BUILD_TESTS "Build tests" ON)
option(NPU_FFI_BUILD_EXAMPLES "Build examples" ON)
set(NPU_FFI_VTA_DIR "" CACHE PATH "Path to VTA installation directory (containing include/ and lib/)")

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(NPU_FFI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

if(NPU_FFI_FROM_SOURCE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --sourcedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE tvm_ffi_ROOT
  )
  message(STATUS "Building tvm_ffi from source: ${tvm_ffi_ROOT}")
  add_subdirectory(${tvm_ffi_ROOT} tvm_ffi)
else()
  find_package(tvm_ffi CONFIG REQUIRED)
endif()

if(NOT NPU_FFI_VTA_USE_STUB)
  message(STATUS "Looking for real VTA runtime...")
  
  if(NPU_FFI_VTA_DIR)
    set(VTA_SEARCH_PATHS "${NPU_FFI_VTA_DIR}")
  else()
    set(VTA_SEARCH_PATHS
      /usr/local
      /usr
      "${CMAKE_SOURCE_DIR}/../external/chaos/npu_tvm/vta"
      "${CMAKE_SOURCE_DIR}/../../external/chaos/npu_tvm/vta"
    )
  endif()

  find_path(VTA_INCLUDE_DIR
    NAMES vta/runtime/runtime.h
    PATHS ${VTA_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "VTA include directory"
  )

  find_library(VTA_RUNTIME_LIBRARY
    NAMES vta_runtime VTARuntime vta
    PATHS ${VTA_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "VTA runtime library"
  )

  if(VTA_INCLUDE_DIR AND VTA_RUNTIME_LIBRARY)
    message(STATUS "Found VTA include directory: ${VTA_INCLUDE_DIR}")
    message(STATUS "Found VTA runtime library: ${VTA_RUNTIME_LIBRARY}")
    
    if(NOT TARGET VTA::VTA)
      add_library(VTA::VTA SHARED IMPORTED)
      set_target_properties(VTA::VTA PROPERTIES
        IMPORTED_LOCATION "${VTA_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${VTA_INCLUDE_DIR}"
      )
    endif()
    set(NPU_FFI_HAS_REAL_VTA ON)
  else()
    message(FATAL_ERROR "Real VTA runtime not found. Please set NPU_FFI_VTA_DIR or enable NPU_FFI_VTA_USE_STUB=ON")
  endif()
else()
  set(NPU_FFI_HAS_REAL_VTA OFF)
endif()

if(NPU_FFI_ENABLE_PROTOBUF)
  find_package(Protobuf CONFIG QUIET)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf QUIET)
  endif()
  if(Protobuf_FOUND)
    if(Protobuf_VERSION VERSION_LESS "7.0.0")
      message(FATAL_ERROR "Protobuf version >= 7.0.0 required, found ${Protobuf_VERSION}")
    endif()
    message(STATUS "Found Protobuf: ${Protobuf_VERSION}")
    set(NPU_FFI_HAS_PROTOBUF ON)
  else()
    message(STATUS "Protobuf not found - protobuf serialization will not be available in C++")
    set(NPU_FFI_HAS_PROTOBUF OFF)
  endif()
else()
  message(STATUS "Protobuf support disabled (NPU_FFI_ENABLE_PROTOBUF=OFF)")
  set(NPU_FFI_HAS_PROTOBUF OFF)
endif()

add_subdirectory(src)

if(NPU_FFI_HAS_PROTOBUF)
  add_subdirectory(proto)
endif()

if(NPU_FFI_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(NPU_FFI_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
