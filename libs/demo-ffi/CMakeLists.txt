cmake_minimum_required(VERSION 3.26)
project(demo_ffi LANGUAGES CXX C)

option(DEMO_FFI_USE_STUB "Use stub runtime implementation" ON)
option(DEMO_FFI_FROM_SOURCE "Build tvm_ffi from source" OFF)
set(DEMO_FFI_DIR "" CACHE PATH "Path to demo installation directory")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(DEMO_FFI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

if(DEMO_FFI_FROM_SOURCE)
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

if(NOT DEMO_FFI_USE_STUB)
  message(STATUS "Looking for real demo runtime...")

  if(DEMO_FFI_DIR)
    set(DEMO_SEARCH_PATHS "${DEMO_FFI_DIR}")
  else()
    set(DEMO_SEARCH_PATHS
      /usr/local
      /usr
    )
  endif()

  find_path(DEMO_INCLUDE_DIR
    NAMES demo/runtime.h
    PATHS ${DEMO_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "demo include directory"
  )

  find_library(DEMO_RUNTIME_LIBRARY
    NAMES demo_runtime
    PATHS ${DEMO_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "demo runtime library"
  )

  if(DEMO_INCLUDE_DIR AND DEMO_RUNTIME_LIBRARY)
    message(STATUS "Found demo include directory: ${DEMO_INCLUDE_DIR}")
    message(STATUS "Found demo runtime library: ${DEMO_RUNTIME_LIBRARY}")

    if(NOT TARGET DEMO::DEMO)
      add_library(DEMO::DEMO SHARED IMPORTED)
      set_target_properties(DEMO::DEMO PROPERTIES
        IMPORTED_LOCATION "${DEMO_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${DEMO_INCLUDE_DIR}"
      )
    endif()
    set(DEMO_FFI_HAS_REAL_DEMO ON)
  else()
    message(FATAL_ERROR "Real demo runtime not found. Please set DEMO_FFI_DIR or enable DEMO_FFI_USE_STUB=ON")
  endif()
else()
  set(DEMO_FFI_HAS_REAL_DEMO OFF)
endif()

include_directories(${DEMO_FFI_INCLUDE_DIR})
add_subdirectory(src)
