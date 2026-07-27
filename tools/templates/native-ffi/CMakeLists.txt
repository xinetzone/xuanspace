cmake_minimum_required(VERSION 3.26)
project({{package_name}} LANGUAGES CXX C)

option({{package_name|upper}}_USE_STUB "Use stub runtime implementation" ON)
option({{package_name|upper}}_FROM_SOURCE "Build tvm_ffi from source" OFF)
set({{package_name|upper}}_DIR "" CACHE PATH "Path to {{module_name}} installation directory")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set({{package_name|upper}}_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

if({{package_name|upper}}_FROM_SOURCE)
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

if(NOT {{package_name|upper}}_USE_STUB)
  message(STATUS "Looking for real {{module_name}} runtime...")

  if({{package_name|upper}}_DIR)
    set({{module_name|upper}}_SEARCH_PATHS "${{{package_name|upper}}_DIR}")
  else()
    set({{module_name|upper}}_SEARCH_PATHS
      /usr/local
      /usr
    )
  endif()

  find_path({{module_name|upper}}_INCLUDE_DIR
    NAMES {{module_name}}/runtime.h
    PATHS ${{{module_name|upper}}_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "{{module_name}} include directory"
  )

  find_library({{module_name|upper}}_RUNTIME_LIBRARY
    NAMES {{module_name}}_runtime
    PATHS ${{{module_name|upper}}_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "{{module_name}} runtime library"
  )

  if({{module_name|upper}}_INCLUDE_DIR AND {{module_name|upper}}_RUNTIME_LIBRARY)
    message(STATUS "Found {{module_name}} include directory: ${{{module_name|upper}}_INCLUDE_DIR}")
    message(STATUS "Found {{module_name}} runtime library: ${{{module_name|upper}}_RUNTIME_LIBRARY}")

    if(NOT TARGET {{module_name|upper}}::{{module_name|upper}})
      add_library({{module_name|upper}}::{{module_name|upper}} SHARED IMPORTED)
      set_target_properties({{module_name|upper}}::{{module_name|upper}} PROPERTIES
        IMPORTED_LOCATION "${{{module_name|upper}}_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${{{module_name|upper}}_INCLUDE_DIR}"
      )
    endif()
    set({{package_name|upper}}_HAS_REAL_{{module_name|upper}} ON)
  else()
    message(FATAL_ERROR "Real {{module_name}} runtime not found. Please set {{package_name|upper}}_DIR or enable {{package_name|upper}}_USE_STUB=ON")
  endif()
else()
  set({{package_name|upper}}_HAS_REAL_{{module_name|upper}} OFF)
endif()

include_directories(${{{package_name|upper}}_INCLUDE_DIR})
add_subdirectory(src)
