# DetectBLAS.cmake - BLAS/OpenBLAS 检测与配置
# 提供变量：BLAS_FOUND, BLAS_LIBRARIES, BLAS_INCLUDE_DIRS
#
# 注意：本文件不命名为 FindBLAS.cmake 是为了避免与 CMake 内置 FindBLAS 模块冲突，
# 否则 find_package(BLAS) 会递归调用本文件导致无限循环。

# ── Check if BLAS is explicitly disabled ──
if(DEFINED CAFFE_USE_BLAS AND NOT CAFFE_USE_BLAS)
  set(BLAS_FOUND OFF)
  set(BLAS_LIBRARIES "")
  set(BLAS_INCLUDE_DIRS "")
  message(STATUS "BLAS explicitly disabled (CAFFE_USE_BLAS=OFF) - building with pure C++ fallback")
  return()
endif()

# ── BLAS detection (OpenBLAS via conda or system) ──
set(BLAS_FOUND OFF)
set(BLAS_LIBRARIES "")
set(BLAS_INCLUDE_DIRS "")

# Collect conda-related search hints
set(_blas_search_paths "")

if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _blas_search_paths "$ENV{CONDA_PREFIX}")
endif()
if(CMAKE_PREFIX_PATH)
  list(APPEND _blas_search_paths ${CMAKE_PREFIX_PATH})
endif()
if(Python_SITEARCH)
  # Some conda envs may have blas in unexpected locations
  get_filename_component(_python_prefix "${Python_SITEARCH}/../.." ABSOLUTE)
  list(APPEND _blas_search_paths "${_python_prefix}")
endif()
# On Windows conda, protobuf may be found from a different env than CONDA_PREFIX
# (e.g. py314 env even when CONDA_PREFIX points to base). Add the protobuf prefix
# as an additional search hint.
if(Protobuf_INCLUDE_DIR)
  get_filename_component(_protobuf_env_prefix "${Protobuf_INCLUDE_DIR}/../.." ABSOLUTE)
  if(NOT "${_protobuf_env_prefix}" IN_LIST _blas_search_paths)
    list(APPEND _blas_search_paths "${_protobuf_env_prefix}")
  endif()
endif()

# Platform-specific path suffixes (Windows conda uses Library/ prefix)
if(WIN32)
  set(_blas_include_suffixes include include/openblas Library/include Library/include/openblas)
  set(_blas_lib_names libopenblas openblas)
  set(_blas_lib_suffixes lib lib64 Library/lib Library/bin)
else()
  set(_blas_include_suffixes include include/openblas)
  set(_blas_lib_names openblas openblasp openblas.so.0)
  set(_blas_lib_suffixes lib lib64)
endif()

# ── Phase 1: Targeted search in conda/prefix paths ──
find_path(OPENBLAS_INCLUDE_DIR
  NAMES cblas.h openblas_config.h
  HINTS ${_blas_search_paths}
  PATH_SUFFIXES ${_blas_include_suffixes}
  NO_DEFAULT_PATH
)

find_library(OPENBLAS_LIBRARY
  NAMES ${_blas_lib_names}
  HINTS ${_blas_search_paths}
  PATH_SUFFIXES ${_blas_lib_suffixes}
  NO_DEFAULT_PATH
)

# ── Phase 2: Fallback to system default paths if targeted search fails ──
if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
  message(STATUS "OpenBLAS not found in conda prefix paths, trying system default paths...")
  find_path(OPENBLAS_INCLUDE_DIR
    NAMES cblas.h openblas_config.h
    PATH_SUFFIXES ${_blas_include_suffixes}
  )
  find_library(OPENBLAS_LIBRARY
    NAMES ${_blas_lib_names}
  )
endif()

if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY)
  set(BLAS_FOUND ON)
  set(BLAS_LIBRARIES "${OPENBLAS_LIBRARY}")
  set(BLAS_INCLUDE_DIRS "${OPENBLAS_INCLUDE_DIR}")
  message(STATUS "Found OpenBLAS: ${OPENBLAS_LIBRARY}")
  message(STATUS "OpenBLAS include: ${OPENBLAS_INCLUDE_DIR}")
else()
  message(STATUS "BLAS/OpenBLAS not found - building without BLAS acceleration (will use fallback C++ implementations)")
  if(NOT OPENBLAS_INCLUDE_DIR)
    if(WIN32)
      message(STATUS "  -> cblas.h / openblas_config.h not found (install via: conda install -c conda-forge libopenblas)")
    else()
      message(STATUS "  -> cblas.h / openblas_config.h not found (install libopenblas-dev or openblas-devel)")
    endif()
  endif()
  if(NOT OPENBLAS_LIBRARY)
    if(WIN32)
      message(STATUS "  -> openblas.lib / libopenblas.lib not found (install via: conda install -c conda-forge libopenblas)")
    else()
      message(STATUS "  -> libopenblas.so not found (install libopenblas)")
    endif()
  endif()
endif()
