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
set(_blas_include_suffixes "openblas" "")

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

# ── Phase 1: Targeted search in conda/prefix paths ──
find_path(OPENBLAS_INCLUDE_DIR
  NAMES cblas.h openblas_config.h
  HINTS ${_blas_search_paths}
  PATH_SUFFIXES include include/openblas
  NO_DEFAULT_PATH
)

find_library(OPENBLAS_LIBRARY
  NAMES openblas openblasp openblas.so.0  # openblasp=pthreads variant, .so.0=runtime soname
  HINTS ${_blas_search_paths}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
)

# ── Phase 2: Fallback to system default paths if targeted search fails ──
if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
  message(STATUS "OpenBLAS not found in conda prefix paths, trying system default paths...")
  find_path(OPENBLAS_INCLUDE_DIR
    NAMES cblas.h openblas_config.h
    PATH_SUFFIXES openblas
  )
  find_library(OPENBLAS_LIBRARY
    NAMES openblas openblasp blas
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
    message(STATUS "  -> cblas.h / openblas_config.h not found (install libopenblas-dev or openblas-devel)")
  endif()
  if(NOT OPENBLAS_LIBRARY)
    message(STATUS "  -> libopenblas.so not found (install libopenblas)")
  endif()
endif()
