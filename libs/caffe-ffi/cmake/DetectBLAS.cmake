# DetectBLAS.cmake - BLAS/OpenBLAS 检测与配置
# 提供变量：BLAS_FOUND, BLAS_LIBRARIES, BLAS_INCLUDE_DIRS
#
# 注意：本文件不命名为 FindBLAS.cmake 是为了避免与 CMake 内置 FindBLAS 模块冲突，
# 否则 find_package(BLAS) 会递归调用本文件导致无限循环。

# ── BLAS detection (OpenBLAS via conda or system) ──
set(BLAS_FOUND OFF)
set(BLAS_LIBRARIES "")
set(BLAS_INCLUDE_DIRS "")

# Manual search for OpenBLAS (common in conda environments)
find_path(OPENBLAS_INCLUDE_DIR
  NAMES cblas.h openblas_config.h
  PATHS
    "${CMAKE_PREFIX_PATH}/include"
    "$ENV{CONDA_PREFIX}/Library/include"
    "$ENV{CONDA_PREFIX}/include"
  PATH_SUFFIXES openblas
  NO_DEFAULT_PATH
)
find_library(OPENBLAS_LIBRARY
  NAMES openblas openblas.lib
  PATHS
    "${CMAKE_PREFIX_PATH}/lib"
    "$ENV{CONDA_PREFIX}/Library/lib"
    "$ENV{CONDA_PREFIX}/lib"
  NO_DEFAULT_PATH
)

if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY)
  set(BLAS_FOUND ON)
  set(BLAS_LIBRARIES "${OPENBLAS_LIBRARY}")
  set(BLAS_INCLUDE_DIRS "${OPENBLAS_INCLUDE_DIR}")
  message(STATUS "Found OpenBLAS: ${OPENBLAS_LIBRARY} (include: ${OPENBLAS_INCLUDE_DIR})")
else()
  message(STATUS "BLAS/OpenBLAS not found - building without BLAS acceleration (will use fallback C++ implementations)")
endif()
