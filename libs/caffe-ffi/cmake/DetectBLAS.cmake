# DetectBLAS.cmake - caffe-ffi 项目的 BLAS/OpenBLAS 检测入口
# ================================================================
#
# 本文件是 caffe-ffi 项目对 DetectOpenBLAS.cmake 可复用模块的薄封装。
# 仅包含项目特定的逻辑，核心检测逻辑委托给 DetectOpenBLAS.cmake。
#
# 项目特定行为：
# 1. 使用 CAFFE_USE_BLAS 变量控制启用/禁用（而非通用的 USE_BLAS）
# 2. 从 Protobuf_INCLUDE_DIR 反推 conda 环境前缀作为额外搜索提示
#    （当 CONDA_PREFIX 指向 base 环境而 OpenBLAS 在子环境时）
#
# 提供变量：BLAS_FOUND, BLAS_LIBRARIES, BLAS_INCLUDE_DIRS

# ── 项目特定的禁用检查 ──
if(DEFINED CAFFE_USE_BLAS AND NOT CAFFE_USE_BLAS)
  set(BLAS_FOUND OFF)
  set(BLAS_LIBRARIES "")
  set(BLAS_INCLUDE_DIRS "")
  message(STATUS "BLAS explicitly disabled (CAFFE_USE_BLAS=OFF) - building with pure C++ fallback")
  return()
endif()

# ── 收集项目特定的搜索提示 ──
set(_caffe_extra_hints "")

# Protobuf 可能安装在不同的 conda 环境中，从它的路径反推正确的环境前缀
if(Protobuf_INCLUDE_DIR)
  get_filename_component(_protobuf_prefix "${Protobuf_INCLUDE_DIR}/../.." ABSOLUTE)
  if(NOT "${_protobuf_prefix}" IN_LIST _caffe_extra_hints)
    list(APPEND _caffe_extra_hints "${_protobuf_prefix}")
  endif()
endif()

# ── 委托给可复用模块 ──
include("${CMAKE_CURRENT_LIST_DIR}/DetectOpenBLAS.cmake")
detect_openblas(EXTRA_HINTS ${_caffe_extra_hints})