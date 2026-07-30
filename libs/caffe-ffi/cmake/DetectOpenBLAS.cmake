# DetectOpenBLAS.cmake — 可复用的 OpenBLAS 检测模块
# ============================================================
#
# 跨平台（Windows/Linux/macOS）自动检测 OpenBLAS，支持 conda、系统包管理器、
# 手动编译等多种安装方式。可直接复制到任意项目的 cmake/ 目录使用。
#
# ## 快速使用
#
#     include(DetectOpenBLAS)
#     detect_openblas()
#     target_link_libraries(my_target PRIVATE ${BLAS_LIBRARIES})
#     target_include_directories(my_target PRIVATE ${BLAS_INCLUDE_DIRS})
#
# ## 自定义搜索路径
#
#     detect_openblas(EXTRA_HINTS "/opt/custom" "/usr/local/myblas")
#
# ## 显式禁用 BLAS
#
#     set(USE_BLAS OFF CACHE BOOL "Disable BLAS")
#     detect_openblas()
#
# ## 输出变量
#
#   BLAS_FOUND        — BOOL, 是否找到 OpenBLAS
#   BLAS_LIBRARIES    — 库文件路径
#   BLAS_INCLUDE_DIRS — 头文件目录
#
# ## 支持的安装方式
#
#   | 平台     | 安装方式                          | 自动检测 |
#   |----------|----------------------------------|---------|
#   | Windows  | conda install libopenblas         | ✅      |
#   | Windows  | vcpkg install openblas            | ✅ (Phase 2) |
#   | Linux    | apt install libopenblas-dev       | ✅      |
#   | Linux    | conda install libopenblas         | ✅      |
#   | macOS    | brew install openblas             | ✅ (Phase 2) |
#   | macOS    | conda install libopenblas         | ✅      |
#   | 通用     | 手动编译到 /usr/local             | ✅ (Phase 2) |
#
# ## 检测策略（两阶段）
#
#   Phase 1 — 精确搜索：在 conda/CMAKE_PREFIX_PATH/Python 环境前缀中搜索，
#            使用 NO_DEFAULT_PATH 避免误检系统 BLAS
#   Phase 2 — 系统回退：在系统默认路径中搜索，作为兜底
#
# ## 平台差异
#
#   | 项目         | Windows (conda)          | Linux/macOS              |
#   |-------------|--------------------------|--------------------------|
#   | 头文件路径    | Library/include/openblas  | include/openblas          |
#   | 库名         | libopenblas, openblas    | openblas, openblasp, .so |
#   | 库文件路径    | Library/lib, Library/bin  | lib, lib64               |
#
# ## 注意事项
#
#   - 文件名使用 DetectOpenBLAS.cmake 而非 FindOpenBLAS.cmake，避免与
#     CMake 内置 find_package(OpenBLAS) 产生命名冲突和无限递归
#   - 检测到 BLAS 后不自动链接，由调用方通过 target_link_libraries 显式链接
#   - 未找到时不报 FATAL_ERROR，仅输出 STATUS 消息，由调用方决定是否继续

# ── 公开函数 ──────────────────────────────────────────────

function(detect_openblas)
  # 解析参数
  set(_options "")
  set(_one_value "")
  set(_multi_value EXTRA_HINTS)
  cmake_parse_arguments(_blas "${_options}" "${_one_value}" "${_multi_value}" ${ARGN})

  # ── 显式禁用检查 ──
  if(DEFINED USE_BLAS AND NOT USE_BLAS)
    set(BLAS_FOUND OFF PARENT_SCOPE)
    set(BLAS_LIBRARIES "" PARENT_SCOPE)
    set(BLAS_INCLUDE_DIRS "" PARENT_SCOPE)
    message(STATUS "BLAS explicitly disabled (USE_BLAS=OFF)")
    return()
  endif()

  set(BLAS_FOUND OFF)
  set(BLAS_LIBRARIES "")
  set(BLAS_INCLUDE_DIRS "")

  # ── 收集搜索提示路径 ──
  set(_search_hints "")

  # conda 环境前缀
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _search_hints "$ENV{CONDA_PREFIX}")
  endif()

  # CMake 前缀路径
  if(CMAKE_PREFIX_PATH)
    list(APPEND _search_hints ${CMAKE_PREFIX_PATH})
  endif()

  # Python 安装位置（conda 环境可能通过 Python 路径反推）
  if(Python_SITEARCH)
    get_filename_component(_py_prefix "${Python_SITEARCH}/../.." ABSOLUTE)
    list(APPEND _search_hints "${_py_prefix}")
  endif()

  # 调用方提供的额外提示路径
  if(_blas_EXTRA_HINTS)
    list(APPEND _search_hints ${_blas_EXTRA_HINTS})
  endif()

  # ── 平台特定配置 ──
  if(WIN32)
    set(_include_suffixes include include/openblas Library/include Library/include/openblas)
    set(_lib_names libopenblas openblas)
    set(_lib_suffixes lib lib64 Library/lib Library/bin)
  else()
    set(_include_suffixes include include/openblas)
    set(_lib_names openblas openblasp openblas.so.0)
    set(_lib_suffixes lib lib64)
  endif()

  # ── Phase 1: 在提示路径中精确搜索 ──
  find_path(OPENBLAS_INCLUDE_DIR
    NAMES cblas.h openblas_config.h
    HINTS ${_search_hints}
    PATH_SUFFIXES ${_include_suffixes}
    NO_DEFAULT_PATH
  )

  find_library(OPENBLAS_LIBRARY
    NAMES ${_lib_names}
    HINTS ${_search_hints}
    PATH_SUFFIXES ${_lib_suffixes}
    NO_DEFAULT_PATH
  )

  # ── Phase 2: 系统默认路径回退 ──
  if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
    message(STATUS "OpenBLAS not found in prefix paths, trying system defaults...")
    find_path(OPENBLAS_INCLUDE_DIR
      NAMES cblas.h openblas_config.h
      PATH_SUFFIXES ${_include_suffixes}
    )
    find_library(OPENBLAS_LIBRARY
      NAMES ${_lib_names}
    )
  endif()

  # ── 结果处理 ──
  if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY)
    set(BLAS_FOUND ON PARENT_SCOPE)
    set(BLAS_LIBRARIES "${OPENBLAS_LIBRARY}" PARENT_SCOPE)
    set(BLAS_INCLUDE_DIRS "${OPENBLAS_INCLUDE_DIR}" PARENT_SCOPE)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIBRARY}")
    message(STATUS "OpenBLAS include: ${OPENBLAS_INCLUDE_DIR}")
  else()
    message(STATUS "BLAS/OpenBLAS not found - building without BLAS acceleration")
    if(NOT OPENBLAS_INCLUDE_DIR)
      _blas_print_install_hint("header")
    endif()
    if(NOT OPENBLAS_LIBRARY)
      _blas_print_install_hint("library")
    endif()
  endif()

  # 导出缓存变量供外部检查
  set(BLAS_FOUND "${BLAS_FOUND}" PARENT_SCOPE)
  set(BLAS_LIBRARIES "${BLAS_LIBRARIES}" PARENT_SCOPE)
  set(BLAS_INCLUDE_DIRS "${BLAS_INCLUDE_DIRS}" PARENT_SCOPE)
endfunction()

# ── 内部辅助函数 ──────────────────────────────────────────

function(_blas_print_install_hint _missing)
  if(WIN32)
    message(STATUS "  -> Install via: conda install -c conda-forge libopenblas")
    message(STATUS "  -> Or via: vcpkg install openblas")
  elseif(APPLE)
    message(STATUS "  -> Install via: brew install openblas")
    message(STATUS "  -> Or via: conda install -c conda-forge libopenblas")
  else()
    message(STATUS "  -> Install via: sudo apt install libopenblas-dev (Debian/Ubuntu)")
    message(STATUS "  -> Or via: sudo dnf install openblas-devel (Fedora)")
    message(STATUS "  -> Or via: conda install -c conda-forge libopenblas")
  endif()
endfunction()