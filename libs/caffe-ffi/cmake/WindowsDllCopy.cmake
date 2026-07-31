# WindowsDllCopy.cmake - Windows 平台运行时 DLL 复制配置
# 提供可复用的 DLL 复制函数，供主库和测试目标共同使用，消除重复代码。
#
# 公共函数：
#   caffe_ffi_copy_dll_if_exists(target dll_path)     - 通用单 DLL 复制
#   caffe_ffi_copy_target_dll(target dep_target)     - 复制指定依赖目标的 DLL（自动处理 WIN32 IMPORTED 反模式）
#   caffe_ffi_copy_tvm_ffi_dll(target)               - 复制 tvm_ffi 共享库
#   caffe_ffi_copy_openblas_dlls(target)             - 复制 OpenBLAS DLLs
#   caffe_ffi_copy_protobuf_dlls(target)             - 复制 Protobuf DLLs
#   caffe_ffi_copy_abseil_dlls(target)               - 复制 abseil DLLs
#   caffe_ffi_copy_utf8_dlls(target)                 - 复制 utf8_range DLLs
#   caffe_ffi_copy_runtime_dlls(target)              - 聚合函数：复制所有运行时依赖 DLLs
#
# 反模式修复（Anti-pattern A1）：
#   WIN32 下许多 CMake 包（如 tvm_ffi-config.cmake）只设置 IMPORTED_IMPLIB（.lib），
#   不设置 IMPORTED_LOCATION（.dll），导致 $<TARGET_FILE:dep> 生成表达式为空，
#   POST_BUILD copy_if_different 静默失败（日志显示"Copying..."但实际未复制）。
#   本模块通过 _caffe_ffi_resolve_imported_dll() 函数自动探测 DLL 路径：
#     1. 优先读取 IMPORTED_LOCATION / IMPORTED_LOCATION_<CONFIG>
#     2. 若缺失，从 IMPORTED_IMPLIB 同目录推算 .dll 文件名
#     3. 验证文件存在性后再复制，避免静默失败

if(MSVC)
  # Collect conda env Library/bin directories as primary search paths.
  # These take precedence over Protobuf_DIR-derived paths because CMake may have
  # found Protobuf in a different conda env than the active Python environment.
  set(_caffe_ffi_conda_dll_dirs "")
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _caffe_ffi_conda_dll_dirs "$ENV{CONDA_PREFIX}/Library/bin")
  endif()
  if(Python3_ROOT_DIR)
    list(APPEND _caffe_ffi_conda_dll_dirs "${Python3_ROOT_DIR}/Library/bin")
  endif()
  if(PYTHON_ROOT_DIR)
    list(APPEND _caffe_ffi_conda_dll_dirs "${PYTHON_ROOT_DIR}/Library/bin")
  endif()
  list(REMOVE_DUPLICATES _caffe_ffi_conda_dll_dirs)

  # 内部辅助宏：校验目标参数
  macro(_caffe_ffi_validate_copy_target target_name func_name)
    if(NOT target_name)
      message(FATAL_ERROR
        "${func_name}(): 缺少必需参数 <target_name>\n"
        "  用法: ${func_name}(<target_name>)"
      )
    endif()
    if(NOT TARGET ${target_name})
      message(FATAL_ERROR
        "${func_name}(): 目标 '${target_name}' 不存在\n"
        "  请确保在调用此函数之前已通过 add_library() 或 add_executable() 创建该目标"
      )
    endif()
  endmacro()

  # ── 核心：WIN32 IMPORTED DLL 路径解析（修复反模式 A1）──
  #
  # 给定一个 IMPORTED SHARED 目标，解析其 WIN32 DLL 的实际路径。
  # 解决仅设 IMPORTED_IMPLIB 不设 IMPORTED_LOCATION 导致 $<TARGET_FILE> 为空的问题。
  #
  # 用法：_caffe_ffi_resolve_imported_dll(dep_target out_dll_path_var)
  #   dep_target:        IMPORTED 目标名（如 tvm_ffi::shared）
  #   out_dll_path_var:  输出变量名，解析成功时设为 DLL 绝对路径；失败时为空
  function(_caffe_ffi_resolve_imported_dll dep_target out_dll_path_var)
    set(_dll_path "")

    # 1. 尝试读取 IMPORTED_LOCATION（非 WIN32 或包配置正确时可用）
    get_target_property(_loc ${dep_target} IMPORTED_LOCATION)
    if(_loc AND EXISTS "${_loc}")
      set(_dll_path "${_loc}")
    endif()

    # 2. 尝试按配置读取 IMPORTED_LOCATION_<CONFIG>（多配置生成器）
    if(NOT _dll_path)
      foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        get_target_property(_loc_cfg ${dep_target} IMPORTED_LOCATION_${_cfg})
        if(_loc_cfg AND EXISTS "${_loc_cfg}")
          set(_dll_path "${_loc_cfg}")
          break()
        endif()
      endforeach()
    endif()

    # 3. 反模式修复：若 IMPORTED_LOCATION 缺失，从 IMPORTED_IMPLIB 推算 DLL 路径
    #    （tvm_ffi-config.cmake 等包的 WIN32 分支只设 .lib 不设 .dll）
    if(NOT _dll_path)
      get_target_property(_implib ${dep_target} IMPORTED_IMPLIB)
      if(_implib AND EXISTS "${_implib}")
        get_filename_component(_dll_dir "${_implib}" DIRECTORY)
        get_filename_component(_dll_name_we "${_implib}" NAME_WE)
        set(_candidate "${_dll_dir}/${_dll_name_we}.dll")
        if(EXISTS "${_candidate}")
          set(_dll_path "${_candidate}")
        else()
          # 某些包 .lib 和 .dll 在不同目录（如 conda: lib/ 下是 .lib，bin/ 下是 .dll）
          get_filename_component(_dll_dir_parent "${_dll_dir}" DIRECTORY)
          set(_candidate2 "${_dll_dir_parent}/bin/${_dll_name_we}.dll")
          if(EXISTS "${_candidate2}")
            set(_dll_path "${_candidate2}")
          endif()
        endif()
      endif()
    endif()

    # 4. 最后尝试按配置 IMPORTED_IMPLIB_<CONFIG> 推算
    if(NOT _dll_path)
      foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        get_target_property(_implib_cfg ${dep_target} IMPORTED_IMPLIB_${_cfg})
        if(_implib_cfg AND EXISTS "${_implib_cfg}")
          get_filename_component(_dll_dir "${_implib_cfg}" DIRECTORY)
          get_filename_component(_dll_name_we "${_implib_cfg}" NAME_WE)
          set(_candidate "${_dll_dir}/${_dll_name_we}.dll")
          if(EXISTS "${_candidate}")
            set(_dll_path "${_candidate}")
            break()
          endif()
          get_filename_component(_dll_dir_parent "${_dll_dir}" DIRECTORY)
          set(_candidate2 "${_dll_dir_parent}/bin/${_dll_name_we}.dll")
          if(EXISTS "${_candidate2}")
            set(_dll_path "${_candidate2}")
            break()
          endif()
        endif()
      endforeach()
    endif()

    set(${out_dll_path_var} "${_dll_path}" PARENT_SCOPE)
  endfunction()

  function(caffe_ffi_copy_dll_if_exists target_name dll_path)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_dll_if_exists")
    if(NOT dll_path)
      message(FATAL_ERROR
        "caffe_ffi_copy_dll_if_exists(): 缺少必需参数 <dll_path>\n"
        "  用法: caffe_ffi_copy_dll_if_exists(<target_name> <dll_path>)"
      )
    endif()
    if(dll_path AND EXISTS "${dll_path}")
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${dll_path}"
          "$<TARGET_FILE_DIR:${target_name}>"
        COMMENT "Copying ${dll_path} to output directory"
      )
    endif()
  endfunction()

  function(caffe_ffi_copy_target_dll target_name dependency_target)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_target_dll")
    if(NOT dependency_target)
      message(FATAL_ERROR
        "caffe_ffi_copy_target_dll(): 缺少必需参数 <dependency_target>\n"
        "  用法: caffe_ffi_copy_target_dll(<target_name> <dependency_target>)"
      )
    endif()
    if(NOT TARGET ${dependency_target})
      message(FATAL_ERROR
        "caffe_ffi_copy_target_dll(): 依赖目标 '${dependency_target}' 不存在"
      )
    endif()

    # 检查目标是否为 IMPORTED
    get_target_property(_is_imported ${dependency_target} IMPORTED)
    if(_is_imported)
      # IMPORTED 目标：使用 _caffe_ffi_resolve_imported_dll 安全解析 DLL 路径
      # （修复反模式 A1：不依赖 $<TARGET_FILE>，避免 IMPORTED_LOCATION 缺失时空拷贝）
      _caffe_ffi_resolve_imported_dll(${dependency_target} _resolved_dll)
      if(_resolved_dll)
        add_custom_command(TARGET ${target_name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_resolved_dll}"
            "$<TARGET_FILE_DIR:${target_name}>"
          COMMENT "Copying ${dependency_target} DLL (${_resolved_dll}) to output directory"
        )
      else()
        # 无法解析 DLL 路径时，回退到 $<TARGET_FILE> 并发出警告
        message(WARNING
          "caffe_ffi_copy_target_dll(): could not resolve DLL path for IMPORTED target "
          "'${dependency_target}'. Falling back to TARGET_FILE generator expression, "
          "which may fail on WIN32 if IMPORTED_LOCATION is not set."
        )
        add_custom_command(TARGET ${target_name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:${dependency_target}>"
            "$<TARGET_FILE_DIR:${target_name}>"
          COMMENT "Copying ${dependency_target} DLL to output directory (fallback)"
        )
      endif()
    else()
      # 非 IMPORTED 目标（本项目构建的目标如 _caffe_ffi）：$<TARGET_FILE> 可靠
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:${dependency_target}>"
          "$<TARGET_FILE_DIR:${target_name}>"
        COMMENT "Copying ${dependency_target} DLL to output directory"
      )
    endif()
  endfunction()

  function(caffe_ffi_copy_tvm_ffi_dll target_name)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_tvm_ffi_dll")
    # 使用通用的 IMPORTED DLL 解析函数处理 tvm_ffi
    caffe_ffi_copy_target_dll(${target_name} tvm_ffi::shared)
  endfunction()

  function(caffe_ffi_copy_openblas_dlls target_name)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_openblas_dlls")
    if(BLAS_FOUND AND BLAS_LIBRARIES)
      foreach(_blas_lib ${BLAS_LIBRARIES})
        get_filename_component(_blas_lib_dir "${_blas_lib}" DIRECTORY)
        file(GLOB _openblas_dlls "${_blas_lib_dir}/../bin/libopenblas*.dll" "${_blas_lib_dir}/../bin/openblas*.dll")
        foreach(_dll ${_openblas_dlls})
          if(EXISTS "${_dll}")
            add_custom_command(TARGET ${target_name} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                "$<TARGET_FILE_DIR:${target_name}>"
              COMMENT "Copying OpenBLAS DLL to output directory"
            )
          endif()
        endforeach()
      endforeach()
    endif()
  endfunction()

  function(caffe_ffi_copy_protobuf_dlls target_name)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_protobuf_dlls")
    # Search conda env dirs first (most reliable), then Protobuf_DIR-derived paths
    set(_protobuf_dll_dirs ${_caffe_ffi_conda_dll_dirs} "${Protobuf_DIR}/../../../bin" "${Protobuf_DIR}/../../bin")
    foreach(_dll_dir ${_protobuf_dll_dirs})
      file(GLOB _protobuf_dlls "${_dll_dir}/libprotobuf*.dll" "${_dll_dir}/libprotoc*.dll" "${_dll_dir}/zlib*.dll")
      foreach(_dll ${_protobuf_dlls})
        if(EXISTS "${_dll}")
          add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_dll}"
              "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying ${_dll} to output directory"
          )
        endif()
      endforeach()
    endforeach()
  endfunction()

  function(caffe_ffi_copy_abseil_dlls target_name)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_abseil_dlls")
    get_filename_component(_absl_dir "${Protobuf_DIR}" DIRECTORY)
    # Search conda env dirs first, then Protobuf_DIR-derived paths
    set(_absl_dll_dirs ${_caffe_ffi_conda_dll_dirs} "${_absl_dir}/bin" "${_absl_dir}/../bin")
    foreach(_dll_dir ${_absl_dll_dirs})
      file(GLOB _absl_dlls "${_dll_dir}/absl_*.dll" "${_dll_dir}/abseil*.dll")
      foreach(_dll ${_absl_dlls})
        if(EXISTS "${_dll}")
          add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_dll}"
              "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying ${_dll} to output directory"
          )
        endif()
      endforeach()
    endforeach()
  endfunction()

  function(caffe_ffi_copy_utf8_dlls target_name)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_utf8_dlls")
    get_filename_component(_absl_dir "${Protobuf_DIR}" DIRECTORY)
    # Search conda env dirs first, then Protobuf_DIR-derived paths
    set(_utf8_dirs ${_caffe_ffi_conda_dll_dirs} "${_absl_dir}/bin" "${_absl_dir}/../bin")
    foreach(_dll_dir ${_utf8_dirs})
      file(GLOB _utf8_dlls "${_dll_dir}/utf8_range*.dll" "${_dll_dir}/utf8_validity*.dll")
      foreach(_dll ${_utf8_dlls})
        if(EXISTS "${_dll}")
          add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_dll}"
              "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying ${_dll} to output directory"
          )
        endif()
      endforeach()
    endforeach()
  endfunction()

  function(caffe_ffi_copy_runtime_dlls target_name)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_runtime_dlls")
    caffe_ffi_copy_tvm_ffi_dll(${target_name})
    caffe_ffi_copy_openblas_dlls(${target_name})
    caffe_ffi_copy_protobuf_dlls(${target_name})
    caffe_ffi_copy_abseil_dlls(${target_name})
    caffe_ffi_copy_utf8_dlls(${target_name})
  endfunction()

  # ── 主库 _caffe_ffi 的运行时 DLL 复制 ──
  caffe_ffi_copy_runtime_dlls(_caffe_ffi)
endif()
