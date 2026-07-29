# WindowsDllCopy.cmake - Windows 平台运行时 DLL 复制配置
# 提供可复用的 DLL 复制函数，供主库和测试目标共同使用，消除重复代码。
#
# 公共函数：
#   caffe_ffi_copy_dll_if_exists(target dll_path)     - 通用单 DLL 复制
#   caffe_ffi_copy_target_dll(target dep_target)     - 复制指定依赖目标的 DLL
#   caffe_ffi_copy_tvm_ffi_dll(target)               - 复制 tvm_ffi 共享库
#   caffe_ffi_copy_openblas_dlls(target)             - 复制 OpenBLAS DLLs
#   caffe_ffi_copy_protobuf_dlls(target)             - 复制 Protobuf DLLs
#   caffe_ffi_copy_abseil_dlls(target)               - 复制 abseil DLLs
#   caffe_ffi_copy_utf8_dlls(target)                 - 复制 utf8_range DLLs
#   caffe_ffi_copy_runtime_dlls(target)              - 聚合函数：复制所有运行时依赖 DLLs

if(MSVC)
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
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:${dependency_target}>"
        "$<TARGET_FILE_DIR:${target_name}>"
      COMMENT "Copying ${dependency_target} DLL to output directory"
    )
  endfunction()

  function(caffe_ffi_copy_tvm_ffi_dll target_name)
    _caffe_ffi_validate_copy_target("${target_name}" "caffe_ffi_copy_tvm_ffi_dll")
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:tvm_ffi::shared>"
        "$<TARGET_FILE_DIR:${target_name}>"
      COMMENT "Copying tvm_ffi shared library to output directory"
    )
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
    set(_protobuf_dll_dirs "${Protobuf_DIR}/../../../bin" "${Protobuf_DIR}/../../bin")
    foreach(_dll_dir ${_protobuf_dll_dirs})
      file(GLOB _protobuf_dlls "${_dll_dir}/libprotobuf*.dll" "${_dll_dir}/libprotoc*.dll")
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
    set(_absl_dll_dirs "${_absl_dir}/bin" "${_absl_dir}/../bin")
    if(DEFINED ENV{CONDA_PREFIX})
      list(APPEND _absl_dll_dirs "$ENV{CONDA_PREFIX}/Library/bin")
    endif()
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
    set(_utf8_dirs "${_absl_dir}/bin" "${_absl_dir}/../bin")
    foreach(_dll_dir ${_utf8_dirs})
      file(GLOB _utf8_dlls "${_dll_dir}/utf8_range*.dll")
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
