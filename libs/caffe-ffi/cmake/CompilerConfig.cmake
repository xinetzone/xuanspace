# CompilerConfig.cmake - 公共编译配置函数
# 提供 caffe_ffi_configure_target(target VISIBILITY <PUBLIC|PRIVATE|INTERFACE>) 函数，
# 统一设置 include 目录、编译定义、编译选项、链接库，消除主库和测试目标之间的重复配置。
#
# 用法:
#   caffe_ffi_configure_target(<target_name>
#     [VISIBILITY <PUBLIC|PRIVATE|INTERFACE>]   # 默认 PUBLIC
#   )
#
# 示例:
#   caffe_ffi_configure_target(_caffe_ffi VISIBILITY PUBLIC)
#   caffe_ffi_configure_target(caffe_ffi_tests VISIBILITY PRIVATE)

function(caffe_ffi_configure_target target_name)
  # ── 参数校验 ──
  if(NOT target_name)
    message(FATAL_ERROR
      "caffe_ffi_configure_target(): 缺少必需参数 <target_name>\n"
      "  用法: caffe_ffi_configure_target(<target_name> [VISIBILITY <PUBLIC|PRIVATE|INTERFACE>])\n"
      "  示例: caffe_ffi_configure_target(_caffe_ffi VISIBILITY PUBLIC)"
    )
  endif()

  cmake_parse_arguments(ARG "" "VISIBILITY" "" ${ARGN})

  if(NOT ARG_VISIBILITY)
    set(ARG_VISIBILITY PUBLIC)
  endif()

  # 校验 VISIBILITY 参数合法性
  if(NOT ARG_VISIBILITY MATCHES "^(PUBLIC|PRIVATE|INTERFACE)$")
    message(FATAL_ERROR
      "caffe_ffi_configure_target(): VISIBILITY 参数值无效: '${ARG_VISIBILITY}'\n"
      "  有效值为: PUBLIC, PRIVATE, INTERFACE\n"
      "  主库用 PUBLIC，测试/可执行文件用 PRIVATE"
    )
  endif()

  # 校验目标是否存在
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR
      "caffe_ffi_configure_target(): 目标 '${target_name}' 不存在\n"
      "  请确保在调用此函数之前已通过 add_library() 或 add_executable() 创建该目标\n"
      "  注意: include(CompilerConfig) 必须在 add_library/add_executable 之前执行，"
      "但函数调用必须在目标创建之后"
    )
  endif()

  message(STATUS "[caffe_ffi] Configuring target: ${target_name} (visibility=${ARG_VISIBILITY})")

  # ── Include directories ──
  message(STATUS "[caffe_ffi]   Include dirs: ${CAFFE_FFI_INCLUDE_DIR}; ${CAFFE_FFI_GEN_PROTO_DIR}; ${Protobuf_INCLUDE_DIRS}")
  target_include_directories(${target_name} ${ARG_VISIBILITY}
    "${CAFFE_FFI_INCLUDE_DIR}"
    "${CAFFE_FFI_GEN_PROTO_DIR}"
    "${Protobuf_INCLUDE_DIRS}"
  )
  if(BLAS_INCLUDE_DIRS)
    message(STATUS "[caffe_ffi]   BLAS include: ${BLAS_INCLUDE_DIRS}")
    target_include_directories(${target_name} ${ARG_VISIBILITY} "${BLAS_INCLUDE_DIRS}")
  endif()

  # Compile definitions
  set(_compile_defs_log "CAFFE_FFI_VERSION=${PROJECT_VERSION}; TVM_FFI_USE_BUILTIN_TYPETRAITS")
  target_compile_definitions(${target_name} ${ARG_VISIBILITY}
    CAFFE_FFI_VERSION="${PROJECT_VERSION}"
    TVM_FFI_USE_BUILTIN_TYPETRAITS  # 强制使用 vendor tvm-ffi 内置 TypeTraits，防止自定义特化冲突
  )
  if(CAFFE_CPU_ONLY)
    target_compile_definitions(${target_name} ${ARG_VISIBILITY} CPU_ONLY)
    string(APPEND _compile_defs_log "; CPU_ONLY")
  endif()
  if(CAFFE_FFI_ENABLE_DEBUG_LOG)
    target_compile_definitions(${target_name} ${ARG_VISIBILITY} CAFFE_FFI_ENABLE_DEBUG_LOG)
    string(APPEND _compile_defs_log "; CAFFE_FFI_ENABLE_DEBUG_LOG")
  endif()
  if(CAFFE_FFI_ENABLE_BACKTRACE)
    target_compile_definitions(${target_name} ${ARG_VISIBILITY} CAFFE_FFI_ENABLE_BACKTRACE)
    string(APPEND _compile_defs_log "; CAFFE_FFI_ENABLE_BACKTRACE")
  endif()
  if(BLAS_FOUND OR BLAS_LIBRARIES)
    target_compile_definitions(${target_name} ${ARG_VISIBILITY} CAFFE_USE_BLAS HAVE_CBLAS_H)
    string(APPEND _compile_defs_log "; CAFFE_USE_BLAS; HAVE_CBLAS_H")
  endif()
  message(STATUS "[caffe_ffi]   Compile definitions: ${_compile_defs_log}")

  # Compile options
  if(MSVC)
    message(STATUS "[caffe_ffi]   Compile options (MSVC): /W3 /WX /utf-8")
    target_compile_options(${target_name} ${ARG_VISIBILITY} /W3 /WX /utf-8)
  else()
    message(STATUS "[caffe_ffi]   Compile options (GCC/Clang): -Wall -Wextra -Werror -fvisibility=hidden -fvisibility-inlines-hidden")
    target_compile_options(${target_name} ${ARG_VISIBILITY}
      -Wall -Wextra -Werror -Wno-unused-parameter
      -fvisibility=hidden              # 默认隐藏所有符号，防止 WEAK 符号泄漏
      -fvisibility-inlines-hidden      # 隐藏内联/模板实例化产生的 WEAK 符号
    )
  endif()

  # Link libraries
  set(_link_libs_log "protobuf::libprotobuf; Threads::Threads")
  target_link_libraries(${target_name} ${ARG_VISIBILITY}
    protobuf::libprotobuf
    Threads::Threads
  )
  if(BLAS_LIBRARIES)
    target_link_libraries(${target_name} ${ARG_VISIBILITY} ${BLAS_LIBRARIES})
    string(APPEND _link_libs_log "; BLAS(${BLAS_LIBRARIES})")
  endif()
  if(MSVC)
    target_link_libraries(${target_name} ${ARG_VISIBILITY} DbgHelp.lib)
    string(APPEND _link_libs_log "; DbgHelp.lib")
  endif()
  message(STATUS "[caffe_ffi]   Link libraries: ${_link_libs_log}")

  # GNU linker: 排除所有静态库符号，防止 WEAK 符号多副本冲突
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "[caffe_ffi]   Linker flags (GNU): -Wl,--exclude-libs,ALL")
    target_link_options(${target_name} ${ARG_VISIBILITY}
      -Wl,--exclude-libs,ALL
    )
  endif()
endfunction()
