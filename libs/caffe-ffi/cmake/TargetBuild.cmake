# TargetBuild.cmake - 主库 _caffe_ffi 目标构建配置

file(GLOB CAFFE_FFI_CORE_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/caffe_ffi/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/caffe_ffi/*.cpp"
)
file(GLOB CAFFE_FFI_LAYER_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/caffe_ffi/layers/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/caffe_ffi/layers/*.cpp"
)
list(APPEND CAFFE_FFI_CORE_SRCS ${CAFFE_FFI_LAYER_SRCS})

add_library(_caffe_ffi SHARED
  ${CAFFE_FFI_CORE_SRCS}
  ${CAFFE_FFI_PROTO_SRCS}
)

tvm_ffi_configure_target(_caffe_ffi LINK_SHARED ON LINK_HEADER ON MSVC_FLAGS ON DEBUG_SYMBOL ON)

set_target_properties(_caffe_ffi PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_caffe_ffi"
)

if(DEFINED SKBUILD_PROJECT_NAME)
  set_target_properties(_caffe_ffi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/caffe_ffi"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/caffe_ffi"
  )
  if(MSVC)
    foreach(config_type Release RELEASE Debug DEBUG RelWithDebInfo)
      set_target_properties(_caffe_ffi PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/python/caffe_ffi"
        LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/python/caffe_ffi"
        ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/python/caffe_ffi"
      )
    endforeach()
  endif()
endif()

# 公共编译配置（include/definitions/options/link）
caffe_ffi_configure_target(_caffe_ffi VISIBILITY PUBLIC)

# tvm_ffi header 链接（主库特有）
target_link_libraries(_caffe_ffi PUBLIC tvm_ffi::header)

if(MSVC)
  set_target_properties(_caffe_ffi PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
  )
endif()
