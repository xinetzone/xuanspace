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

message(STATUS "[caffe_ffi] _caffe_ffi DLL sources: ${CAFFE_FFI_CORE_SRCS}")
list(LENGTH CAFFE_FFI_CORE_SRCS _core_src_count)
message(STATUS "[caffe_ffi] _caffe_ffi DLL source count: ${_core_src_count}")

add_library(_caffe_ffi SHARED
  ${CAFFE_FFI_CORE_SRCS}
  ${CAFFE_FFI_PROTO_SRCS}
)
message(STATUS "[caffe_ffi] _caffe_ffi target created (SHARED library)")

tvm_ffi_configure_target(_caffe_ffi LINK_SHARED ON LINK_HEADER ON MSVC_FLAGS ON DEBUG_SYMBOL ON)
message(STATUS "[caffe_ffi] _caffe_ffi: tvm_ffi_configure_target applied (LINK_SHARED+LINK_HEADER+MSVC_FLAGS+DEBUG_SYMBOL)")

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

# 数据符号导出（WINDOWS_EXPORT_ALL_SYMBOLS 只导出函数，不导出全局变量）
target_compile_definitions(_caffe_ffi PRIVATE CAFFE_FFI_EXPORTS)
message(STATUS "[caffe_ffi] _caffe_ffi DLL: CAFFE_FFI_EXPORTS enabled (data symbol export)")

# COW Phase 2 compile-time switch (PUBLIC because cpu_mutable_data() is inline in blob.hpp)
if(CAFFE_FFI_ENABLE_COW)
  target_compile_definitions(_caffe_ffi PUBLIC CAFFE_FFI_ENABLE_COW)
  message(STATUS "[caffe_ffi] COW optimization: ENABLED (Phase 2)")
else()
  message(STATUS "[caffe_ffi] COW optimization: DISABLED")
endif()

# COW Phase 3 compile-time switch (batch refcount, lazy reshape) — PUBLIC for header inlines
if(CAFFE_FFI_ENABLE_COW_PHASE3)
  target_compile_definitions(_caffe_ffi PUBLIC CAFFE_FFI_ENABLE_COW_PHASE3)
  message(STATUS "[caffe_ffi] COW Phase 3 batch optimizations: ENABLED")
else()
  message(STATUS "[caffe_ffi] COW Phase 3 batch optimizations: DISABLED")
endif()

# tvm_ffi header 链接（主库特有）
target_link_libraries(_caffe_ffi PUBLIC tvm_ffi::header)
message(STATUS "[caffe_ffi] _caffe_ffi links: tvm_ffi::header (PUBLIC)")

if(MSVC)
  set_target_properties(_caffe_ffi PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
  )
  message(STATUS "[caffe_ffi] _caffe_ffi: WINDOWS_EXPORT_ALL_SYMBOLS=TRUE (function symbols)")
else()
  # Export all symbols on Linux/macOS (matches MSVC WINDOWS_EXPORT_ALL_SYMBOLS behavior)
  set_target_properties(_caffe_ffi PROPERTIES
    C_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN FALSE
  )
  message(STATUS "[caffe_ffi] _caffe_ffi: visibility=default (all symbols exported)")
endif()
