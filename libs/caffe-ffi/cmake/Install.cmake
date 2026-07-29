# Install.cmake - 安装规则配置
install(TARGETS _caffe_ffi
  LIBRARY DESTINATION caffe_ffi
  RUNTIME DESTINATION caffe_ffi
)

if(EXISTS "${CAFFE_FFI_PROTO_PY}")
  install(FILES "${CAFFE_FFI_PROTO_PY}" DESTINATION caffe_ffi/proto)
endif()
