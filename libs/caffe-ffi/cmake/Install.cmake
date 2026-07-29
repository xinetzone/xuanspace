# Install.cmake - 安装规则配置
install(TARGETS _caffe_ffi
  LIBRARY DESTINATION caffe_ffi
  RUNTIME DESTINATION caffe_ffi
)

# 安装 protoc 生成的 Python 文件到 caffe_ffi 包目录
# 注意：wheel.packages 中包含 python/caffe_ffi/caffe_pb2.py（预生成版本），
# scikit-build-core 会将 wheel.packages 文件复制到 CMake install 树之上，
# 因此此处安装的生成版本会被预生成版本覆盖——两者应保持同步。
# 若修改了 .proto 文件，请重新运行 protoc 生成并更新源码树中的预生成文件。
if(EXISTS "${CAFFE_FFI_PROTO_PY}")
  install(FILES "${CAFFE_FFI_PROTO_PY}" DESTINATION caffe_ffi)
endif()
