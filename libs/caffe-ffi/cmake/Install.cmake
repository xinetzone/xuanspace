# Install.cmake - 安装规则配置
# 注意：DESTINATION 使用 "."（相对于 CMAKE_INSTALL_PREFIX），
# 因为 scikit-build-core 的 wheel.install-dir = "caffe_ffi" 已经将整个
# CMake install tree 放置到 site-packages/caffe_ffi/，
# 如果这里再写 DESTINATION caffe_ffi 会导致 site-packages/caffe_ffi/caffe_ffi/ 双重嵌套。
install(TARGETS _caffe_ffi
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)

# 安装 protoc 生成的 Python 文件到包目录（与 wheel.packages 中的预生成版本同步）
# wheel.packages 会将 python/caffe_ffi/ 下的文件复制到 install tree 根目录，
# 此处安装到 "." 确保生成版本与预生成版本在同一位置。
if(EXISTS "${CAFFE_FFI_PROTO_PY}")
  install(FILES "${CAFFE_FFI_PROTO_PY}" DESTINATION .)
endif()
