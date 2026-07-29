# ProtoCompile.cmake - Protobuf  proto 文件编译配置
set(CAFFE_FFI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CAFFE_FFI_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(CAFFE_FFI_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(CAFFE_FFI_GEN_PROTO_DIR "${CMAKE_BINARY_DIR}/caffe_proto_gen")
file(MAKE_DIRECTORY "${CAFFE_FFI_GEN_PROTO_DIR}")

set(CAFFE_FFI_PROTO_FILE "${CAFFE_FFI_PROTO_DIR}/caffe/proto/caffe.proto")
set(CAFFE_FFI_PROTO_HDR "${CAFFE_FFI_GEN_PROTO_DIR}/caffe/proto/caffe.pb.h")
set(CAFFE_FFI_PROTO_SRC "${CAFFE_FFI_GEN_PROTO_DIR}/caffe/proto/caffe.pb.cc")
set(CAFFE_FFI_PROTO_PY "${CAFFE_FFI_GEN_PROTO_DIR}/caffe/proto/caffe_pb2.py")

if(EXISTS "${CAFFE_FFI_PROTO_FILE}")
  find_program(PROTOBUF_PROTOC protoc REQUIRED)
  file(MAKE_DIRECTORY "${CAFFE_FFI_GEN_PROTO_DIR}/caffe/proto")
  add_custom_command(
    OUTPUT "${CAFFE_FFI_PROTO_SRC}" "${CAFFE_FFI_PROTO_HDR}" "${CAFFE_FFI_PROTO_PY}"
    COMMAND ${PROTOBUF_PROTOC}
      --proto_path=${CAFFE_FFI_PROTO_DIR}
      --cpp_out=${CAFFE_FFI_GEN_PROTO_DIR}
      --python_out=${CAFFE_FFI_GEN_PROTO_DIR}
      caffe/proto/caffe.proto
    DEPENDS "${CAFFE_FFI_PROTO_FILE}"
    COMMENT "Generating caffe.pb.cc/caffe.pb.h/caffe_pb2.py from caffe.proto"
  )
  set(CAFFE_FFI_PROTO_SRCS "${CAFFE_FFI_PROTO_SRC}")
  set(CAFFE_FFI_PROTO_HDRS "${CAFFE_FFI_PROTO_HDR}")
else()
  message(STATUS "caffe.proto not found at ${CAFFE_FFI_PROTO_FILE} - proto generation will be added in Task 2")
  set(CAFFE_FFI_PROTO_SRCS "")
  set(CAFFE_FFI_PROTO_HDRS "")
endif()
