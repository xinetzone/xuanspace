# proto/CMakeLists.txt - Protobuf code generation for VTA config schema
#
# This directory is only included when NPU_FFI_ENABLE_PROTOBUF=ON and Protobuf is found.
# Since the stub runtime does not require C++ protobuf bindings, this is completely optional.
# Python bindings are pre-generated and shipped with the package (see python/npu_ffi/vta/vta_config_pb2.py).

if(NOT Protobuf_FOUND)
  message(WARNING "Protobuf not found, skipping C++ proto generation")
  return()
endif()

set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROTO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY "${PROTO_GENERATED_DIR}")

set(VTA_PROTO_FILE "${PROTO_SRC_DIR}/vta_config.proto")

# Generate C++ sources using protobuf_generate
if(COMMAND protobuf_generate)
  protobuf_generate(
    LANGUAGE cpp
    OUT_VAR VTA_PROTO_SRCS
    PROTOS "${VTA_PROTO_FILE}"
    PROTOC_OUT_DIR "${PROTO_GENERATED_DIR}"
    IMPORT_DIRS "${PROTO_SRC_DIR}"
  )
else()
  # Fallback: manual protoc invocation
  add_custom_command(
    OUTPUT "${PROTO_GENERATED_DIR}/vta_config.pb.cc" "${PROTO_GENERATED_DIR}/vta_config.pb.h"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out="${PROTO_GENERATED_DIR}" -I"${PROTO_SRC_DIR}" "${VTA_PROTO_FILE}"
    DEPENDS "${VTA_PROTO_FILE}"
    COMMENT "Generating C++ protobuf bindings for vta_config.proto"
    VERBATIM
  )
  set(VTA_PROTO_SRCS "${PROTO_GENERATED_DIR}/vta_config.pb.cc")
endif()

# Generate Python bindings via custom target (optional, for regeneration during build)
add_custom_target(npu_ffi_gen_py_proto
  COMMAND ${Python_EXECUTABLE} -m grpc_tools.protoc
    -I"${PROTO_SRC_DIR}"
    --python_out="${CMAKE_SOURCE_DIR}/python/npu_ffi/vta"
    "${VTA_PROTO_FILE}"
  DEPENDS "${VTA_PROTO_FILE}"
  COMMENT "Regenerating Python protobuf bindings for vta_config.proto"
  VERBATIM
)

# Header-only interface library for consumers that need protobuf config in C++
add_library(npu_ffi_proto INTERFACE)
target_sources(npu_ffi_proto INTERFACE ${VTA_PROTO_SRCS})
target_include_directories(npu_ffi_proto INTERFACE "${PROTO_GENERATED_DIR}")
target_link_libraries(npu_ffi_proto INTERFACE protobuf::libprotobuf)

message(STATUS "VTA config protobuf C++ bindings will be generated at: ${PROTO_GENERATED_DIR}")
