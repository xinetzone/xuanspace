cmake_minimum_required(VERSION 3.26)
project(caffe_ffi VERSION 0.1.0 LANGUAGES CXX)

# Disable Unity/Jumbo builds (CMake 4.x enables them by default for MSVC; they break
# template instantiation order for tvm::ffi::Array<ObjectPtr<T>> with SFINAE checks).
set(CMAKE_UNITY_BUILD OFF CACHE BOOL "Unity build" FORCE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Options)
include(Dependencies)
include(CompilerConfig)
include(ProtoCompile)
include(TargetBuild)
include(WindowsDllCopy)
if(CAFFE_FFI_BUILD_TESTS)
  include(Tests)
endif()
include(Install)
