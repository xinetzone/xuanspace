# Options.cmake - 构建选项与全局编译设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CAFFE_CPU_ONLY "Build Caffe-FFI with CPU only support" ON)
option(CAFFE_FFI_ENABLE_DEBUG_LOG "Enable detailed debug logging for memory/container operations" ON)
option(CAFFE_FFI_ENABLE_BACKTRACE "Enable stack backtrace support for memory leak diagnosis" ON)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
