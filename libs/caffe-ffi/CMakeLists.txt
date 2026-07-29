cmake_minimum_required(VERSION 3.26)
project(caffe_ffi VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Options)
include(Dependencies)
include(CompilerConfig)
include(ProtoCompile)
include(TargetBuild)
include(WindowsDllCopy)
include(Tests)
include(Install)
