cmake_minimum_required(VERSION 3.26)
project(xuanspace LANGUAGES NONE)

# 玄境根 CMakeLists.txt
# 根项目为纯 Python 枢纽包，仅聚合 dev 依赖与 CLI 工具链，无需 C++ 编译器。
# 原生扩展子项目（caffe-ffi/npu-ffi/demo-ffi 等）在各自目录内通过
# scikit-build-core + CMake + Ninja 独立构建，不在此处 add_subdirectory 聚合。

# 根项目无 Python 包体，install(DIRECTORY ...) 留空占位。
# 若未来需要在根级聚合安装纯 Python 包，可在此新增：
# if(SKBUILD)
#   install(DIRECTORY src/xuanspace/ DESTINATION xuanspace)
# endif()