cmake_minimum_required(VERSION 3.26)
project(xuanspace LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 注意：根 CMakeLists.txt 仅聚合包含原生扩展的子项目
# 纯 Python 项目不需要在此处添加
# 实际项目按需取消注释或添加子目录：

# add_subdirectory(libs/xuan-ext-demo)
