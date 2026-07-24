cmake_minimum_required(VERSION 3.26)
project(xuan_ext_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 注意：这是一个骨架示例，实际项目中需要：
# 1. 使用 find_package(pybind11 CONFIG REQUIRED) 查找 pybind11
# 2. 创建 pybind11 扩展模块：
#    pybind11_add_module(_xuan_ext_demo src/bindings.cpp)
# 3. 或者使用 Python C API 创建原生扩展
# 4. 使用 target_link_libraries 链接依赖库
# 5. 安装扩展模块到正确位置

# 示例扩展模块占位（取消注释以启用）
# add_subdirectory(src)
