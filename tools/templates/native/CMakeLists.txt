# CMake 最低版本要求 (3.26+)
cmake_minimum_required(VERSION 3.26...3.31 FATAL_ERROR)

# 项目定义
project({{package_name}} LANGUAGES CXX)

# C++ 标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 pybind11
find_package(pybind11 CONFIG REQUIRED)

# 查找 Python
find_package(Python 3.14 COMPONENTS Interpreter Development REQUIRED)

# pybind11 扩展模块
pybind11_add_module(_core MODULE
    src/bindings.cpp
)

# 编译选项
target_compile_features(_core PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(_core PRIVATE /utf-8 /W4 /WX)
else()
    target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# 安装规则：将编译后的扩展模块安装到 Python 包目录
install(TARGETS _core LIBRARY DESTINATION {{package_name}})
