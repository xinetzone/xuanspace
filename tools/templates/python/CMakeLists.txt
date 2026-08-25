# {{name}} - Xuanspace 纯 Python 子项目
# 使用 scikit-build-core 构建（纯 Python 包，LANGUAGES NONE 无需 C++ 编译器）
cmake_minimum_required(VERSION 3.26...3.31 FATAL_ERROR)
project({{package_name}} LANGUAGES NONE)

if(SKBUILD)
  # 将 src/{{package_name}}/ 下的 Python 包安装到 wheel 的 {{package_name}}/ 目录
  install(DIRECTORY src/{{package_name}}/ DESTINATION {{package_name}})
endif()