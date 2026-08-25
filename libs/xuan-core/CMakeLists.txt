# xuan-core - Xuanspace 核心工具库
# 使用 scikit-build-core 构建（纯 Python 包，LANGUAGES NONE 无需 C++ 编译器）
cmake_minimum_required(VERSION 3.26...3.31 FATAL_ERROR)
project(xuan_core LANGUAGES NONE)

if(SKBUILD)
  # 将 src/xuan_core/ 下的 Python 包安装到 wheel 的 xuan_core/ 目录
  install(DIRECTORY src/xuan_core/ DESTINATION xuan_core)
endif()