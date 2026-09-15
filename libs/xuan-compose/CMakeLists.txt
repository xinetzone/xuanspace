# SPDX-License-Identifier: GPL-2.0-only
# xuan-compose - Xuanspace 纯 Python 子项目
# 使用 scikit-build-core 构建（纯 Python 包，LANGUAGES NONE 无需 C++ 编译器）
cmake_minimum_required(VERSION 3.26...3.31 FATAL_ERROR)
project(xuan_compose LANGUAGES NONE)

if(SKBUILD)
  # 将 src/xuan_compose/ 下的 Python 包安装到 wheel 的 xuan_compose/ 目录
  install(DIRECTORY src/xuan_compose/ DESTINATION xuan_compose)
endif()
