# xs-cli - Xuanspace CLI 工具
# 使用 scikit-build-core 构建（纯 Python 包，LANGUAGES NONE 无需 C++ 编译器）
# 与 monorepo 中其他子项目保持构建系统统一
cmake_minimum_required(VERSION 3.26...3.31 FATAL_ERROR)
project(xs LANGUAGES NONE)

if(SKBUILD)
  # 将 src/xs/ 下的 Python 包安装到 wheel 的 xs/ 目录
  install(DIRECTORY src/xs/ DESTINATION xs)
endif()
