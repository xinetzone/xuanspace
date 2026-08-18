# okf-toolchain - OKF v0.2 工具链
# 使用 scikit-build-core 构建（纯 Python 包，LANGUAGES NONE 无需 C++ 编译器）
# 预留 C/C++ 扩展点：后续挂载原生模块时，在此新增 install(TARGETS ...) 目标
cmake_minimum_required(VERSION 3.26...3.31 FATAL_ERROR)
project(okf LANGUAGES NONE)

if(SKBUILD)
  # 将 src/okf/ 下的 Python 包安装到 wheel 的 okf/ 目录
  install(DIRECTORY src/okf/ DESTINATION okf)
endif()