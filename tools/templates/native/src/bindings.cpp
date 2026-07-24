/**
 * @file bindings.cpp
 * @brief pybind11 绑定示例 - 提供基础的数学运算函数
 * Xuanspace（玄境）原生扩展子项目
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief 整数加法函数
 * @param a 第一个整数
 * @param b 第二个整数
 * @return 两数之和
 */
int add(int a, int b) {
    return a + b;
}

/**
 * @brief 浮点数加法函数
 * @param a 第一个浮点数
 * @param b 第二个浮点数
 * @return 两数之和
 */
double add_f(double a, double b) {
    return a + b;
}

/**
 * @brief pybind11 模块定义
 */
PYBIND11_MODULE(_core, m) {
    m.doc() = "{{package_name}} 原生扩展核心模块";

    m.def("add", &add,
          "整数加法函数",
          py::arg("a"),
          py::arg("b"));

    m.def("add_f", &add_f,
          "浮点数加法函数",
          py::arg("a"),
          py::arg("b"));

    m.attr("__version__") = "0.1.0";
}
