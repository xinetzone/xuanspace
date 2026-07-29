#ifndef CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_
#define CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_

/*!
 * \file test_harness.hpp
 * \brief Minimal header-only unit test framework for caffe-ffi C++ tests.
 *
 * Provides gtest-like macros: TEST(), EXPECT_EQ, EXPECT_NE, EXPECT_LT,
 * EXPECT_LE, EXPECT_GT, EXPECT_GE, EXPECT_TRUE, EXPECT_FALSE, EXPECT_THROW.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace caffe_ffi {
namespace testing {

struct TestInfo {
  std::string suite_name;
  std::string test_name;
  std::function<void()> func;
};

struct TestRegistry {
  std::vector<TestInfo> tests;
  int passed = 0;
  int failed = 0;

  static TestRegistry& Instance() {
    static TestRegistry reg;
    return reg;
  }

  void AddTest(const std::string& suite, const std::string& name,
               std::function<void()> func) {
    tests.push_back({suite, name, std::move(func)});
  }

  int RunAll() {
    for (auto& t : tests) {
      std::printf("[ RUN      ] %s.%s\n", t.suite_name.c_str(),
                  t.test_name.c_str());
      std::fflush(stdout);
      try {
        t.func();
        std::printf("[  PASSED  ] %s.%s\n", t.suite_name.c_str(),
                    t.test_name.c_str());
        passed++;
      } catch (const std::exception& e) {
        std::printf("[  FAILED  ] %s.%s\n", t.suite_name.c_str(),
                    t.test_name.c_str());
        std::printf("             Exception: %s\n", e.what());
        failed++;
      } catch (...) {
        std::printf("[  FAILED  ] %s.%s\n", t.suite_name.c_str(),
                    t.test_name.c_str());
        std::printf("             Unknown exception\n");
        failed++;
      }
    }
    std::printf("\n[==========] %d tests ran, %d passed, %d failed\n",
                static_cast<int>(tests.size()), passed, failed);
    return failed > 0 ? 1 : 0;
  }
};

struct TestRegistrar {
  TestRegistrar(const char* suite, const char* name,
                std::function<void()> func) {
    TestRegistry::Instance().AddTest(suite, name, std::move(func));
  }
};

inline int RunAllTests() { return TestRegistry::Instance().RunAll(); }

}  // namespace testing
}  // namespace caffe_ffi

// ---- Macros ----

#define TEST(suite, name)                                                         \
  static void Test_##suite##_##name();                                             \
  static ::caffe_ffi::testing::TestRegistrar                                      \
      registrar_##suite##_##name(#suite, #name, Test_##suite##_##name);            \
  static void Test_##suite##_##name()

#define CAFFE_FFI_TEST_FAIL(msg)                                                   \
  do {                                                                             \
    std::ostringstream _oss;                                                       \
    _oss << msg;                                                                   \
    throw std::runtime_error(_oss.str());                                          \
  } while (0)

#define EXPECT_TRUE(cond)                                                          \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      CAFFE_FFI_TEST_FAIL("EXPECT_TRUE(" #cond ") failed at " __FILE__ ":"         \
                          << __LINE__);                                            \
    }                                                                              \
  } while (0)

#define EXPECT_FALSE(cond) EXPECT_TRUE(!(cond))

#define EXPECT_EQ(a, b)                                                           \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    if (!(_a == _b)) {                                                             \
      CAFFE_FFI_TEST_FAIL("EXPECT_EQ(" #a ", " #b ") failed at " __FILE__ ":"      \
                          << __LINE__ << "\n  Expected: " << _b                    \
                          << "\n  Actual:   " << _a);                              \
    }                                                                              \
  } while (0)

#define EXPECT_NE(a, b)                                                           \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    if (!(_a != _b)) {                                                             \
      CAFFE_FFI_TEST_FAIL("EXPECT_NE(" #a ", " #b ") failed at " __FILE__ ":"      \
                          << __LINE__ << "\n  Both equal: " << _a);                \
    }                                                                              \
  } while (0)

#define EXPECT_LT(a, b)                                                           \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    if (!(_a < _b)) {                                                              \
      CAFFE_FFI_TEST_FAIL("EXPECT_LT(" #a " < " #b ") failed at " __FILE__ ":"     \
                          << __LINE__ << "\n  " << _a << " < " << _b               \
                          << " is false");                                         \
    }                                                                              \
  } while (0)

#define EXPECT_LE(a, b)                                                           \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    if (!(_a <= _b)) {                                                             \
      CAFFE_FFI_TEST_FAIL("EXPECT_LE(" #a " <= " #b ") failed at " __FILE__ ":"    \
                          << __LINE__ << "\n  " << _a << " <= " << _b              \
                          << " is false");                                         \
    }                                                                              \
  } while (0)

#define EXPECT_GT(a, b) EXPECT_LT(b, a)
#define EXPECT_GE(a, b) EXPECT_LE(b, a)

#define EXPECT_NEAR(a, b, abs_err)                                                \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    auto _diff = std::abs(_a - _b);                                               \
    if (!(_diff <= (abs_err))) {                                                   \
      CAFFE_FFI_TEST_FAIL("EXPECT_NEAR(" #a ", " #b ", " #abs_err ") failed at "  \
                          __FILE__ ":" << __LINE__ << "\n  " << _a << " vs "       \
                          << _b << ", diff=" << _diff << " exceeds " << (abs_err));\
    }                                                                              \
  } while (0)

#define EXPECT_THROW(stmt, exception_type)                                        \
  do {                                                                             \
    bool _threw = false;                                                          \
    try {                                                                         \
      stmt;                                                                       \
    } catch (const exception_type&) {                                             \
      _threw = true;                                                              \
    } catch (const std::exception& _e) {                                          \
      CAFFE_FFI_TEST_FAIL("EXPECT_THROW(" #stmt ", " #exception_type               \
                          ") failed at " __FILE__ ":" << __LINE__                  \
                          << ": wrong exception type: " << _e.what());             \
    }                                                                             \
    if (!_threw) {                                                                \
      CAFFE_FFI_TEST_FAIL("EXPECT_THROW(" #stmt ", " #exception_type               \
                          ") failed at " __FILE__ ":" << __LINE__                  \
                          << ": no exception thrown");                             \
    }                                                                              \
  } while (0)

#endif  // CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_
