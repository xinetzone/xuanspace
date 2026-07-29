#ifndef CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_
#define CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_

/*!
 * \file test_harness.hpp
 * \brief Minimal header-only unit test framework for caffe-ffi C++ tests.
 *
 * Provides gtest-like macros: TEST(), EXPECT_EQ, EXPECT_NE, EXPECT_LT,
 * EXPECT_LE, EXPECT_GT, EXPECT_GE, EXPECT_TRUE, EXPECT_FALSE, EXPECT_THROW.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace caffe_ffi {
namespace testing {

// ── Type-safe comparison helpers ──
// These resolve signed/unsigned mismatch warnings by promoting both operands
// to their common type before comparison, mirroring the usual arithmetic
// conversions in a well-defined, warning-free manner.
namespace detail {

template <typename T, typename U>
constexpr bool CmpEq(const T& a, const U& b) {
  using C = std::common_type_t<T, U>;
  return static_cast<C>(a) == static_cast<C>(b);
}

template <typename T, typename U>
constexpr bool CmpNe(const T& a, const U& b) {
  using C = std::common_type_t<T, U>;
  return static_cast<C>(a) != static_cast<C>(b);
}

template <typename T, typename U>
constexpr bool CmpLt(const T& a, const U& b) {
  using C = std::common_type_t<T, U>;
  return static_cast<C>(a) < static_cast<C>(b);
}

template <typename T, typename U>
constexpr bool CmpLe(const T& a, const U& b) {
  using C = std::common_type_t<T, U>;
  return static_cast<C>(a) <= static_cast<C>(b);
}

}  // namespace detail

struct TestInfo {
  std::string suite_name;
  std::string test_name;
  std::function<void()> func;
  double elapsed_ms = 0.0;
  bool passed_flag = false;
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
    tests.push_back({suite, name, std::move(func), 0.0, false});
  }

  int RunAll() {
    using Clock = std::chrono::high_resolution_clock;
    auto total_start = Clock::now();

    for (auto& t : tests) {
      std::printf("[ RUN      ] %s.%s\n", t.suite_name.c_str(),
                  t.test_name.c_str());
      std::fflush(stdout);
      auto t_start = Clock::now();
      try {
        t.func();
        auto t_end = Clock::now();
        t.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        t.passed_flag = true;
        std::printf("[  PASSED  ] %s.%s (%.2f ms)\n", t.suite_name.c_str(),
                    t.test_name.c_str(), t.elapsed_ms);
        passed++;
      } catch (const std::exception& e) {
        auto t_end = Clock::now();
        t.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        t.passed_flag = false;
        std::printf("[  FAILED  ] %s.%s (%.2f ms)\n", t.suite_name.c_str(),
                    t.test_name.c_str(), t.elapsed_ms);
        std::printf("             Exception: %s\n", e.what());
        failed++;
      } catch (...) {
        auto t_end = Clock::now();
        t.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        t.passed_flag = false;
        std::printf("[  FAILED  ] %s.%s (%.2f ms)\n", t.suite_name.c_str(),
                    t.test_name.c_str(), t.elapsed_ms);
        std::printf("             Unknown exception\n");
        failed++;
      }
    }

    auto total_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::printf("\n[==========] %d tests ran, %d passed, %d failed (%.2f ms total)\n",
                static_cast<int>(tests.size()), passed, failed, total_ms);

    // ── Per-suite timing summary ──
    std::unordered_map<std::string, std::pair<int, double>> suite_stats;
    for (const auto& t : tests) {
      auto& s = suite_stats[t.suite_name];
      s.first++;
      s.second += t.elapsed_ms;
    }
    if (!suite_stats.empty()) {
      std::printf("[----------] Global test environment tear-down\n");
      std::printf("[==========] Per-suite summary:\n");
      std::vector<std::pair<std::string, std::pair<int, double>>> sorted_suites(
          suite_stats.begin(), suite_stats.end());
      std::sort(sorted_suites.begin(), sorted_suites.end(),
                [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
      for (const auto& s : sorted_suites) {
        std::printf("[  SUITE   ] %-20s %3d tests, %8.2f ms total, avg %6.2f ms\n",
                    s.first.c_str(), s.second.first, s.second.second,
                    s.second.second / s.second.first);
      }
    }

    // ── Slowest individual tests ──
    std::vector<const TestInfo*> sorted_tests;
    sorted_tests.reserve(tests.size());
    for (const auto& t : tests) sorted_tests.push_back(&t);
    std::sort(sorted_tests.begin(), sorted_tests.end(),
              [](const TestInfo* a, const TestInfo* b) { return a->elapsed_ms > b->elapsed_ms; });
    size_t top_n = std::min<size_t>(5, sorted_tests.size());
    std::printf("[----------] Top %zu slowest test(s):\n", top_n);
    for (size_t i = 0; i < top_n; i++) {
      const auto* t = sorted_tests[i];
      std::printf("[  SLOW    ] #%zu %s.%s (%.2f ms)\n",
                  i + 1, t->suite_name.c_str(), t->test_name.c_str(), t->elapsed_ms);
    }

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
    if (!::caffe_ffi::testing::detail::CmpEq(_a, _b)) {                            \
      CAFFE_FFI_TEST_FAIL("EXPECT_EQ(" #a ", " #b ") failed at " __FILE__ ":"      \
                          << __LINE__ << "\n  Expected: " << _b                    \
                          << "\n  Actual:   " << _a);                              \
    }                                                                              \
  } while (0)

#define EXPECT_NE(a, b)                                                           \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    if (!::caffe_ffi::testing::detail::CmpNe(_a, _b)) {                            \
      CAFFE_FFI_TEST_FAIL("EXPECT_NE(" #a ", " #b ") failed at " __FILE__ ":"      \
                          << __LINE__ << "\n  Both equal: " << _a);                \
    }                                                                              \
  } while (0)

#define EXPECT_LT(a, b)                                                           \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    if (!::caffe_ffi::testing::detail::CmpLt(_a, _b)) {                            \
      CAFFE_FFI_TEST_FAIL("EXPECT_LT(" #a " < " #b ") failed at " __FILE__ ":"     \
                          << __LINE__ << "\n  " << _a << " < " << _b               \
                          << " is false");                                         \
    }                                                                              \
  } while (0)

#define EXPECT_LE(a, b)                                                           \
  do {                                                                             \
    auto _a = (a);                                                                 \
    auto _b = (b);                                                                 \
    if (!::caffe_ffi::testing::detail::CmpLe(_a, _b)) {                            \
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
