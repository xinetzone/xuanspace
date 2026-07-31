#ifndef CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_
#define CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_

/*!
 * \file test_harness.hpp
 * \brief Minimal header-only unit test framework for caffe-ffi C++ tests.
 *
 * Provides gtest-like macros: TEST(), EXPECT_EQ, EXPECT_NE, EXPECT_LT,
 * EXPECT_LE, EXPECT_GT, EXPECT_GE, EXPECT_TRUE, EXPECT_FALSE, EXPECT_NEAR,
 * EXPECT_FLOAT_EQ, EXPECT_STREQ, EXPECT_THROW, ASSERT_EQ, ASSERT_TRUE, etc.
 * Supports gtest-style streaming: EXPECT_EQ(a,b) << "extra message";
 *
 * The AssertHelper class and comparison primitives are provided by
 * <caffe_ffi/utils/assert_helper.hpp> (shared with production CHECK macros).
 * This header adds test-only EXPECT_*/ASSERT_* macros and the test registry.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "caffe_ffi/utils/assert_helper.hpp"

namespace caffe_ffi {
namespace testing {

namespace detail {

// Reuse shared AssertHelper and comparison primitives from utils
using ::caffe_ffi::utils::AssertHelper;
using ::caffe_ffi::utils::detail::CmpEq;
using ::caffe_ffi::utils::detail::CmpNe;
using ::caffe_ffi::utils::detail::CmpLt;
using ::caffe_ffi::utils::detail::CmpLe;
using ::caffe_ffi::utils::detail::LocMsg;

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

  static TestRegistry& Instance() {
    static TestRegistry reg;
    return reg;
  }

  void AddTest(const std::string& suite, const std::string& name,
               std::function<void()> func) {
    tests.push_back({suite, name, std::move(func), 0.0, false});
  }

  int RunAll(const char* filter = nullptr) {
    using Clock = std::chrono::high_resolution_clock;
    auto total_start = Clock::now();

    std::string filter_str(filter ? filter : "");
    int passed = 0;
    int failed = 0;
    int ran = 0;
    std::vector<TestInfo*> ran_tests;

    for (auto& t : tests) {
      t.passed_flag = false;
      t.elapsed_ms = 0.0;

      if (!filter_str.empty()) {
        std::string full_name = t.suite_name + "." + t.test_name;
        if (full_name.find(filter_str) == std::string::npos &&
            t.suite_name.find(filter_str) == std::string::npos) {
          continue;
        }
      }
      ran++;
      ran_tests.push_back(&t);

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

    if (!filter_str.empty()) {
      std::printf("\n[==========] %d tests ran (filter: '%s'), %d passed, %d failed (%.2f ms total)\n",
                  ran, filter_str.c_str(), passed, failed, total_ms);
    } else {
      std::printf("\n[==========] %d tests ran, %d passed, %d failed (%.2f ms total)\n",
                  ran, passed, failed, total_ms);
    }

    std::unordered_map<std::string, std::pair<int, double>> suite_stats;
    for (const auto* t : ran_tests) {
      auto& s = suite_stats[t->suite_name];
      s.first++;
      s.second += t->elapsed_ms;
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

    std::vector<const TestInfo*> sorted_tests(ran_tests.begin(), ran_tests.end());
    std::sort(sorted_tests.begin(), sorted_tests.end(),
              [](const TestInfo* a, const TestInfo* b) { return a->elapsed_ms > b->elapsed_ms; });
    size_t top_n = std::min<size_t>(5, sorted_tests.size());
    if (top_n > 0) {
      std::printf("[----------] Top %zu slowest test(s):\n", top_n);
      for (size_t i = 0; i < top_n; i++) {
        const auto* t = sorted_tests[i];
        std::printf("[  SLOW    ] #%zu %s.%s (%.2f ms)\n",
                    i + 1, t->suite_name.c_str(), t->test_name.c_str(), t->elapsed_ms);
      }
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

inline int RunAllTests(const char* filter = nullptr) {
  return TestRegistry::Instance().RunAll(filter);
}

}  // namespace testing
}  // namespace caffe_ffi

// ---- Test Macros ----
//
// All EXPECT_* macros use the IIFE (Immediately-Invoked Function Expression)
// pattern: each macro expands to a lambda that is immediately called, returning
// an AssertHelper temporary. The temporary lives until the semicolon, supporting
// gtest-style << streaming:
//
//   EXPECT_NEAR(x, y, eps) << "at index " << i;
//
// The lambda approach avoids dangling-else issues and works correctly in all
// contexts (loops, if/else without braces, etc.).
//
// Test macros reuse the shared AssertHelper from caffe_ffi/utils/assert_helper.hpp
// and the comparison primitives (CmpEq, CmpNe, etc.) defined there.

#define TEST(suite, name)                                                         \
  static void Test_##suite##_##name();                                             \
  static ::caffe_ffi::testing::TestRegistrar                                      \
      registrar_##suite##_##name(#suite, #name, Test_##suite##_##name);            \
  static void Test_##suite##_##name()

// Test-local aliases (separate from production CAFFE_FFI_CHECK_* macros)
#define CAFFE_FFI_PASS() ::caffe_ffi::testing::detail::AssertHelper(false)
#define CAFFE_FFI_FAIL(msg) ::caffe_ffi::testing::detail::AssertHelper(true, msg)
#define CAFFE_FFI_TEST_LOC ::caffe_ffi::testing::detail::LocMsg(__FILE__, __LINE__)

#define EXPECT_TRUE(cond) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    if (cond) return CAFFE_FFI_PASS(); \
    return CAFFE_FFI_FAIL(std::string("EXPECT_TRUE(" #cond ") failed at ") + CAFFE_FFI_TEST_LOC); \
  }()

#define EXPECT_FALSE(cond) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    if (!(cond)) return CAFFE_FFI_PASS(); \
    return CAFFE_FFI_FAIL(std::string("EXPECT_FALSE(" #cond ") failed at ") + CAFFE_FFI_TEST_LOC); \
  }()

#define EXPECT_EQ(a, b) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::testing::detail::CmpEq(_a, _b)) return CAFFE_FFI_PASS(); \
    std::ostringstream _oss; \
    _oss << "EXPECT_EQ(" #a ", " #b ") failed at " << CAFFE_FFI_TEST_LOC \
         << "\n  Expected: " << _b << "\n  Actual:   " << _a; \
    return CAFFE_FFI_FAIL(_oss.str()); \
  }()

#define EXPECT_NE(a, b) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::testing::detail::CmpNe(_a, _b)) return CAFFE_FFI_PASS(); \
    std::ostringstream _oss; \
    _oss << "EXPECT_NE(" #a ", " #b ") failed at " << CAFFE_FFI_TEST_LOC \
         << "\n  Both equal: " << _a; \
    return CAFFE_FFI_FAIL(_oss.str()); \
  }()

#define EXPECT_LT(a, b) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::testing::detail::CmpLt(_a, _b)) return CAFFE_FFI_PASS(); \
    std::ostringstream _oss; \
    _oss << "EXPECT_LT(" #a " < " #b ") failed at " << CAFFE_FFI_TEST_LOC \
         << "\n  " << _a << " < " << _b << " is false"; \
    return CAFFE_FFI_FAIL(_oss.str()); \
  }()

#define EXPECT_LE(a, b) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::testing::detail::CmpLe(_a, _b)) return CAFFE_FFI_PASS(); \
    std::ostringstream _oss; \
    _oss << "EXPECT_LE(" #a " <= " #b ") failed at " << CAFFE_FFI_TEST_LOC \
         << "\n  " << _a << " <= " << _b << " is false"; \
    return CAFFE_FFI_FAIL(_oss.str()); \
  }()

#define EXPECT_GT(a, b) EXPECT_LT(b, a)
#define EXPECT_GE(a, b) EXPECT_LE(b, a)

#define EXPECT_NEAR(a, b, abs_err) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    auto _diff = std::abs(_a - _b); \
    if (_diff <= (abs_err)) return CAFFE_FFI_PASS(); \
    std::ostringstream _oss; \
    _oss << "EXPECT_NEAR(" #a ", " #b ", " #abs_err ") failed at " << CAFFE_FFI_TEST_LOC \
         << "\n  " << _a << " vs " << _b << ", diff=" << _diff << " exceeds " << (abs_err); \
    return CAFFE_FFI_FAIL(_oss.str()); \
  }()

#define EXPECT_FLOAT_EQ(a, b) EXPECT_NEAR(a, b, 1e-4)

#define EXPECT_STREQ(a, b) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    const char* _a = (a); const char* _b = (b); \
    if (_a != nullptr && _b != nullptr && std::strcmp(_a, _b) == 0) return CAFFE_FFI_PASS(); \
    std::ostringstream _oss; \
    _oss << "EXPECT_STREQ(" #a ", " #b ") failed at " << CAFFE_FFI_TEST_LOC \
         << "\n  Expected: " << (_b ? _b : "nullptr") \
         << "\n  Actual:   " << (_a ? _a : "nullptr"); \
    return CAFFE_FFI_FAIL(_oss.str()); \
  }()

#define EXPECT_THROW(stmt, exception_type) \
  [&]() -> ::caffe_ffi::testing::detail::AssertHelper { \
    bool _threw = false; \
    try { stmt; } \
    catch (const exception_type&) { _threw = true; } \
    catch (const std::exception& _e) { \
      std::ostringstream _oss; \
      _oss << "EXPECT_THROW(" #stmt ", " #exception_type ") failed at " << CAFFE_FFI_TEST_LOC \
           << ": wrong exception type: " << _e.what(); \
      return CAFFE_FFI_FAIL(_oss.str()); \
    } \
    if (!_threw) { \
      std::ostringstream _oss; \
      _oss << "EXPECT_THROW(" #stmt ", " #exception_type ") failed at " << CAFFE_FFI_TEST_LOC \
           << ": no exception thrown"; \
      return CAFFE_FFI_FAIL(_oss.str()); \
    } \
    return CAFFE_FFI_PASS(); \
  }()

// ASSERT_* macros (equivalent to EXPECT in this exception-based framework)
#define ASSERT_TRUE(cond) EXPECT_TRUE(cond)
#define ASSERT_FALSE(cond) EXPECT_FALSE(cond)
#define ASSERT_EQ(a, b) EXPECT_EQ(a, b)
#define ASSERT_NE(a, b) EXPECT_NE(a, b)
#define ASSERT_LT(a, b) EXPECT_LT(a, b)
#define ASSERT_LE(a, b) EXPECT_LE(a, b)
#define ASSERT_GT(a, b) EXPECT_GT(a, b)
#define ASSERT_GE(a, b) EXPECT_GE(a, b)
#define ASSERT_NEAR(a, b, abs_err) EXPECT_NEAR(a, b, abs_err)
#define ASSERT_FLOAT_EQ(a, b) EXPECT_FLOAT_EQ(a, b)
#define ASSERT_STREQ(a, b) EXPECT_STREQ(a, b)
#define ASSERT_THROW(stmt, exception_type) EXPECT_THROW(stmt, exception_type)

#endif  // CAFFE_FFI_TESTS_CPP_TEST_HARNESS_HPP_
