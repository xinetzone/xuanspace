#ifndef CAFFE_FFI_UTILS_ASSERT_HELPER_HPP_
#define CAFFE_FFI_UTILS_ASSERT_HELPER_HPP_

/*!
 * \file assert_helper.hpp
 * \brief Reusable IIFE-based assertion helper with gtest-style streaming messages.
 *
 * Provides AssertHelper temporary-object pattern for CHECK/EXPECT macros that
 * support << streaming:
 *
 *   CAFFE_FFI_CHECK(x > 0) << "x must be positive, got " << x;
 *   CAFFE_FFI_CHECK_EQ(a, b) << "mismatch at index " << i;
 *
 * Design:
 * - Each macro expands to an immediately-invoked lambda (IIFE) returning an
 *   AssertHelper temporary.
 * - The temporary lives until the semicolon, collecting any << streamed messages.
 * - On destruction, if the assertion failed, it throws std::runtime_error with
 *   the full message.
 * - Move constructor transfers ownership (ostringstream is move-only in C++17+);
 *   copy is deleted to prevent double-throw.
 *
 * This avoids the classic do{}while(0) limitation (no return value → no streaming)
 * and is safe in all contexts (if/else without braces, loops, etc.).
 *
 * Two tiers of macros:
 *   1. CAFFE_FFI_CHECK_*  — for production code (always active, throws on failure)
 *   2. EXPECT_*/ASSERT_*  — for test code (defined in test_harness.hpp)
 *
 * Usage for defining new assertion macros:
 *
 *   #define MY_CHECK(cond) \
 *     [&]() -> ::caffe_ffi::utils::AssertHelper { \
 *       if (cond) return ::caffe_ffi::utils::AssertHelper(false); \
 *       return ::caffe_ffi::utils::AssertHelper(true, \
 *         std::string("MY_CHECK failed at ") + __FILE__ + ":" + std::to_string(__LINE__)); \
 *     }()
 */

#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace caffe_ffi {
namespace utils {

class AssertHelper {
 public:
  explicit AssertHelper(bool failed) : failed_(failed) {}
  AssertHelper(bool failed, std::string msg) : failed_(failed), msg_(std::move(msg)) {}

  ~AssertHelper() noexcept(false) {
    if (failed_) {
      throw std::runtime_error(msg_ + oss_.str());
    }
  }

  AssertHelper(const AssertHelper&) = delete;
  AssertHelper& operator=(const AssertHelper&) = delete;
  AssertHelper& operator=(AssertHelper&&) = delete;

  AssertHelper(AssertHelper&& other) noexcept
      : failed_(other.failed_),
        msg_(std::move(other.msg_)),
        oss_(std::move(other.oss_)) {
    other.failed_ = false;
  }

  template <typename T>
  AssertHelper& operator<<(const T& val) {
    if (failed_) oss_ << val;
    return *this;
  }

  AssertHelper& operator<<(std::ostream& (*manip)(std::ostream&)) {
    if (failed_) oss_ << manip;
    return *this;
  }

 private:
  bool failed_;
  std::string msg_;
  std::ostringstream oss_;
};

namespace detail {

// Location string helper
inline std::string LocMsg(const char* file, int line) {
  return std::string(file) + ":" + std::to_string(line);
}

// Comparison helpers (use common_type to avoid signed/unsigned comparison warnings)
template <typename T, typename U>
constexpr bool CmpEq(const T& a, const U& b) {
  using C = std::common_type_t<T, U>;
  return static_cast<C>(a) == static_cast<C>(b);
}

template <typename T, typename U>
constexpr bool CmpNe(const T& a, const U& b) {
  return !CmpEq(a, b);
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

template <typename T, typename U>
constexpr bool CmpGt(const T& a, const U& b) {
  return CmpLt(b, a);
}

template <typename T, typename U>
constexpr bool CmpGe(const T& a, const U& b) {
  return CmpLe(b, a);
}

}  // namespace detail
}  // namespace utils
}  // namespace caffe_ffi

// ── Production-code CHECK macros (always active) ──
//
// These use the fully qualified ::caffe_ffi::utils::AssertHelper type so they
// can be used from any namespace without additional using-declarations.
// Unlike gtest's EXPECT_*, these throw immediately (like assert/CHECK in glog),
// making them suitable for invariant checks in runtime code paths.

#define CAFFE_FFI_CHECK_PASS() ::caffe_ffi::utils::AssertHelper(false)
#define CAFFE_FFI_CHECK_FAIL(msg) ::caffe_ffi::utils::AssertHelper(true, msg)
#define CAFFE_FFI_LOC ::caffe_ffi::utils::detail::LocMsg(__FILE__, __LINE__)

#define CAFFE_FFI_CHECK(cond) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    if (cond) return CAFFE_FFI_CHECK_PASS(); \
    return CAFFE_FFI_CHECK_FAIL( \
        std::string("CHECK failed: ") + #cond + " at " + CAFFE_FFI_LOC); \
  }()

#define CAFFE_FFI_CHECK_MSG(cond, msg) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    if (cond) return CAFFE_FFI_CHECK_PASS(); \
    return CAFFE_FFI_CHECK_FAIL( \
        std::string("CHECK failed: ") + (msg) + " at " + CAFFE_FFI_LOC); \
  }()

#define CAFFE_FFI_CHECK_EQ(a, b) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::utils::detail::CmpEq(_a, _b)) return CAFFE_FFI_CHECK_PASS(); \
    std::ostringstream _oss; \
    _oss << "CHECK_EQ(" #a ", " #b ") failed at " << CAFFE_FFI_LOC \
         << "\n  Expected: " << _b << "\n  Actual:   " << _a; \
    return CAFFE_FFI_CHECK_FAIL(_oss.str()); \
  }()

#define CAFFE_FFI_CHECK_NE(a, b) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::utils::detail::CmpNe(_a, _b)) return CAFFE_FFI_CHECK_PASS(); \
    std::ostringstream _oss; \
    _oss << "CHECK_NE(" #a ", " #b ") failed at " << CAFFE_FFI_LOC \
         << "\n  Both equal: " << _a; \
    return CAFFE_FFI_CHECK_FAIL(_oss.str()); \
  }()

#define CAFFE_FFI_CHECK_LT(a, b) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::utils::detail::CmpLt(_a, _b)) return CAFFE_FFI_CHECK_PASS(); \
    std::ostringstream _oss; \
    _oss << "CHECK_LT(" #a " < " #b ") failed at " << CAFFE_FFI_LOC \
         << "\n  " << _a << " < " << _b << " is false"; \
    return CAFFE_FFI_CHECK_FAIL(_oss.str()); \
  }()

#define CAFFE_FFI_CHECK_LE(a, b) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    if (::caffe_ffi::utils::detail::CmpLe(_a, _b)) return CAFFE_FFI_CHECK_PASS(); \
    std::ostringstream _oss; \
    _oss << "CHECK_LE(" #a " <= " #b ") failed at " << CAFFE_FFI_LOC \
         << "\n  " << _a << " <= " << _b << " is false"; \
    return CAFFE_FFI_CHECK_FAIL(_oss.str()); \
  }()

#define CAFFE_FFI_CHECK_GT(a, b) CAFFE_FFI_CHECK_LT(b, a)
#define CAFFE_FFI_CHECK_GE(a, b) CAFFE_FFI_CHECK_LE(b, a)

#define CAFFE_FFI_CHECK_NEAR(a, b, abs_err) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    auto _a = (a); auto _b = (b); \
    auto _diff = std::abs(_a - _b); \
    if (_diff <= (abs_err)) return CAFFE_FFI_CHECK_PASS(); \
    std::ostringstream _oss; \
    _oss << "CHECK_NEAR(" #a ", " #b ", " #abs_err ") failed at " << CAFFE_FFI_LOC \
         << "\n  " << _a << " vs " << _b << ", diff=" << _diff \
         << " exceeds " << (abs_err); \
    return CAFFE_FFI_CHECK_FAIL(_oss.str()); \
  }()

#define CAFFE_FFI_CHECK_NOTNULL(ptr) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    if ((ptr) != nullptr) return CAFFE_FFI_CHECK_PASS(); \
    return CAFFE_FFI_CHECK_FAIL( \
        std::string("CHECK_NOTNULL(" #ptr ") failed at ") + CAFFE_FFI_LOC); \
  }()

#define CAFFE_FFI_CHECK_THROW(stmt, exception_type) \
  [&]() -> ::caffe_ffi::utils::AssertHelper { \
    bool _threw = false; \
    try { stmt; } \
    catch (const exception_type&) { _threw = true; } \
    catch (const std::exception& _e) { \
      std::ostringstream _oss; \
      _oss << "CHECK_THROW(" #stmt ", " #exception_type ") failed at " << CAFFE_FFI_LOC \
           << ": wrong exception type: " << _e.what(); \
      return CAFFE_FFI_CHECK_FAIL(_oss.str()); \
    } \
    if (!_threw) { \
      std::ostringstream _oss; \
      _oss << "CHECK_THROW(" #stmt ", " #exception_type ") failed at " << CAFFE_FFI_LOC \
           << ": no exception thrown"; \
      return CAFFE_FFI_CHECK_FAIL(_oss.str()); \
    } \
    return CAFFE_FFI_CHECK_PASS(); \
  }()

#endif  // CAFFE_FFI_UTILS_ASSERT_HELPER_HPP_
