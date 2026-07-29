#ifndef CAFFE_FFI_ERROR_HPP_
#define CAFFE_FFI_ERROR_HPP_

/*!
 * \file caffe_ffi/error.hpp
 * \brief Typed exception macros for caffe-ffi, mapping to Python exception types via TVM FFI.
 *
 * TVM FFI automatically maps C++ exception kind strings to Python exception classes:
 *   ValueError   → ValueError       (invalid argument values, shape mismatches)
 *   TypeError    → TypeError        (wrong types, undefined tensors)
 *   KeyError     → KeyError         (blob/layer not found by name)
 *   IndexError   → IndexError       (axis out of range)
 *   RuntimeError → RuntimeError     (general runtime failures, I/O errors)
 *   InternalError → RuntimeError    (internal invariant violations, ICHECK)
 *   MemoryError  → MemoryError      (allocation failures)
 *   NotImplementedError → NotImplementedError
 *
 * Use CAFFE_FFI_ICHECK for internal invariants (should never fail if code is correct).
 * Use CAFFE_FFI_CHECK_* for user-facing validation errors that map to specific Python types.
 *
 * Note: The ErrorKind parameter must be a bare identifier (e.g., ValueError), NOT a
 * namespace-qualified name, because TVM_FFI_THROW uses preprocessor stringification (#).
 */

#include <tvm/ffi/error.h>

namespace caffe_ffi {

/*!
 * \brief Exception kind string constants for programmatic use.
 *
 * These match the bare identifiers used in the CAFFE_FFI_CHECK_* macros.
 * Useful when constructing Error objects directly or for kind comparison.
 */
namespace error_kind {
constexpr const char* kValueError = "ValueError";
constexpr const char* kTypeError = "TypeError";
constexpr const char* kKeyError = "KeyError";
constexpr const char* kIndexError = "IndexError";
constexpr const char* kRuntimeError = "RuntimeError";
constexpr const char* kInternalError = "InternalError";
constexpr const char* kMemoryError = "MemoryError";
constexpr const char* kNotImplementedError = "NotImplementedError";
}  // namespace error_kind

}  // namespace caffe_ffi

// ── Internal invariant checks (never fail if code is correct) ──
// These map to InternalError → Python RuntimeError. Use for defensive programming
// against logic bugs, NOT for user input validation.

// (CAFFE_FFI_ICHECK is already provided by TVM_FFI_ICHECK; we keep the TVM_FFI
//  prefix for internal checks to distinguish them from user-facing checks.)

// ── User-facing value checks → Python ValueError ──
// Use for: invalid shapes, out-of-range values, negative dimensions,
// dtype mismatches on user input, invalid model configuration.

#define CAFFE_FFI_CHECK_VALUE(cond) \
  TVM_FFI_CHECK(cond, ValueError)

#define CAFFE_FFI_CHECK_VALUE_EQ(x, y) TVM_FFI_CHECK_EQ(x, y, ValueError)
#define CAFFE_FFI_CHECK_VALUE_NE(x, y) TVM_FFI_CHECK_NE(x, y, ValueError)
#define CAFFE_FFI_CHECK_VALUE_LT(x, y) TVM_FFI_CHECK_LT(x, y, ValueError)
#define CAFFE_FFI_CHECK_VALUE_LE(x, y) TVM_FFI_CHECK_LE(x, y, ValueError)
#define CAFFE_FFI_CHECK_VALUE_GT(x, y) TVM_FFI_CHECK_GT(x, y, ValueError)
#define CAFFE_FFI_CHECK_VALUE_GE(x, y) TVM_FFI_CHECK_GE(x, y, ValueError)

// ── User-facing type checks → Python TypeError ──
// Use for: undefined tensors, wrong dtype categories, type mismatch errors,
// null objects passed across FFI boundary.

#define CAFFE_FFI_CHECK_TYPE(cond) \
  TVM_FFI_CHECK(cond, TypeError)

#define CAFFE_FFI_CHECK_TYPE_EQ(x, y) TVM_FFI_CHECK_EQ(x, y, TypeError)
#define CAFFE_FFI_CHECK_TYPE_NE(x, y) TVM_FFI_CHECK_NE(x, y, TypeError)
#define CAFFE_FFI_CHECK_TYPE_NOTNULL(x) TVM_FFI_CHECK_NOTNULL(x, TypeError)

// ── User-facing key checks → Python KeyError ──
// Use for: blob_by_name(), layer_by_name() lookups, unknown input names,
// missing blob references in model definitions.

#define CAFFE_FFI_CHECK_KEY(cond) \
  TVM_FFI_CHECK(cond, KeyError)

// ── User-facing index checks → Python IndexError ──
// Use for: axis index out of bounds, dimension access violations,
// invalid layer ranges.

#define CAFFE_FFI_CHECK_INDEX(cond) \
  TVM_FFI_CHECK(cond, IndexError)

#define CAFFE_FFI_CHECK_INDEX_EQ(x, y) TVM_FFI_CHECK_EQ(x, y, IndexError)
#define CAFFE_FFI_CHECK_INDEX_NE(x, y) TVM_FFI_CHECK_NE(x, y, IndexError)
#define CAFFE_FFI_CHECK_INDEX_LT(x, y) TVM_FFI_CHECK_LT(x, y, IndexError)
#define CAFFE_FFI_CHECK_INDEX_LE(x, y) TVM_FFI_CHECK_LE(x, y, IndexError)
#define CAFFE_FFI_CHECK_INDEX_GT(x, y) TVM_FFI_CHECK_GT(x, y, IndexError)
#define CAFFE_FFI_CHECK_INDEX_GE(x, y) TVM_FFI_CHECK_GE(x, y, IndexError)

// ── User-facing runtime checks → Python RuntimeError ──
// Use for: file I/O failures, protobuf parse errors, unsupported operations,
// duplicate blob definitions, corrupted model files.

#define CAFFE_FFI_CHECK_RUNTIME(cond) \
  TVM_FFI_CHECK(cond, RuntimeError)

#define CAFFE_FFI_CHECK_RUNTIME_EQ(x, y) TVM_FFI_CHECK_EQ(x, y, RuntimeError)
#define CAFFE_FFI_CHECK_RUNTIME_NE(x, y) TVM_FFI_CHECK_NE(x, y, RuntimeError)

// ── Throw helpers ──

/*!
 * \brief Throw a typed exception with the given error kind.
 * \param ErrorKind A bare identifier (ValueError, TypeError, KeyError, etc.).
 *
 * Usage: CAFFE_FFI_THROW(ValueError) << "Invalid shape: " << shape;
 */
#define CAFFE_FFI_THROW(ErrorKind) \
  TVM_FFI_THROW(ErrorKind)

/*!
 * \brief Log to stderr before throwing (for startup/top-level errors that can't be caught).
 */
#define CAFFE_FFI_LOG_AND_THROW(ErrorKind) \
  TVM_FFI_LOG_AND_THROW(ErrorKind)

#endif  // CAFFE_FFI_ERROR_HPP_
