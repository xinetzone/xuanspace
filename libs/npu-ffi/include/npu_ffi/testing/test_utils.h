/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

/*!
 * \file npu_ffi/testing/test_utils.h
 * \brief Reusable test utilities and framework for npu-ffi tests.
 *
 * Provides zero-dependency test macros (no gtest required) and
 * helper classes for testing NPU instructions against the stub backend.
 *
 * Usage pattern for new instruction tests:
 * 1. Include this header
 * 2. Define test functions with signature: void test_<instruction_name>()
 * 3. Use ASSERT_* macros for verification
 * 4. Register in main() with TEST_CASE()
 * 5. Group tests by category matching the five-layer FFI architecture
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "npu_ffi/vta/runtime.h"
#include "npu_ffi/vta/buffer.h"
#include "npu_ffi/vta/command_context.h"
#include "npu_ffi/vta/handle.h"
#include "npu_ffi/vta/types.h"

namespace npu_ffi {
namespace vta {
namespace testing {

/*!
 * \brief Simple test registry and runner for zero-dependency C++ tests.
 *
 * Usage:
 *   TestRunner runner;
 *   runner.add_test("category/test_name", test_function);
 *   return runner.run_all() ? 0 : 1;
 */
class TestRunner {
 public:
  /*!
   * \brief Add a test case to the registry.
   * \param name Test name in "category/test_name" format.
   * \param fn Test function (void() signature).
   */
  void add_test(const char* name, std::function<void()> fn) {
    tests_.push_back({name, std::move(fn)});
  }

  /*!
   * \brief Run all registered tests and print results.
   * \return true if all tests passed, false otherwise.
   */
  bool run_all() {
    int passed = 0;
    int failed = 0;
    std::string current_category;

    for (const auto& t : tests_) {
      std::string name(t.name);
      auto slash = name.find('/');
      std::string category = slash != std::string::npos ? name.substr(0, slash) : "misc";
      std::string test_name = slash != std::string::npos ? name.substr(slash + 1) : name;

      if (category != current_category) {
        if (!current_category.empty()) printf("\n");
        printf("%s tests:\n", category.c_str());
        current_category = category;
      }

      printf("  TEST: %s ... ", test_name.c_str());
      bool ok = true;
      try {
        t.fn();
      } catch (const std::exception& e) {
        printf("FAILED with exception: %s\n", e.what());
        ok = false;
      } catch (...) {
        printf("FAILED with unknown exception\n");
        ok = false;
      }

      if (ok) {
        printf("PASSED\n");
        passed++;
      } else {
        failed++;
      }
    }

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    printf("========================================\n");
    return failed == 0;
  }

 private:
  struct TestCase {
    const char* name;
    std::function<void()> fn;
  };
  std::vector<TestCase> tests_;
};

/*!
 * \brief RAII helper that allocates a buffer and ensures cleanup.
 */
class ScopedBuffer {
 public:
  explicit ScopedBuffer(size_t size) : buf_(size) {}
  ~ScopedBuffer() = default;

  ScopedBuffer(const ScopedBuffer&) = delete;
  ScopedBuffer& operator=(const ScopedBuffer&) = delete;

  Buffer& get() { return buf_; }
  void* data() { return buf_.data(); }
  size_t size() const { return buf_.size(); }

 private:
  Buffer buf_;
};

/*!
 * \brief RAII helper that creates a CommandContext for a test scope.
 */
class ScopedContext {
 public:
  explicit ScopedContext(uint32_t wait_cycles = 0) : ctx_(wait_cycles) {}
  ~ScopedContext() = default;

  ScopedContext(const ScopedContext&) = delete;
  ScopedContext& operator=(const ScopedContext&) = delete;

  CommandContext& get() { return ctx_; }
  CommandHandle cmd() { return ctx_.handle(); }

 private:
  CommandContext ctx_;
};

/*!
 * \brief Fill a buffer with a known pattern for verification.
 * \param buf Buffer to fill.
 * \param pattern Byte pattern value.
 */
inline void fill_pattern(Buffer& buf, uint8_t pattern) {
  if (buf.data() == nullptr || buf.size() == 0) return;
  std::memset(buf.data(), pattern, buf.size());
}

/*!
 * \brief Verify a buffer contains a specific byte pattern.
 * \param buf Buffer to check.
 * \param pattern Expected byte pattern.
 * \return true if all bytes match the pattern.
 */
inline bool verify_pattern(const Buffer& buf, uint8_t pattern) {
  if (buf.data() == nullptr) return false;
  const uint8_t* p = static_cast<const uint8_t*>(buf.data());
  for (size_t i = 0; i < buf.size(); i++) {
    if (p[i] != pattern) return false;
  }
  return true;
}

}  // namespace testing
}  // namespace vta
}  // namespace npu_ffi

/*!
 * \defgroup TestAssertionMacros Test Assertion Macros
 * \brief Zero-dependency test assertion macros.
 *
 * These throw const char* on failure, which TestRunner catches.
 * They do not depend on gtest or any external framework.
 * @{
 */

#define TEST_ASSERT_TRUE(cond) \
  do { \
    if (!(cond)) { \
      printf("\n    ASSERT_TRUE failed: %s (line %d)\n", #cond, __LINE__); \
      throw "assertion failed"; \
    } \
  } while (0)

#define TEST_ASSERT_FALSE(cond) TEST_ASSERT_TRUE(!(cond))
#define TEST_ASSERT_EQ(a, b) TEST_ASSERT_TRUE((a) == (b))
#define TEST_ASSERT_NE(a, b) TEST_ASSERT_TRUE((a) != (b))
#define TEST_ASSERT_NOT_NULL(ptr) TEST_ASSERT_TRUE((ptr) != nullptr)
#define TEST_ASSERT_NULL(ptr) TEST_ASSERT_TRUE((ptr) == nullptr)
#define TEST_ASSERT_GE(a, b) TEST_ASSERT_TRUE((a) >= (b))
#define TEST_ASSERT_LE(a, b) TEST_ASSERT_TRUE((a) <= (b))
#define TEST_ASSERT_GT(a, b) TEST_ASSERT_TRUE((a) > (b))
#define TEST_ASSERT_LT(a, b) TEST_ASSERT_TRUE((a) < (b))

/*!
 * \brief Assert that a buffer contains the expected pattern.
 */
#define TEST_ASSERT_PATTERN(buf, pattern) \
  do { \
    if (!::npu_ffi::vta::testing::verify_pattern((buf), (pattern))) { \
      printf("\n    ASSERT_PATTERN failed (line %d)\n", __LINE__); \
      throw "pattern mismatch"; \
    } \
  } while (0)

/*! @} */

/*!
 * \defgroup InstructionTestPattern Instruction Test Pattern Macros
 * \brief Macros for structuring NPU instruction tests.
 *
 * Use these to create consistent test cases for new NPU instructions.
 * Each instruction test should follow the arrange-act-assert pattern:
 * 1. ARRANGE: allocate buffers, create context
 * 2. ACT: issue the instruction(s)
 * 3. ASSERT: verify expected state (synchronize first!)
 *
 * @{
 */

/*!
 * \brief Begin an instruction test with automatic context and cleanup.
 *
 * Creates a CommandContext for the test scope. The context automatically
 * synchronizes when it goes out of scope (RAII).
 *
 * Usage:
 *   TEST_INSTRUCTION_BEGIN(ctx) {
 *     // ... test body using ctx.cmd() ...
 *   } TEST_INSTRUCTION_END
 */
#define TEST_INSTRUCTION_BEGIN(ctx_name) \
  { \
    ::npu_ffi::vta::testing::ScopedContext ctx_name(0); \
    CommandHandle cmd = ctx_name.cmd(); \
    (void)cmd;

#define TEST_INSTRUCTION_END \
  }

/*! @} */
