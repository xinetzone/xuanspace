/**
 * @file test_asan_demo.cpp
 * @brief ASan 演示子程序的单元测试（可选，受 CAFFE_FFI_ENABLE_ASAN 守卫）
 *
 * 说明：
 *   本文件仅在 CAFFE_FFI_ENABLE_ASAN 开启时编译，常规构建下不产生任何代码。
 *   由于自研测试框架（test_harness.hpp）在单个进程内顺序运行所有用例，
 *   若直接在用例内触发 heap-buffer-overflow / 泄漏，ASan 会中止整个
 *   caffe_ffi_tests 进程，导致其余用例一并失败。因此本文件不注册"期望崩溃"
 *   的断言，而是仅将 demo 子程序以可独立调用的函数形式暴露，供 ad-hoc 验证：
 *
 *     g++ -fsanitize=address -g -O0 -o asan_demo examples/asan_demo.cpp \
 *         && ./asan_demo
 *
 *   leak_demo / heap_overflow_demo 即由本文件暴露的可独立调用子程序，
 *   与 examples/asan_demo.cpp 中的实现保持一致。
 */

#include "test_harness.hpp"

#ifdef CAFFE_FFI_ENABLE_ASAN

namespace caffe_ffi {
namespace testing {

// 与 examples/asan_demo.cpp 的实现保持一致，独立暴露为可调用函数。
// 注意：调用这些函数会触发 ASan 崩溃，故不在本框架内实际执行（见文件头注释）。

// 内存泄漏：new 分配后故意不释放 → 触发 LeakSanitizer
void leak_demo() {
  int* p = new int(42);
  (void)p;  // 故意不释放；仅用于避免"未使用变量"告警
}

// 堆越界写入：new[] 分配 size=10 后越界写 arr[10] → 触发 heap-buffer-overflow
void heap_overflow_demo() {
  constexpr int kSize = 10;
  int* arr = new int[kSize];
  arr[10] = 42;  // 越界索引写入
  delete[] arr;  // 匹配的 delete[]（ASan 在越界写时已中止，实际不可达）
}

// 仅验证子程序以可调用函数形式暴露（不实际触发崩溃）。
TEST(AsanDemo, SubroutinesExposed) {
  using Fn = void (*)();
  Fn null_fn = nullptr;
  Fn leak = leak_demo;
  Fn overflow = heap_overflow_demo;
  EXPECT_NE(leak, null_fn);
  EXPECT_NE(overflow, null_fn);
}

}  // namespace testing
}  // namespace caffe_ffi

#endif  // CAFFE_FFI_ENABLE_ASAN