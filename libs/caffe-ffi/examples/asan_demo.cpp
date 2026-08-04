/**
 * @file asan_demo.cpp
 * @brief 自包含的 AddressSanitizer (ASan) 演示程序
 *
 * 演示两类 ASan（AddressSanitizer）可捕获的运行时错误：
 *   1. 内存泄漏（leak_demo）
 *      用 new 分配内存后故意不释放，由 LeakSanitizer 在程序退出时捕获。
 *   2. 堆越界写入（heap_overflow_demo）
 *      用 new[] 分配 int 数组（size=10），然后写入越界索引 arr[10]，
 *      由 ASan 捕获 "heap-buffer-overflow"。
 *
 * 编译运行（要求 g++ >= 5，支持 -fsanitize=address）：
 *   g++ -fsanitize=address -g -O0 -o asan_demo asan_demo.cpp && ./asan_demo
 *
 * 注意：必须使用 -O0（或 -O1 以下）。在 -O1 及以上，gcc 会判定
 * arr[10] = 42 是"死存储"（arr 在 delete[] 前从未被读取）并将其优化掉，
 * 导致 ASan 无法捕获越界写。用 -O0 可防止该优化，确保演示生效。
 *
 * 预期输出：
 *   - heap_overflow_demo 触发 "heap-buffer-overflow" 报告并中止
 *   - 程序退出时 LeakSanitizer 报告 leak_demo 的泄漏
 *
 * 注意：new 与 delete 严格匹配（new→delete、new[]→delete[]），
 * 演示的是"故意不释放（泄漏）"，而非 delete 不匹配。
 */

#include <iostream>

// ---------------------------------------------------------------------------
// 1. 内存泄漏演示：new 分配后故意不释放
// ---------------------------------------------------------------------------
void leak_demo() {
  // new 分配单个 int，随后不再 delete → 指针泄漏，触发 LeakSanitizer
  int* p = new int(42);
  std::cout << "[demo] leak_demo() 分配了 int(42)@" << p
            << " 但故意不释放（将被 LeakSanitizer 报告）\n";
}

// ---------------------------------------------------------------------------
// 2. 堆越界写入演示：new[] 分配后越界索引写入
// ---------------------------------------------------------------------------
void heap_overflow_demo() {
  constexpr int kSize = 10;
  // new[] 分配 10 个 int，合法索引为 [0, 9]
  int* arr = new int[kSize];

  // 越界写入：索引 10 超出 [0, 9]，触发 heap-buffer-overflow
  arr[10] = 42;

  std::cout << "[demo] heap_overflow_demo() 越界写 arr[10] 完成"
            << "（未启用 ASan 时此处才可能执行到）\n";

  // 匹配的 delete[]（ASan 在越界写 arr[10] 时已中止，此行实际不可达）
  delete[] arr;
}

// ---------------------------------------------------------------------------
// 3. 主入口
// ---------------------------------------------------------------------------
int main() {
  std::cout << "=== caffe-ffi ASan 演示 ===\n\n";

  std::cout << "[demo] 调用 leak_demo() ...\n";
  leak_demo();

  std::cout << "\n[demo] 调用 heap_overflow_demo() ...\n";
  heap_overflow_demo();

  std::cout << "\n[demo] 演示结束\n";
  std::cout << "提示：未启用 ASan（-fsanitize=address）时，泄漏与越界不会被检测。\n";
  return 0;
}