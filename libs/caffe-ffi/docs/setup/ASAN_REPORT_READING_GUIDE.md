---
title: "caffe-ffi ASan 报告堆栈解读指南"
date: 2026-08-04
tags: [asan, memory, debugging, setup]
source: examples/asan_demo.cpp（Task 17b 演示用例）
---

# caffe-ffi ASan 报告堆栈解读指南

> 本文档服务于 caffe-ffi 的 **Task 17b（ASan 内存管理验证）** 的前置能力，帮助开发者阅读并定位 AddressSanitizer（ASan）报告中的越界访问，尤其是越界写入（buffer overflow）问题。

## 一、ASan 报告核心字段解读

ASan 报告由若干段落组成，核心字段如下：

### 1.1 操作类型与字节数：`WRITE of size N` / `READ of size N`

- `WRITE of size N`：程序对内存执行了**写入**操作，且单次写入 `N` 字节。这是越界写入（heap/stack buffer overflow）的典型标记。
- `READ of size N`：程序对内存执行了**读取**操作，单次读取 `N` 字节。
- 其中 `N` 与元素类型/指令宽度相关。例如对 `int arr[10]` 写入 `arr[10]`，一条 `int` 赋值通常是 `WRITE of size 4`。

### 1.2 地址表示：`0x...` 与越界方向

- `0x...` 是触发访问的内存地址（十六进制）。
- 报告会给出该地址与所属缓冲区边界的关系：
  - `to the right`：地址在缓冲区**右侧**（即高地址方向，数组末尾之后）越界。
  - `to the left`：地址在缓冲区**左侧**（即低地址方向，数组起始之前）越界。
- 通过地址偏移可判断越界距离。例如 `0x... is located 0 bytes to the right of 40-byte region` 表示本次访问恰好落在缓冲区末尾右侧 0 字节处（即越过最后一个元素 1 个元素）。

### 1.3 错误类型

| 错误类型 | 说明 |
|---------|------|
| `heap-buffer-overflow` | 对堆上分配（`new`/`malloc`）的缓冲区越界访问 |
| `stack-buffer-overflow` | 对栈上分配的数组/缓冲区越界访问 |
| `use-after-free` | 在内存被释放（`delete`/`free`）之后再访问 |
| `heap-use-after-free` | 堆上对象释放后仍被访问 |
| `stack-buffer-underflow` | 对栈缓冲区的起始地址之前越界访问 |
| `global-buffer-overflow` | 对全局数组越界访问 |
| `double-free` | 对一个指针重复释放 |

### 1.4 栈帧：`#0 ... #N`

- `#0` 是**触发现场**——即真正执行越界访问的那一层调用（通常是内联的库函数，如 `memcpy`、`std::copy` 或越界赋值本身）。
- `#1`, `#2`, ... `#N` 是**上层调用链**，从触发现场逐层向上回溯到业务代码。
- 定位问题时，**向上找第一个带业务函数名与源码行号的帧**，它通常就是越界写入的源头。

### 1.5 红区（redzone）概念

- ASan 编译时会在每个已分配的缓冲区**周围填充一段不可访问的字节区**，称为 **redzone（红区）**。
- 越界访问会命中红区，从而被 ASan 捕获并触发错误报告。
- 因此，ASan 报告中的 `to the right/left` 方向，实际上反映的是访问地址落入了缓冲区右侧或左侧的 redzone。

## 二、定位越界写入位置的方法

### 2.1 编译期开启调试符号

为让 ASan 报告能给出函数名与源码行号，编译时必须开启：

- `-g`：生成调试符号（debug info）。
- `-fno-omit-frame-pointer`：保留帧指针，保证栈回溯（stack trace）完整。
- `-fsanitize=address`：启用 ASan 插桩。

示例：

```bash
g++ -g -fno-omit-frame-pointer -fsanitize=address -o asan_demo examples/asan_demo.cpp
```

### 2.2 直接从栈帧读取位置

在开启 `-g` 后，报告栈帧会直接给出函数名与源码行号，例如：

```
#1 0x... in heap_overflow_demo() asan_demo.cpp:48
```

含义：越界写入发生在 `asan_demo.cpp` 第 48 行的 `heap_overflow_demo()` 函数内。

### 2.3 仅地址无符号时用 `addr2line` 反查

若栈帧只有地址、没有符号（例如 strip 过的二进制或未装 `-g`），可用 `addr2line` 反查：

```bash
addr2line -e asan_demo 0x401234
```

将输出该地址对应的源码文件和行号，例如：

```
/path/to/examples/asan_demo.cpp:48
```

## 三、真实 ASan 报告示例（逐行解读）

以下示例基于 `examples/asan_demo.cpp` 中的 `heap_overflow_demo()`：函数在堆上分配一个 `size = 10` 的 `int` 数组，然后写入越界的 `arr[10]`（第 11 个元素），触发 `heap-buffer-overflow`。

```text
=================================================================
==2494==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x504000000038 at pc 0x5be64bff1363 bp 0x7fff7b42f240 sp 0x7fff7b42f230
WRITE of size 4 at 0x504000000038 thread T0
    #0 0x5be64bff1362 in heap_overflow_demo() /mnt/d/.../examples/asan_demo.cpp:48
    #1 0x5be64bff1409 in main /mnt/d/.../examples/asan_demo.cpp:67
    #2 0x783f4a82a1c9  (/lib/x86_64-linux-gnu/libc.so.6+0x2a1c9) (BuildId: ...)
    #3 0x783f4a82a28a in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x2a28a) (BuildId: ...)
    #4 0x5be64bff11a4 in _start (/tmp/asan_demo0+0x11a4) (BuildId: ...)

0x504000000038 is located 0 bytes after 40-byte region [0x504000000010,0x504000000038)
allocated by thread T0 here:
    #0 0x783f4b0fe6c8 in operator new[](unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:98
    #1 0x5be64bff131f in heap_overflow_demo() /mnt/d/.../examples/asan_demo.cpp:45
    #2 0x5be64bff1409 in main /mnt/d/.../examples/asan_demo.cpp:67
    #3 0x783f4a82a1c9  (/lib/x86_64-linux-gnu/libc.so.6+0x2a1c9) (BuildId: ...)
    #4 0x783f4a82a28a in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x2a28a) (BuildId: ...)
    #5 0x5be64bff11a4 in _start (/tmp/asan_demo0+0x11a4) (BuildId: ...)

SUMMARY: AddressSanitizer: heap-buffer-overflow /mnt/d/.../examples/asan_demo.cpp:48 in heap_overflow_demo()
Shadow bytes around the buggy address:
  0x503ffffffd80: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x503ffffffe00: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x503ffffffe80: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x503fffffff00: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x503fffffff80: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x504000000000: fa fa 00 00 00 00 00[fa]fa fa fa fa fa fa fa fa
  0x504000000080: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x504000000100: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x504000000180: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x504000000200: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x504000000280: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
==2494==ABORTING
```

### 逐行解读

| 报告行 | 含义 |
|-------|------|
| `==2494==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x504000000038 ...` | 进程 PID 2494 触发错误，类型为 `heap-buffer-overflow`，访问地址为 `0x504000000038`。 |
| `WRITE of size 4 at 0x504000000038 thread T0` | 对地址 `0x504000000038` 执行了 **4 字节写入**（即 `int` 赋值），发生在线程 T0。 |
| `#0 0x5be64bff1362 in heap_overflow_demo() ...asan_demo.cpp:48` | 触发现场：`heap_overflow_demo()` 中 `asan_demo.cpp` 第 48 行（即 `arr[10] = 42;`）。 |
| `#1 0x5be64bff1409 in main ...asan_demo.cpp:67` | 上层调用链：`main` 在 `asan_demo.cpp` 第 67 行调用 `heap_overflow_demo()`。 |
| `#2 / #3 / #4 ...` | 运行时启动与 libc 加载帧，一般无需关注。 |
| `0x504000000038 is located 0 bytes after 40-byte region [0x504000000010,0x504000000038)` | 访问地址落在 **40 字节区域（10 个 int）末尾之后 0 字节处**，即正好越过数组末尾第 1 个元素，落入右 redzone。 |
| `allocated by thread T0 here:` 及 `#0 ... operator new[]` / `#1 ... asan_demo.cpp:45` | 内存分配溯源：该缓冲区在 `asan_demo.cpp` 第 45 行通过 `new int[10]` 分配。 |
| `SUMMARY: AddressSanitizer: heap-buffer-overflow ...asan_demo.cpp:48 in heap_overflow_demo()` | 一行摘要，快速定位错误类型与位置。 |
| `Shadow bytes ... =>0x504000000000: fa fa 00 00 00 00 00[fa]fa fa ...` | 影子内存（shadow memory）视图：`00` 表示可寻址，`fa` 表示堆左 redzone。`[fa]` 标记的正是被越界写入的右 redzone 位置。 |

### 定位结论

- 越界写入发生点：`examples/asan_demo.cpp:48`（`heap_overflow_demo()` 内 `arr[10]` 越界写）。
- 越界方向：`to the right`（数组末尾之后）。
- 修复方式：将越界写改为合法下标（如 `arr[9]`），或把数组扩容为 `size = 11`。

## 四、快速排查清单

1. 查看报告首行 `ERROR: AddressSanitizer: <类型>`，确认错误类型。
2. 查看 `WRITE/READ of size N`，确认是读还是写、字节数。
3. 在栈帧中从 `#0` 向上找第一个带业务函数名与源码行号的帧。
4. 根据 `to the right/left` 与 `located N bytes to the right of M-byte region` 判断越界方向与距离。
5. 如需反查裸地址，执行 `addr2line -e <binary> <address>`。
6. 修复后重新编译并复跑，确认报告消失。