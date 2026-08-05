# P2 Flaky Test 诊断与压力测试指南

## 问题背景

`test_p2_other_ops.py`（Upsample / MemoryData / DummyData / Python / HDF5Output / WindowData）曾出现间歇性失败（flaky），根因是 **C++ 静态回调注册表污染**——`caffe_ffi.data_io.register` 和 `caffe_ffi.python_layer.register` 的回调存储在进程级 `std::unordered_map` 中，跨测试用例残留，导致后续测试中同名 key 的 stale callback 被意外触发。

## 已实施的防御措施

### 1. 回调注册表自动清理（根因修复）

**文件**: `tests/python/conftest.py` → `_clear_callback_registries` fixture (autouse)

每个测试用例执行前后自动调用：
- `caffe_ffi.data_io.clear`
- `caffe_ffi.python_layer.clear`

确保测试间无回调残留。

### 2. Net 状态诊断工具

**文件**: `tests/python/caffe_test_helpers.py`

两个辅助函数：

| 函数 | 用途 |
|------|------|
| `dump_net_state(net, tag)` | 输出所有 blob 的 shape/count/data tensor 大小/diff 状态，以及所有 layer 的 type/param_blobs |
| `make_net_with_diag(prototxt, tag)` | `make_net` 的诊断包装版，构造后自动 dump 状态，返回 `(net, diag_string)` |

**已埋点的8个测试用例**（在 net 构造后/Forward 前/Forward 后三个关键节点调用 `dump_net_state`）：
- `TestUpsample::test_forward_nearest_neighbor`
- `TestUpsample::test_backward_gradient_sums_over_block`
- `TestMemoryData::test_forward_zeros_when_no_data`
- `TestPythonLayer::test_forward_no_op_without_callback`
- `TestPythonLayer::test_forward_invokes_callback`
- `TestHDF5Output::test_forward_invokes_callback_with_bottom`
- `TestHDF5Output::test_forward_no_callback_skips_write`
- `TestWindowData::test_forward_fills_data_and_label`
- `TestWindowData::test_forward_no_callback_zeros`

### 3. 50次紧循环压力测试脚本

**文件**: `scripts/stress_test_p2.py`

## 使用方法

### 快速验证 flaky 是否修复

```bash
# 在 Docker 容器 (caffe-ffi-jupyter) 内执行：
cd /SpecWeave/projects/xuanspace/libs/caffe-ffi
python scripts/stress_test_p2.py            # 默认 50 轮，首次失败即停止
python scripts/stress_test_p2.py -n 100     # 100 轮
python scripts/stress_test_p2.py -k hdf5    # 只跑含 "hdf5" 的测试
python scripts/stress_test_p2.py --no-fail-fast  # 跑完全部轮次不中断
```

### 预期输出

```
[stress] Starting 50 tight-loop iterations...
  [run   1/50] PASS  (2.42s, RSS 120196 KB, Δ+52304 KB)
  ...
  [run  50/50] PASS  (1.51s, RSS 161088 KB, Δ+0 KB)
[stress] ===== SUMMARY =====
[stress] Failures: 0
[stress] ALL RUNS PASSED ✓
```

RSS 应在约15轮后稳定（Δ≈0 KB），表明无内存泄漏。

### 首次失败时的产物

脚本在第一次检测到失败时自动捕获以下产物到 `tests/python/.temp/stress_p2/`：

| 文件 | 内容 |
|------|------|
| `failure_traceback_*_runN.txt` | 完整异常 traceback（含异常类型和消息） |
| `pytest_output_runN.txt` | 该轮 pytest 的完整 stdout/stderr |
| `memory_snapshot_on_failure_runN.txt` | 失败前后 tracemalloc 对比（top 20 内存分配位置） |
| `memory_snapshot_cumulative_runN.txt` | 从第1轮至今的累积内存 diff |
| `net_diag_runN.txt` | 代表性 net 的 blob/layer 状态 dump |
| `rss_info_runN.txt` | RSS 内存记录（before/after/delta/total_elapsed） |

### 启用 C++ 层详细日志

如需查看 C++ 层的 InsertSplits/LayerSetUp/Reshape/Forward 日志：

```bash
CAFFE_FFI_CPP_LOG_LEVEL=0 python -m pytest tests/python/test_p2_other_ops.py -v -s
```

日志级别：0=TRACE, 1=DEBUG, 2=INFO, 3=WARN（默认）, 4=ERROR（最安静）。

### 在新测试中复用诊断工具

```python
from .caffe_test_helpers import make_net, make_net_with_diag, dump_net_state

# 方式1：简单构造+诊断
net, diag = make_net_with_diag(prototxt_str, tag="my_test")

# 方式2：在关键节点手动 dump
net = make_net(prototxt_str)
dump_net_state(net, tag="before_forward")
out = net.Forward(inputs)
dump_net_state(net, tag="after_forward")
```

### 全量回归测试

```bash
python -m pytest tests/python/ --tb=short
# 预期: 2107 passed, 3 skipped
```

## 验证记录

| 日期 | 压力轮次 | 结果 | 全量套件 |
|------|---------|------|---------|
| 2026-08-05 | 50 轮 | 50/50 PASS (78.1s, RSS稳定~161MB) | 2107 passed, 3 skipped, 0 failed (22.0s) |
