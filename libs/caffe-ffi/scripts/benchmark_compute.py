"""P0/P1/P2 分层计算性能 benchmark（Task 31：BLAS 后端 / OpenMP 并行化）。

分三层量化 caffe-ffi 计算密集层的性能收益：
  P0 microbenchmark : 单层 GEMM/卷积/池化/逐元素核的原始吞吐（FLOPs/s）
  P1 layer benchmark: 单层 Forward 的平均耗时（InnerProduct / Pooling / Eltwise）
  P2 network benchmark: 端到端网络 Forward 耗时

用法（在 Linux/WSL 环境，先构建 caffe-ffi 扩展）：
  python scripts/benchmark_compute.py
  # 仅跑 P0 层：
  python scripts/benchmark_compute.py --level P0
  # 控制 OpenMP 线程数（若以 OpenMP 编译）：
  OMP_NUM_THREADS=4 python scripts/benchmark_compute.py

说明：
- 结果受编译配置影响：BLAS(OpenBLAS 多线程) / OpenMP(纯 C++ 并行) / 串行。
- 脚本会打印当前编译配置（BLAS/OpenMP 是否启用），便于对比不同构建。
- 性能须先测量再优化，本脚本为 P4 性能优化提供量化基线。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# 定位编译产物（Linux: build/python/caffe_ffi/_caffe_ffi.so；Windows: .../_caffe_ffi.dll）
import glob

_src_root = os.path.dirname(os.path.abspath(__file__))
_build_jobs = [
    os.path.join(_src_root, "..", "build", "python", "caffe_ffi"),
    os.path.join(_src_root, "..", "build", "Release"),
    os.path.join(_src_root, "..", "build"),
]
_ext_dir = None
for _d in _build_jobs:
    if glob.glob(os.path.join(_d, "_caffe_ffi.*")):
        _ext_dir = _d
        break
if _ext_dir:
    os.environ["PATH"] = _ext_dir + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt":
        os.add_dll_directory(os.path.abspath(_ext_dir))

import caffe_ffi
from caffe_ffi import Net
from caffe_ffi.io import net_param_from_string, net_from_param


def bench_ms(name, fn, warmup=3, repeat=20):
    """返回平均耗时（ms）与 95 分位。"""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


def gflops(flops, ms):
    if ms <= 0:
        return float("inf")
    return flops / (ms * 1e-3) / 1e9


def build_ip_net(in_d, out, batch=1):
    """构造 Input -> InnerProduct 单层网络，batch 作为 M 维（GEMM 并行粒度）。"""
    proto = f"""
name: "bench_ip"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: {batch} dim: {in_d} }} }}
}}
layer {{
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param {{
    num_output: {out}
    weight_filler {{ type: "xavier" }}
    bias_filler {{ type: "constant" }}
  }}
}}
"""
    return net_from_param(net_param_from_string(proto))


def build_pool_net(n, c, h, w, k, s):
    proto = f"""
name: "bench_pool"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: {n} dim: {c} dim: {h} dim: {w} }} }}
}}
layer {{
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pool"
  pooling_param {{
    pool: MAX
    kernel_size: {k}
    stride: {s}
  }}
}}
"""
    return net_from_param(net_param_from_string(proto))


def build_eltwise_net(n, c, h, w):
    proto = f"""
name: "bench_elt"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: {n} dim: {c} dim: {h} dim: {w} }} }}
}}
layer {{
  name: "data2"
  type: "Input"
  top: "data2"
  input_param {{ shape {{ dim: {n} dim: {c} dim: {h} dim: {w} }} }}
}}
layer {{
  name: "elt"
  type: "Eltwise"
  bottom: "data"
  bottom: "data2"
  top: "elt"
  eltwise_param {{ operation: SUM }}
}}
"""
    return net_from_param(net_param_from_string(proto))


def main():
    ap = argparse.ArgumentParser(description="caffe-ffi 计算性能 benchmark")
    ap.add_argument("--level", default="P012", choices=["P0", "P1", "P2", "P012"])
    args = ap.parse_args()

    print("=" * 72)
    print("caffe-ffi Compute Benchmark (P0/P1/P2)")
    print("=" * 72)
    print(f"FFI available: {caffe_ffi._ffi_api.is_available()}")
    print(f"Version: {caffe_ffi.version()}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print()

    # ── P0: GEMM 微基准（InnerProduct 即 MxK @ KxN）──
    if args.level in ("P0", "P012"):
        print("-" * 72)
        print("P0 microbenchmark: GEMM (InnerProduct) FLOPs/s")
        print("-" * 72)
        # batch=16 使 M 维足够大，OpenMP 在 M 维上的并行收益才能体现
        for in_d, out, batch in [(512, 512, 16), (1024, 1024, 16), (2048, 1024, 8), (4096, 1024, 8)]:
            net = build_ip_net(in_d, out, batch)
            # InnerProduct 默认 axis=1，输入须为 2-D [batch, in_d]（非 1-D 腌平列表）
            x = np.random.randn(batch, in_d).astype(np.float32)
            _ = net.Forward({"data": x})
            flops = 2 * batch * in_d * out  # 2*M*N*K
            avg, p95 = bench_ms(f"IP {batch}x{in_d}x{out}", lambda: net.Forward({"data": x}))
            print(f"  InnerProduct {batch}x{in_d}x{out}: avg={avg:8.3f}ms  "
                  f"p95={p95:8.3f}ms  {gflops(flops, avg):8.2f} GFLOPS")
        print()

    # ── P1: 单层 Forward ──
    if args.level in ("P1", "P012"):
        print("-" * 72)
        print("P1 layer benchmark: single-layer Forward")
        print("-" * 72)

        # Pooling（n 维并行，batch 取较大值以体现 OpenMP 收益）
        for n, c, h, w, k, s in [(8, 64, 56, 56, 3, 2), (8, 128, 28, 28, 3, 2), (16, 256, 14, 14, 3, 2)]:
            net = build_pool_net(n, c, h, w, k, s)
            # Pooling 输入须为 4-D [n, c, h, w]
            x = np.random.randn(n, c, h, w).astype(np.float32)
            _ = net.Forward({"data": x})
            avg, p95 = bench_ms(f"Pooling {n}x{c}x{h}x{w} k={k} s={s}", lambda: net.Forward({"data": x}))
            print(f"  Pooling {n}x{c}x{h}x{w} k={k} s={s}: avg={avg:8.3f}ms  p95={p95:8.3f}ms")
        print()

        # Eltwise（count 维并行，元素量足够大以体现 OpenMP 收益）
        for n, c, h, w in [(8, 64, 56, 56), (8, 256, 28, 28), (16, 512, 14, 14)]:
            net = build_eltwise_net(n, c, h, w)
            # Eltwise 输入须为 4-D [n, c, h, w]
            x = np.random.randn(n, c, h, w).astype(np.float32)
            _ = net.Forward({"data": x, "data2": x})
            avg, p95 = bench_ms(f"Eltwise SUM {n}x{c}x{h}x{w}", lambda: net.Forward({"data": x, "data2": x}))
            print(f"  Eltwise SUM {n}x{c}x{h}x{w}: avg={avg:8.3f}ms  p95={p95:8.3f}ms")
        print()

    # ── P2: 端到端 MLP ──
    if args.level in ("P2", "P012"):
        print("-" * 72)
        print("P2 network benchmark: MLP Forward")
        print("-" * 72)
        mlp = """
name: "mlp_bench"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 784 } } }
layer { name: "ip1" type: "InnerProduct" bottom: "data" top: "ip1"
        inner_product_param { num_output: 256 weight_filler { type: "xavier" } bias_filler { type: "constant" } } }
layer { name: "relu1" type: "ReLU" bottom: "ip1" top: "ip1" }
layer { name: "ip2" type: "InnerProduct" bottom: "ip1" top: "ip2"
        inner_product_param { num_output: 10 weight_filler { type: "xavier" } bias_filler { type: "constant" } } }
layer { name: "prob" type: "Softmax" bottom: "ip2" top: "prob" }
"""
        net = net_from_param(net_param_from_string(mlp))
        x = np.random.randn(1, 784).astype(np.float32)
        _ = net.Forward({"data": x})
        avg, p95 = bench_ms("MLP Forward(bs=1)", lambda: net.Forward({"data": x}), warmup=10, repeat=100)
        print(f"  MLP Forward(bs=1): avg={avg:8.3f}ms  p95={p95:8.3f}ms")
        print()

    print("=" * 72)
    print("Benchmark complete. 对比不同构建配置（BLAS on/off、OpenMP on/off）以获得优化收益。")
    print("=" * 72)


if __name__ == "__main__":
    main()