"""Performance benchmark for Split/Concat nested topologies.

Measures InsertSplits graph transformation overhead and forward inference time
for Inception-style Split→Concat→Split nested networks at various scales.

Scenarios tested:
  - Baseline: linear chain (no splits, 0 split overhead)
  - Inception-2: 2-branch parallel → concat
  - Inception-4: 4-branch parallel → concat
  - Inception-8: 8-branch parallel → concat
  - Deep nested: 3-level Split→Concat→Split chain (Inception-v3 style)
  - Multi-split: 4 independent fan-out points in one network

Each scenario measures:
  - Net construction time (includes InsertSplits pass)
  - Forward inference time (avg over N runs, warmup excluded)
  - Number of auto-inserted Split layers
  - Memory delta (construction + forward)

Run with:
    pytest tests/python/test_split_concat_bench.py -v -s
    # Or as a script for standalone benchmarking:
    python tests/python/test_split_concat_bench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure caffe_ffi is importable
_project_root = Path(__file__).resolve().parent.parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from .conftest import require_cpp_extension, perf_trace  # noqa: E402
from .caffe_test_helpers import make_net, count_splits, assert_finite  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Prototxt builders
# ──────────────────────────────────────────────────────────────────────

def _inception_prototxt(
    name: str,
    num_branches: int,
    dim_in: int = 64,
    dim_branch: int = 16,
    batch: int = 4,
) -> str:
    """Build an Inception-style prototxt with *num_branches* parallel FC branches.

    Topology:
        data (dim_in) → split → fc_0 ... fc_{n-1} (each dim_branch)
                       → concat (axis=1, n*dim_branch features)
                       → split → out_a, out_b
    """
    lines = [f"name: '{name}'",
             f"input: 'data'",
             f"input_shape {{ dim: {batch} dim: {dim_in} }}"]
    for i in range(num_branches):
        lines.append(
            f"layer {{ name: 'fc_{i}' type: 'InnerProduct' "
            f"bottom: 'data' top: 'branch_{i}' "
            f"inner_product_param {{ num_output: {dim_branch} bias_term: false }} }}"
        )
    bottoms = " ".join(f"bottom: 'branch_{i}'" for i in range(num_branches))
    lines.append(
        f"layer {{ name: 'cat' type: 'Concat' {bottoms} top: 'cat_out' "
        f"concat_param {{ axis: 1 }} }}"
    )
    # Two consumers after concat → triggers split for cat_out
    lines.append(
        f"layer {{ name: 'out_a' type: 'InnerProduct' bottom: 'cat_out' top: 'out_a' "
        f"inner_product_param {{ num_output: {dim_branch} bias_term: false }} }}"
    )
    lines.append(
        f"layer {{ name: 'out_b' type: 'InnerProduct' bottom: 'cat_out' top: 'out_b' "
        f"inner_product_param {{ num_output: {dim_branch} bias_term: false }} }}"
    )
    return "\n".join(lines)


def _linear_chain_prototxt(
    name: str,
    num_layers: int,
    dim: int = 64,
    batch: int = 4,
) -> str:
    """Build a linear chain (no fan-out, zero splits)."""
    lines = [f"name: '{name}'",
             f"input: 'data'",
             f"input_shape {{ dim: {batch} dim: {dim} }}"]
    prev = "data"
    for i in range(num_layers):
        top = f"fc_{i}"
        lines.append(
            f"layer {{ name: 'fc_{i}' type: 'InnerProduct' "
            f"bottom: '{prev}' top: '{top}' "
            f"inner_product_param {{ num_output: {dim} bias_term: false }} }}"
        )
        prev = top
    return "\n".join(lines)


def _deep_nested_prototxt(
    name: str,
    levels: int = 3,
    branches_per_level: int = 2,
    dim: int = 32,
    dim_branch: int = 8,
    batch: int = 2,
) -> str:
    """Build a deeply nested Split→Concat→Split chain (Inception-v3 style).

    Each level: input splits into *branches_per_level* FC branches → concat.
    The concat output feeds into the next level (and also a side output),
    creating a split at every level's output.
    """
    lines = [f"name: '{name}'",
             f"input: 'data'",
             f"input_shape {{ dim: {batch} dim: {dim} }}"]
    prev_top = "data"
    for lv in range(levels):
        # Branches
        for b in range(branches_per_level):
            lines.append(
                f"layer {{ name: 'lv{lv}_fc{b}' type: 'InnerProduct' "
                f"bottom: '{prev_top}' top: 'lv{lv}_b{b}' "
                f"inner_product_param {{ num_output: {dim_branch} bias_term: false }} }}"
            )
        # Concat
        b_bottoms = " ".join(f"bottom: 'lv{lv}_b{b}'" for b in range(branches_per_level))
        cat_top = f"lv{lv}_cat"
        lines.append(
            f"layer {{ name: 'lv{lv}_cat' type: 'Concat' {b_bottoms} top: '{cat_top}' "
            f"concat_param {{ axis: 1 }} }}"
        )
        # Side output (creates fan-out: next level + side out → split for cat_top)
        side_top = f"lv{lv}_side"
        lines.append(
            f"layer {{ name: 'lv{lv}_side' type: 'InnerProduct' "
            f"bottom: '{cat_top}' top: '{side_top}' "
            f"inner_product_param {{ num_output: {dim_branch} bias_term: false }} }}"
        )
        prev_top = cat_top
    # Final output layer
    lines.append(
        f"layer {{ name: 'final' type: 'InnerProduct' "
        f"bottom: '{prev_top}' top: 'final_out' "
        f"inner_product_param {{ num_output: {dim_branch} bias_term: false }} }}"
    )
    return "\n".join(lines)


def _multi_fanout_prototxt(
    name: str,
    num_fanout_points: int = 4,
    consumers_per_point: int = 2,
    dim: int = 32,
    batch: int = 2,
) -> str:
    """Build a network with *num_fanout_points* independent fan-out points.

    Each fan-out point is an FC layer whose output goes to *consumers_per_point*
    parallel FC layers, creating that many Split insertions.
    """
    lines = [f"name: '{name}'",
             f"input: 'data'",
             f"input_shape {{ dim: {batch} dim: {dim} }}"]
    prev_top = "data"
    for fp in range(num_fanout_points):
        # Producer layer
        prod_name = f"prod_{fp}"
        prod_top = f"prod_{fp}_out"
        lines.append(
            f"layer {{ name: '{prod_name}' type: 'InnerProduct' "
            f"bottom: '{prev_top}' top: '{prod_top}' "
            f"inner_product_param {{ num_output: {dim} bias_term: false }} }}"
        )
        # Consumer layers (fan-out from prod_top)
        for c in range(consumers_per_point):
            cons_name = f"cons_{fp}_{c}"
            cons_top = f"cons_{fp}_{c}_out"
            lines.append(
                f"layer {{ name: '{cons_name}' type: 'InnerProduct' "
                f"bottom: '{prod_top}' top: '{cons_top}' "
                f"inner_product_param {{ num_output: {dim} bias_term: false }} }}"
            )
        # Concat all consumer outputs to chain to next fan-out point
        c_bottoms = " ".join(
            f"bottom: 'cons_{fp}_{c}_out'" for c in range(consumers_per_point)
        )
        cat_top = f"cat_{fp}"
        lines.append(
            f"layer {{ name: 'cat_{fp}' type: 'Concat' {c_bottoms} top: '{cat_top}' "
            f"concat_param {{ axis: 1 }} }}"
        )
        # Reduce dim back for next layer with a 1-consumer FC (no split)
        reduce_top = f"reduce_{fp}"
        lines.append(
            f"layer {{ name: 'reduce_{fp}' type: 'InnerProduct' "
            f"bottom: '{cat_top}' top: '{reduce_top}' "
            f"inner_product_param {{ num_output: {dim} bias_term: false }} }}"
        )
        prev_top = reduce_top
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────

def _benchmark_scenario(
    prototxt: str,
    scenario_name: str,
    batch: int = 4,
    dim_in: int = 64,
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict:
    """Run a single benchmark scenario and return timing results."""
    # --- Construction ---
    with perf_trace(f"construct[{scenario_name}]", verbose=False) as t0:
        net = make_net(prototxt)
        n_splits = count_splits(net)
        t0["splits"] = n_splits
    construct_ms = t0["elapsed_ms"]

    # --- Forward (warmup + timed) ---
    inp = np.random.randn(batch, dim_in).astype(np.float32)
    for _ in range(warmup_runs):
        net.Forward({"data": inp})

    times = []
    with perf_trace(f"forward[{scenario_name}]", verbose=False) as t1:
        for _ in range(timed_runs):
            t_start = time.perf_counter()
            out = net.Forward({"data": inp})
            t_end = time.perf_counter()
            times.append((t_end - t_start) * 1000.0)
        t1["splits"] = n_splits
        t1["runs"] = timed_runs
        # Verify outputs are finite
        for v in out.values():
            assert_finite(v, label=f"{scenario_name} output")

    times_arr = np.array(times, dtype=np.float64)
    return {
        "scenario": scenario_name,
        "n_splits": n_splits,
        "n_layers": len(list(net.layer_names())),
        "construct_ms": construct_ms,
        "forward_mean_ms": float(np.mean(times_arr)),
        "forward_std_ms": float(np.std(times_arr)),
        "forward_median_ms": float(np.median(times_arr)),
        "forward_min_ms": float(np.min(times_arr)),
        "forward_p95_ms": float(np.percentile(times_arr, 95)),
        "delta_mem": t0["delta_mem"],
        "delta_blobs": t0["delta_blobs"],
    }


def _print_results_table(results: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 100)
    print(f"{'Scenario':<30} {'Splits':>6} {'Layers':>6} "
          f"{'Construct(ms)':>13} {'Fwd mean(ms)':>12} {'Fwd p95(ms)':>11} {'Fwd std(ms)':>11}")
    print("-" * 100)
    for r in results:
        print(f"{r['scenario']:<30} {r['n_splits']:>6} {r['n_layers']:>6} "
              f"{r['construct_ms']:>13.3f} {r['forward_mean_ms']:>12.4f} "
              f"{r['forward_p95_ms']:>11.4f} {r['forward_std_ms']:>11.4f}")
    print("=" * 100)


# ──────────────────────────────────────────────────────────────────────
# Test class
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestSplitConcatBenchmark:
    """Performance benchmarks for Split/Concat nested topologies.

    These tests verify that InsertSplits and forward inference scale
    predictably and that the v1.2.0 fixes do not introduce regressions.
    Timing assertions use generous thresholds (5x median) to avoid
    flaky failures on slow/loaded CI machines.
    """

    def _run_all_benchmarks(self) -> list[dict]:
        results = []
        # Baseline: linear chain (0 splits)
        proto = _linear_chain_prototxt("linear_10", num_layers=10, dim=64)
        results.append(_benchmark_scenario(proto, "linear_10_layers", dim_in=64))

        # Inception variants
        for nb in (2, 4, 8):
            proto = _inception_prototxt(f"inception_{nb}b", num_branches=nb,
                                        dim_in=64, dim_branch=16)
            results.append(_benchmark_scenario(proto, f"inception_{nb}branches",
                                               dim_in=64))

        # Deep nested (3 levels)
        proto = _deep_nested_prototxt("deep_nested_3lv", levels=3,
                                      branches_per_level=2, dim=32, dim_branch=16)
        results.append(_benchmark_scenario(proto, "deep_nested_3levels",
                                           dim_in=32))

        # Multi fan-out (4 independent split points)
        proto = _multi_fanout_prototxt("multi_fanout_4pt", num_fanout_points=4,
                                       consumers_per_point=2, dim=32)
        results.append(_benchmark_scenario(proto, "multi_fanout_4points",
                                           dim_in=32))
        return results

    def test_construction_overhead_scales_linearly(self):
        """Net construction time (including InsertSplits) should not explode
        super-linearly with the number of split points.

        Compares 4-branch vs 8-branch Inception (2x split points = <3x time).
        """
        r4 = _benchmark_scenario(
            _inception_prototxt("bench_4b", 4, dim_in=32, dim_branch=8),
            "inc4", dim_in=32, warmup_runs=1, timed_runs=5)
        r8 = _benchmark_scenario(
            _inception_prototxt("bench_8b", 8, dim_in=32, dim_branch=8),
            "inc8", dim_in=32, warmup_runs=1, timed_runs=5)
        # Construction time ratio should be bounded (generous threshold)
        ratio = r8["construct_ms"] / max(r4["construct_ms"], 0.01)
        assert ratio < 5.0, (
            f"Construction time scaled {ratio:.1f}x from 4-branch to 8-branch "
            f"({r4['construct_ms']:.2f}ms → {r8['construct_ms']:.2f}ms); "
            f"expected <5x (linear overhead for InsertSplits)"
        )

    def test_forward_correctness_all_scenarios(self):
        """All benchmark scenarios produce valid finite outputs."""
        for name, builder, dim_in in [
            ("linear", lambda: _linear_chain_prototxt("chk_lin", 5, dim=32), 32),
            ("inc2", lambda: _inception_prototxt("chk_inc2", 2, dim_in=32, dim_branch=8), 32),
            ("inc4", lambda: _inception_prototxt("chk_inc4", 4, dim_in=32, dim_branch=8), 32),
            ("deep", lambda: _deep_nested_prototxt("chk_deep", 2, 2, 32, 8), 32),
            ("multi", lambda: _multi_fanout_prototxt("chk_multi", 2, 2, 32), 32),
        ]:
            proto = builder()
            net = make_net(proto)
            inp = np.random.randn(2, dim_in).astype(np.float32)
            out = net.Forward({"data": inp})
            for blob_name, blob_val in out.items():
                assert_finite(blob_val, label=f"{name}/{blob_name}")

    def test_split_count_matches_expected(self):
        """Verify that auto-inserted Split counts match topological analysis."""
        cases = [
            (_linear_chain_prototxt("sc_linear", 10, 32), 0, "linear=0"),
            (_inception_prototxt("sc_inc2", 2, 32, 8), 2, "data split + cat split"),
            (_inception_prototxt("sc_inc4", 4, 32, 8), 2, "data split + cat split"),
            (_deep_nested_prototxt("sc_deep", 3, 2, 32, 8), 4, "data + 3 level outputs"),
            (_multi_fanout_prototxt("sc_multi", 4, 2, 32), 4, "4 independent fan-out points"),
        ]
        for proto, expected, msg in cases:
            net = make_net(proto)
            actual = count_splits(net)
            assert actual == expected, (
                f"{msg}: expected {expected} splits, got {actual}"
            )

    def test_print_benchmark_table(self):
        """Run all benchmarks and print results table (visual inspection)."""
        results = self._run_all_benchmarks()
        _print_results_table(results)
        # Sanity: all scenarios completed successfully
        assert len(results) == 6


# ──────────────────────────────────────────────────────────────────────
# Standalone script entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running as a script: python test_split_concat_bench.py
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Run the benchmarks and print
    t = TestSplitConcatBenchmark()
    results = t._run_all_benchmarks()
    _print_results_table(results)
    print("\nCorrectness checks passed.")
