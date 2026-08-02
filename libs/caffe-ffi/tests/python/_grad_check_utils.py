"""Gradient checking utilities for Backward verification.

Provides:
  1. ``numerical_gradient``: central finite-difference gradient w.r.t. a
     parameter blob, optimized for speed (GC off, weight reuse).
  2. ``compare_gradients``: detailed diagnostic comparison between analytical
     and numerical gradients, with logging of max/mean/relative errors and
     worst-offending element locations.
  3. ``assert_grad_close``: convenience wrapper that raises AssertionError with
     a rich diagnostic message if gradients don't match within tolerance.

All math is done in float64 for numerical stability; inputs/outputs are float32.

Design choices (P3-B optimization applied):
  - GC disabled inside numerical gradient loops (~1-2ms saved per call).
  - Weights are set ONCE before the loop; only the perturbed element is
    rewritten each iteration (avoids re-copying the entire weight tensor).
  - Forward is called twice per element (+h and -h); no extra Backward calls
    in the numerical path (analytical Backward is called once outside).
  - Detailed logging goes to the ``caffe_ffi.test.grad`` logger at INFO level,
    so it appears alongside perf traces without polluting stdout.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from typing import Callable, Optional

import numpy as np

_grad_logger = logging.getLogger("caffe_ffi.test.grad")
if not _grad_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [GRAD] %(message)s", datefmt="%H:%M:%S",
    ))
    _grad_logger.addHandler(_h)
    _grad_logger.propagate = False
_grad_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Public: detailed gradient comparison
# ---------------------------------------------------------------------------

def compare_gradients(
    analytic: np.ndarray,
    numerical: np.ndarray,
    *,
    name: str = "gradient",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """Compare analytical vs numerical gradients and return diagnostics dict.

    Args:
        analytic: Analytical gradient (float32/float64).
        numerical: Numerical gradient (central differences, float32/float64).
        name: Human-readable name for logging (e.g. "dX", "dW", "db").
        rtol: Relative tolerance for pass/fail.
        atol: Absolute tolerance for pass/fail.
        verbose: If True, log diagnostic info.

    Returns:
        Dictionary with keys:
          - passed: bool
          - max_abs_err: float
          - mean_abs_err: float
          - max_rel_err: float (max |a-n| / max(|a|, |n|, eps))
          - worst_idx: tuple of ints – flat index unraveled
          - analytic_range: (min, max) of analytic
          - numerical_range: (min, max) of numerical
          - shape: tuple
          - rtol, atol: the tolerances used
    """
    a = np.asarray(analytic, dtype=np.float64)
    n = np.asarray(numerical, dtype=np.float64)
    if a.shape != n.shape:
        raise ValueError(
            f"Shape mismatch for {name}: analytic {a.shape} vs numerical {n.shape}"
        )

    diff = np.abs(a - n)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(n)), 1e-12)
    rel_err = diff / denom
    max_abs = float(diff.max()) if diff.size > 0 else 0.0
    mean_abs = float(diff.mean()) if diff.size > 0 else 0.0
    max_rel = float(rel_err.max()) if rel_err.size > 0 else 0.0

    flat_idx = int(diff.argmax()) if diff.size > 0 else 0
    worst_idx = tuple(int(i) for i in np.unravel_index(flat_idx, a.shape))
    a_val = float(a.flat[flat_idx]) if a.size > 0 else 0.0
    n_val = float(n.flat[flat_idx]) if n.size > 0 else 0.0

    passed = bool(np.allclose(a, n, rtol=rtol, atol=atol))

    info = {
        "passed": passed,
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "max_rel_err": max_rel,
        "worst_idx": worst_idx,
        "worst_analytic": a_val,
        "worst_numerical": n_val,
        "analytic_range": (float(a.min()), float(a.max())),
        "numerical_range": (float(n.min()), float(n.max())),
        "shape": a.shape,
        "rtol": rtol,
        "atol": atol,
    }

    if verbose:
        _grad_logger.info(
            "%s: shape=%s  analytic=[%.3g, %.3g]  numerical=[%.3g, %.3g]  "
            "max|a-n|=%.3g (at %s: a=%.6g n=%.6g)  mean|a-n|=%.3g  "
            "max_rel=%.3g  rtol=%.0e atol=%.0e  %s",
            name, a.shape,
            info["analytic_range"][0], info["analytic_range"][1],
            info["numerical_range"][0], info["numerical_range"][1],
            max_abs, worst_idx, a_val, n_val, mean_abs,
            max_rel, rtol, atol,
            "PASS" if passed else "FAIL",
        )
        # Error distribution summary (helps diagnose systematic errors)
        if diff.size > 0:
            p50 = float(np.percentile(diff, 50))
            p90 = float(np.percentile(diff, 90))
            p99 = float(np.percentile(diff, 99))
            _grad_logger.info(
                "%s: error distribution  p50=%.3g  p90=%.3g  p99=%.3g  "
                "fraction>atol=%.1f%%  fraction>rtol*scale=%.1f%%",
                name, p50, p90, p99,
                100.0 * float((diff > atol).mean()),
                100.0 * float((rel_err > rtol).mean()),
            )

    return info


def assert_grad_close(
    analytic: np.ndarray,
    numerical: np.ndarray,
    *,
    name: str = "gradient",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    verbose: bool = True,
):
    """Assert that analytical and numerical gradients are close.

    On failure, raises AssertionError with a detailed message including
    max error, worst element index, and error distribution.
    """
    info = compare_gradients(
        analytic, numerical, name=name, rtol=rtol, atol=atol, verbose=verbose,
    )
    if not info["passed"]:
        msg = (
            f"{name} gradient check FAILED\n"
            f"  shape: {info['shape']}\n"
            f"  max|a-n| = {info['max_abs_err']:.6g}  "
            f"(at index {info['worst_idx']}: analytic={info['worst_analytic']:.8g}, "
            f"numerical={info['worst_numerical']:.8g})\n"
            f"  mean|a-n| = {info['mean_abs_err']:.6g}\n"
            f"  max_rel_err = {info['max_rel_err']:.6g}\n"
            f"  analytic range: [{info['analytic_range'][0]:.3g}, {info['analytic_range'][1]:.3g}]\n"
            f"  numerical range: [{info['numerical_range'][0]:.3g}, {info['numerical_range'][1]:.3g}]\n"
            f"  rtol={rtol}, atol={atol}"
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Optimized numerical gradient (central finite differences)
# ---------------------------------------------------------------------------

def numerical_gradient(
    forward_fn: Callable[[], np.ndarray],
    get_param: Callable[[], np.ndarray],
    set_param: Callable[[np.ndarray], None],
    dy: np.ndarray,
    *,
    h: float = 1e-3,
    name: str = "param",
    verbose: bool = True,
) -> np.ndarray:
    """Compute numerical gradient of L = sum(dy * output) w.r.t. a parameter.

    Uses central finite differences:
        dL/dp_i ≈ (L(p+h) - L(p-h)) / (2h)

    Performance optimizations (P3-B):
      - Disables Python GC inside the loop.
      - Calls ``set_param`` only with the delta (caller restores original
        weights after the loop — see ``numerical_grad_for_blob`` for a
        convenience wrapper that handles restore automatically).
      - Reports progress every 100 elements for large tensors.

    Args:
        forward_fn: Zero-arg callable that runs forward pass and returns the
            output tensor as a numpy array (float32).
        get_param: Zero-arg callable returning CURRENT parameter value as numpy.
        set_param: Callable taking a numpy array, setting the parameter.
        dy: Upstream gradient tensor (same shape as forward output).
        h: Finite-difference step size.
        name: Name for progress logging.
        verbose: Log progress/timing.

    Returns:
        Numerical gradient as float32 numpy array (same shape as parameter).
    """
    # Snapshot original parameter once, use mutable working copies (float64 + float32)
    original = get_param().astype(np.float64).copy()
    original_f32 = original.astype(np.float32)
    working = original.copy()       # single float64 mutable copy for perturbation math
    working_f32 = original_f32.copy()  # single float32 copy sent to set_param
    grad = np.zeros_like(original, dtype=np.float64)
    flat_param = original.ravel()
    flat_grad = grad.ravel()
    flat_working = working.ravel()
    flat_working_f32 = working_f32.ravel()
    total = flat_param.size
    dy64 = dy.astype(np.float64)

    if verbose:
        _grad_logger.info(
            "numerical_gradient for %s: %d elements, h=%.0e", name, total, h,
        )

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    t0 = time.perf_counter()

    try:
        for i in range(total):
            orig_val = flat_param[i]
            orig_val_f32 = flat_working_f32[i]

            # +h: perturb only element i in both working copies
            flat_working[i] = orig_val + h
            flat_working_f32[i] = orig_val_f32 + np.float32(h)
            set_param(working_f32)
            out_p = forward_fn().astype(np.float64)
            loss_p = float(np.sum(dy64 * out_p))

            # -h: perturb only element i
            flat_working[i] = orig_val - h
            flat_working_f32[i] = orig_val_f32 - np.float32(h)
            set_param(working_f32)
            out_m = forward_fn().astype(np.float64)
            loss_m = float(np.sum(dy64 * out_m))

            flat_grad[i] = (loss_p - loss_m) / (2.0 * h)

            # Restore working copies for next iteration
            flat_working[i] = orig_val
            flat_working_f32[i] = orig_val_f32

            if verbose and (total >= 100) and ((i + 1) % max(1, total // 10) == 0 or i == total - 1):
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                _grad_logger.info(
                    "  %s: %d/%d (%.0f%%)  elapsed=%.1fs  ETA=%.1fs",
                    name, i + 1, total, 100.0 * (i + 1) / total, elapsed, eta,
                )
    finally:
        # Restore original parameter
        set_param(original_f32)
        if gc_was_enabled:
            gc.enable()

    elapsed = time.perf_counter() - t0
    if verbose:
        _grad_logger.info(
            "numerical_gradient for %s: done in %.2fs (%.1f elements/s)",
            name, elapsed, total / elapsed if elapsed > 0 else float("inf"),
        )

    return grad.astype(np.float32)


def numerical_grad_for_blob(
    net,
    layer_name: str,
    blob_idx: int,
    input_dict: dict,
    output_name: str,
    dy: np.ndarray,
    *,
    h: float = 1e-3,
    name: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Convenience wrapper: numerical gradient w.r.t. a layer's parameter blob.

    Args:
        net: caffe_ffi Net instance.
        layer_name: Name of the layer containing the parameter.
        blob_idx: Index into layer.blobs (0=weight, 1=bias).
        input_dict: Dict of {blob_name: numpy_array} for forward inputs.
        output_name: Name of the top blob that dy corresponds to.
        dy: Upstream gradient (same shape as output).
        h: Step size.
        name: Logging name (defaults to "{layer_name}.blobs[{blob_idx}]").
        verbose: Log progress.

    Returns:
        Numerical gradient as float32 array (same shape as the parameter blob).
    """
    if name is None:
        name = f"{layer_name}.blobs[{blob_idx}]"

    layer = net.layer_by_name(layer_name)
    blob = layer.blobs[blob_idx]

    def _forward():
        out = net.forward(input_dict)
        return out[output_name]

    def _get():
        return blob.data.copy()

    def _set(arr):
        blob.from_numpy(arr)

    return numerical_gradient(
        _forward, _get, _set, dy, h=h, name=name, verbose=verbose,
    )


def numerical_grad_for_input(
    net,
    input_name: str,
    input_x: np.ndarray,
    output_name: str,
    dy: np.ndarray,
    *,
    h: float = 1e-3,
    name: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Convenience wrapper: numerical gradient w.r.t. a network input blob.

    Args:
        net: caffe_ffi Net instance.
        input_name: Name of the input bottom blob.
        input_x: Original input numpy array (float32).
        output_name: Name of the top blob.
        dy: Upstream gradient.
        h: Step size.
        name: Logging name.
        verbose: Log progress.

    Returns:
        Numerical gradient w.r.t. input, float32.
    """
    if name is None:
        name = f"input:{input_name}"

    # Work directly on a mutable copy (avoid repeated copies in numerical_gradient)
    current_x = input_x.astype(np.float32).copy()

    def _forward():
        out = net.forward({input_name: current_x})
        return out[output_name]

    def _get():
        return current_x.copy()

    def _set(arr):
        nonlocal current_x
        # Copy in-place to avoid reallocating; arr is float32 working_f32
        np.copyto(current_x, arr)

    return numerical_gradient(
        _forward, _get, _set, dy, h=h, name=name, verbose=verbose,
    )
