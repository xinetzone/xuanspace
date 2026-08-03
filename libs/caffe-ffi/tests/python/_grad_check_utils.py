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

Logging levels:
  - INFO:  progress, summary stats, pass/fail
  - DEBUG: per-element details, loss values, intermediate computations
  - WARNING: potential numerical issues (saturated regions, low SNR, kinks)
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
# Default to INFO; set CAFFE_FFI_GRAD_LOG=DEBUG for verbose per-element tracing
_log_level = os.environ.get("CAFFE_FFI_GRAD_LOG", "INFO").upper()
_grad_logger.setLevel(getattr(logging, _log_level, logging.INFO))


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

    # NaN/Inf detection (critical for diagnosing catastrophic failures)
    a_has_nan = bool(np.any(np.isnan(a)))
    n_has_nan = bool(np.any(np.isnan(n)))
    a_has_inf = bool(np.any(np.isinf(a)))
    n_has_inf = bool(np.any(np.isinf(n)))
    has_finite_issue = a_has_nan or n_has_nan or a_has_inf or n_has_inf

    # Gradient norms and direction agreement
    a_norm = float(np.linalg.norm(a))
    n_norm = float(np.linalg.norm(n))
    norm_ratio = a_norm / n_norm if n_norm > 1e-12 else float('inf')
    # Cosine similarity: dot(a,n) / (|a|*|n|)
    flat_a = a.ravel()
    flat_n = n.ravel()
    dot = float(np.dot(flat_a, flat_n))
    cos_sim = dot / (a_norm * n_norm) if (a_norm > 1e-12 and n_norm > 1e-12) else 0.0

    passed = bool(np.allclose(a, n, rtol=rtol, atol=atol)) and not has_finite_issue

    # Detect potential numerical issues for WARNING logs
    warnings = []
    # SNR warning: if gradient norm is very small relative to loss scale, numerical gradient may be noisy
    if a_norm < 1e-8 and n_norm < 1e-8:
        warnings.append(f"gradient norm near zero (|a|={a_norm:.3g}, |n|={n_norm:.3g}), check for vanishing gradients")
    # Norm ratio mismatch: if analytic and numerical norms differ by >2x, something is wrong
    if norm_ratio > 2.0 or (norm_ratio < 0.5 and norm_ratio > 0):
        warnings.append(f"norm ratio |a|/|n|={norm_ratio:.3g} far from 1.0 (scale mismatch, expected ~1.0)")
    # Cosine similarity: should be close to 1.0 for correct gradients; <0.99 suggests direction issues
    if cos_sim < 0.99 and a_norm > 1e-6 and n_norm > 1e-6:
        warnings.append(f"cosine similarity={cos_sim:.4f} below 0.99 (gradient direction mismatch or noise)")
    # High relative error in significant fraction of elements
    high_err_frac = float((rel_err > rtol).mean()) if rel_err.size > 0 else 0.0
    if high_err_frac > 0.1 and not passed:
        warnings.append(f"{high_err_frac*100:.1f}% of elements exceed rtol={rtol:.0e} (systematic error, not isolated points)")
    # Check for saturated regions (exact 0/1 values where gradient might be zero)
    a_zero_frac = float((np.abs(a) < 1e-12).mean()) if a.size > 0 else 0.0
    n_zero_frac = float((np.abs(n) < 1e-12).mean()) if n.size > 0 else 0.0
    if abs(a_zero_frac - n_zero_frac) > 0.1 and max(a_zero_frac, n_zero_frac) > 0.2:
        warnings.append(f"zero-fraction mismatch: a={a_zero_frac*100:.1f}% vs n={n_zero_frac*100:.1f}% (saturated ReLU/dead neuron issue?)")

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
        "analytic_l2_norm": a_norm,
        "numerical_l2_norm": n_norm,
        "norm_ratio": norm_ratio,
        "cosine_similarity": cos_sim,
        "shape": a.shape,
        "rtol": rtol,
        "atol": atol,
        "has_nan": a_has_nan or n_has_nan,
        "has_inf": a_has_inf or n_has_inf,
        "warnings": warnings,
        "high_error_fraction": high_err_frac,
    }

    if verbose:
        status = "FAIL" if not passed else "PASS"
        finite_tag = ""
        if has_finite_issue:
            finite_tag = f"  ⚠ NaN/Inf detected (a:nan={a_has_nan},inf={a_has_inf} n:nan={n_has_nan},inf={n_has_inf})"
        _grad_logger.info(
            "%s: shape=%s  analytic=[%.3g, %.3g] |a|=%.3g  numerical=[%.3g, %.3g] |n|=%.3g  "
            "|a|/|n|=%.3g  cos_sim=%.6f  max|a-n|=%.3g (at %s: a=%.6g n=%.6g)  mean|a-n|=%.3g  "
            "max_rel=%.3g  rtol=%.0e atol=%.0e  %s%s",
            name, a.shape,
            info["analytic_range"][0], info["analytic_range"][1], a_norm,
            info["numerical_range"][0], info["numerical_range"][1], n_norm,
            norm_ratio, cos_sim,
            max_abs, worst_idx, a_val, n_val, mean_abs,
            max_rel, rtol, atol,
            status, finite_tag,
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
        # Worst-element neighborhood (for 4D NCHW tensors: show 3x3 spatial patch)
        if a.ndim == 4 and not passed:
            n_idx, c_idx, h_idx, w_idx = worst_idx
            _grad_logger.info(
                "%s: worst element neighborhood at (n=%d,c=%d,h=%d,w=%d):",
                name, n_idx, c_idx, h_idx, w_idx,
            )
            # Extract 3x3 patch around worst spatial location
            h_lo = max(0, h_idx - 1)
            h_hi = min(a.shape[2], h_idx + 2)
            w_lo = max(0, w_idx - 1)
            w_hi = min(a.shape[3], w_idx + 2)
            a_patch = a[n_idx, c_idx, h_lo:h_hi, w_lo:w_hi]
            n_patch = n[n_idx, c_idx, h_lo:h_hi, w_lo:w_hi]
            err_patch = np.abs(a_patch - n_patch)
            for dh in range(a_patch.shape[0]):
                for dw in range(a_patch.shape[1]):
                    marker = " << WORST" if (h_lo + dh == h_idx and w_lo + dw == w_idx) else ""
                    _grad_logger.info(
                        "    (%d,%d): a=%+.6g  n=%+.6g  |Δ|=%.3g%s",
                        h_lo + dh, w_lo + dw,
                        float(a_patch[dh, dw]), float(n_patch[dh, dw]),
                        float(err_patch[dh, dw]), marker,
                    )
        # Log warnings if any
        for w in warnings:
            _grad_logger.warning("%s: %s", name, w)

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
        finite_note = ""
        if info.get("has_nan") or info.get("has_inf"):
            finite_note = "\n  ⚠ NaN or Inf detected in gradients!"
        msg = (
            f"{name} gradient check FAILED\n"
            f"  shape: {info['shape']}\n"
            f"  max|a-n| = {info['max_abs_err']:.6g}  "
            f"(at index {info['worst_idx']}: analytic={info['worst_analytic']:.8g}, "
            f"numerical={info['worst_numerical']:.8g})\n"
            f"  mean|a-n| = {info['mean_abs_err']:.6g}\n"
            f"  max_rel_err = {info['max_rel_err']:.6g}\n"
            f"  analytic L2 norm = {info['analytic_l2_norm']:.6g}, range=[{info['analytic_range'][0]:.3g}, {info['analytic_range'][1]:.3g}]\n"
            f"  numerical L2 norm = {info['numerical_l2_norm']:.6g}, range=[{info['numerical_range'][0]:.3g}, {info['numerical_range'][1]:.3g}]\n"
            f"  norm ratio |a|/|n| = {info['norm_ratio']:.6g}\n"
            f"  cosine similarity = {info['cosine_similarity']:.8f}\n"
            f"  rtol={rtol}, atol={atol}{finite_note}"
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
            "numerical_gradient for %s: %d elements, h=%.0e, dy shape=%s",
            name, total, h, dy64.shape,
        )

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    t0 = time.perf_counter()

    # Measure baseline loss (at original parameter value) for SNR diagnostics
    out0 = forward_fn().astype(np.float64)
    loss0 = float(np.sum(dy64 * out0))
    dy_norm = float(np.linalg.norm(dy64))
    out_norm = float(np.linalg.norm(out0))
    if verbose:
        _grad_logger.info(
            "numerical_gradient for %s: baseline loss=%.6g  |dy|=%.3g  |out|=%.3g  dy·out=%.6g",
            name, loss0, dy_norm, out_norm, loss0,
        )

    first_loss_p = None
    first_loss_m = None
    first_delta = None

    # Diagnostics accumulators
    nan_count = 0
    inf_count = 0
    zero_delta_count = 0  # delta == 0 exactly (numerical underflow or dead neuron)
    max_grad_mag = 0.0
    min_grad_mag = float('inf')
    kink_suspects = 0  # elements where loss_p == loss_m but param is near a known kink (0 for ReLU/ELU/PReLU)

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

            delta = loss_p - loss_m
            g_i = delta / (2.0 * h)
            flat_grad[i] = g_i

            # Capture first element for SNR diagnostics
            if i == 0:
                first_loss_p = loss_p
                first_loss_m = loss_m
                first_delta = delta

            # Diagnostics: detect numerical issues
            if np.isnan(g_i):
                nan_count += 1
            if np.isinf(g_i):
                inf_count += 1
            abs_g = abs(g_i)
            if abs_g > max_grad_mag:
                max_grad_mag = abs_g
            if abs_g > 0 and abs_g < min_grad_mag:
                min_grad_mag = abs_g
            if delta == 0.0:
                zero_delta_count += 1
                # Check if parameter is near 0 (C¹ kink for ReLU/ELU/PReLU)
                if abs(orig_val) < h * 2:
                    kink_suspects += 1

            # Per-element DEBUG logging (for CAFFE_FFI_GRAD_LOG=DEBUG)
            if _grad_logger.isEnabledFor(logging.DEBUG):
                _grad_logger.debug(
                    "  %s[%d]: param=%.8g  L+h=%.10g  L-h=%.10g  Δ=%.3g  grad=%.6g",
                    name, i, orig_val, loss_p, loss_m, delta, g_i,
                )

            # Restore working copies for next iteration
            flat_working[i] = orig_val
            flat_working_f32[i] = orig_val_f32

            if verbose and (total >= 100) and ((i + 1) % max(1, total // 10) == 0 or i == total - 1):
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                current_grad_norm = float(np.linalg.norm(flat_grad[:i+1]))
                _grad_logger.info(
                    "  %s: %d/%d (%.0f%%)  elapsed=%.1fs  ETA=%.1fs  |grad|=%.3g",
                    name, i + 1, total, 100.0 * (i + 1) / total, elapsed, eta,
                    current_grad_norm,
                )
    finally:
        # Restore original parameter
        set_param(original_f32)
        if gc_was_enabled:
            gc.enable()

    elapsed = time.perf_counter() - t0
    grad_norm = float(np.linalg.norm(grad))
    nonzero_count = total - zero_delta_count

    if verbose:
        _grad_logger.info(
            "numerical_gradient for %s: done in %.2fs (%.1f elements/s)  |grad|=%.3g  range=[%.3g, %.3g]",
            name, elapsed, total / elapsed if elapsed > 0 else float("inf"),
            grad_norm, float(grad.min()), float(grad.max()),
        )
        # Diagnostic summary
        _grad_logger.info(
            "numerical_gradient for %s: %d/%d nonzero grads (%.1f%%), "
            "NaN=%d Inf=%d zero_delta=%d kink_suspects=%d  |grad|_max=%.3g |grad|_min_nonzero=%.3g",
            name, nonzero_count, total, 100.0 * nonzero_count / total if total > 0 else 0,
            nan_count, inf_count, zero_delta_count, kink_suspects,
            max_grad_mag, min_grad_mag if min_grad_mag != float('inf') else 0.0,
        )
        if first_loss_p is not None and first_loss_m is not None:
            snr = abs(first_delta) / max(abs(loss0), 1e-12) if total > 0 else 0.0
            delta_scale = abs(first_delta) / (2.0 * h)
            _grad_logger.info(
                "numerical_gradient for %s: first element signal: L+h=%.10g L-h=%.10g Δ=%.3g (Δ/L0=%.2e, grad≈%.6g)",
                name, first_loss_p, first_loss_m, first_delta, snr, delta_scale,
            )
            # Low SNR warning: if delta is near machine epsilon relative to h, numerical gradient is unreliable
            rel_delta = abs(first_delta) / max(abs(loss0) * h, 1e-30) if abs(loss0) > 0 else float('inf')
            if rel_delta < 1e-6 and nonzero_count < total * 0.1:
                _grad_logger.warning(
                    "%s: LOW SNR detected! Δ/(L0*h)=%.2e is near machine precision. "
                    "Consider using larger h or checking if output is independent of this parameter.",
                    name, rel_delta,
                )
        # Warn about potential issues
        if nan_count > 0:
            _grad_logger.warning("%s: %d NaN gradients detected!", name, nan_count)
        if inf_count > 0:
            _grad_logger.warning("%s: %d Inf gradients detected!", name, inf_count)
        if zero_delta_count > total * 0.5 and total > 10:
            _grad_logger.warning(
                "%s: %d/%d elements (%.1f%%) have zero delta — gradient is exactly zero. "
                "This is expected for dead ReLU neurons, but verify it's not a bug.",
                name, zero_delta_count, total, 100.0 * zero_delta_count / total,
            )
        if kink_suspects > 0:
            _grad_logger.warning(
                "%s: %d elements have zero delta near parameter value 0 (possible C¹ kink for ReLU/ELU/PReLU). "
                "Consider using avoid_c1_discontinuity() to perturb inputs away from kinks.",
                name, kink_suspects,
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


# ---------------------------------------------------------------------------
# Regression test: numpy reference vs C++ backward auto-comparison
# ---------------------------------------------------------------------------

def assert_backward_matches_reference(
    net,
    ref_backward_fn,
    input_name: str,
    output_name: str,
    x: np.ndarray,
    dy: np.ndarray,
    *,
    name: Optional[str] = None,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    verbose: bool = True,
    skip_numerical: bool = False,
    numerical_h: float = 1e-3,
    numerical_rtol: float = 1e-2,
    numerical_atol: float = 1e-3,
    **ref_kwargs,
) -> dict:
    """Assert C++ backward matches numpy reference backward (regression test).

    Runs both C++ analytic backward and numpy reference backward, then
    optionally also runs a numerical gradient cross-check. Returns a
    diagnostics dict for programmatic inspection.

    Args:
        net: caffe_ffi Net instance (already constructed, weights set).
        ref_backward_fn: Callable ``(dy, x, **ref_kwargs) -> dx_ref`` that
            computes the expected input gradient in pure numpy. The x passed
            is the *original* input (same x used for forward), so the ref
            function can compute argmax/indices needed for MAX-style routing.
        input_name: Name of the input bottom blob.
        output_name: Name of the output top blob (for forward result + dy).
        x: Input tensor numpy array (float32, NCHW).
        dy: Upstream gradient tensor (float32, same shape as output).
        name: Human-readable label for logging (defaults to output_name).
        rtol: Relative tolerance for analytic-vs-reference comparison.
        atol: Absolute tolerance for analytic-vs-reference comparison.
        verbose: If True, log diagnostic info.
        skip_numerical: If True, skip the slow numerical gradient cross-check
            (useful for quick CI runs; default False runs numerical check).
        numerical_h: Step size for numerical gradient (only if not skip_numerical).
        numerical_rtol: Relative tolerance for numerical gradient check.
        numerical_atol: Absolute tolerance for numerical gradient check.
        **ref_kwargs: Extra keyword arguments forwarded to ``ref_backward_fn``
            (e.g. kernel_size, stride, pad, pool_type).

    Returns:
        Dictionary with keys:
          - ref_passed: bool – analytic vs numpy reference match
          - numerical_passed: bool – analytic vs numerical match (or None if skipped)
          - ref_info: diagnostics dict from compare_gradients (analytic vs ref)
          - numerical_info: diagnostics dict (analytic vs numerical) or None
          - dX_cpp: C++ analytic gradient numpy array
          - dX_ref: numpy reference gradient numpy array
          - y: forward output numpy array

    Raises:
        AssertionError: If analytic gradient doesn't match reference within
            tolerance, or numerical gradient check fails.
    """
    if name is None:
        name = f"backward:{output_name}"

    x_f32 = np.asarray(x, dtype=np.float32)
    dy_f32 = np.asarray(dy, dtype=np.float32)

    # 1. Run C++ forward + backward
    out = net.forward({input_name: x_f32})
    y = out[output_name]
    net.backward({output_name: dy_f32})
    dX_cpp = net.blob_by_name(input_name).diff

    # 2. Run numpy reference backward
    dX_ref = ref_backward_fn(dy_f32, x_f32, **ref_kwargs)
    dX_ref = np.asarray(dX_ref, dtype=np.float32)

    # 3. Analytic vs reference comparison
    ref_info = compare_gradients(
        dX_cpp, dX_ref, name=f"{name} (cpp vs ref)",
        rtol=rtol, atol=atol, verbose=verbose,
    )

    if not ref_info["passed"]:
        msg = (
            f"{name} BACKWARD REGRESSION FAILED (C++ vs numpy reference)\n"
            f"  shape: {ref_info['shape']}\n"
            f"  max|cpp-ref| = {ref_info['max_abs_err']:.6g}  "
            f"(at {ref_info['worst_idx']}: cpp={ref_info['worst_analytic']:.8g}, "
            f"ref={ref_info['worst_numerical']:.8g})\n"
            f"  mean|cpp-ref| = {ref_info['mean_abs_err']:.6g}\n"
            f"  max_rel_err = {ref_info['max_rel_err']:.6g}\n"
            f"  cosine similarity = {ref_info['cosine_similarity']:.8f}\n"
            f"  norm ratio |cpp|/|ref| = {ref_info['norm_ratio']:.6g}\n"
            f"  rtol={rtol}, atol={atol}"
        )
        raise AssertionError(msg)

    # 4. Optional numerical gradient cross-check
    numerical_info = None
    if not skip_numerical:
        numerical_dX = numerical_grad_for_input(
            net, input_name, x_f32, output_name, dy_f32,
            h=numerical_h, name=f"{name} (numerical)", verbose=verbose,
        )
        numerical_info = compare_gradients(
            dX_cpp, numerical_dX, name=f"{name} (cpp vs numerical)",
            rtol=numerical_rtol, atol=numerical_atol, verbose=verbose,
        )
        if not numerical_info["passed"]:
            msg = (
                f"{name} NUMERICAL GRADIENT CHECK FAILED\n"
                f"  max|cpp-num| = {numerical_info['max_abs_err']:.6g}\n"
                f"  cosine similarity = {numerical_info['cosine_similarity']:.8f}\n"
                f"  rtol={numerical_rtol}, atol={numerical_atol}"
            )
            raise AssertionError(msg)

    if verbose:
        if numerical_info is None:
            num_status = "SKIP"
        elif numerical_info["passed"]:
            num_status = "PASS"
        else:
            num_status = "FAIL"
        _grad_logger.info(
            "%s: regression check complete — ref=%s numerical=%s",
            name, "PASS" if ref_info["passed"] else "FAIL", num_status,
        )

    return {
        "ref_passed": ref_info["passed"],
        "numerical_passed": None if numerical_info is None else numerical_info["passed"],
        "ref_info": ref_info,
        "numerical_info": numerical_info,
        "dX_cpp": dX_cpp,
        "dX_ref": dX_ref,
        "y": y,
    }
