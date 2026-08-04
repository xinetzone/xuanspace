"""Caffe-FFI test helper library — reusable assertion and construction utilities.

Extracted from test_insert_splits.py v1.2.0 boundary case testing experience.
Provides generic helpers for:
  - Net construction from prototxt strings
  - Split layer counting and naming assertions
  - Split position/ordering assertions
  - Forward pass output shape verification
  - Saturation-safe float assertions (NaN/Inf guards, threshold-based)

Usage in test files:
    from .caffe_test_helpers import (
        make_net, count_splits,
        assert_split_exists, assert_split_after_producer,
        assert_split_at_position, assert_split_order,
        assert_no_split, assert_exact_split_name,
        assert_forward_shapes, assert_finite,
        assert_sigmoid_negative_saturated,
        assert_sigmoid_positive_saturated,
        assert_sigmoid_transition,
    )

All assertion helpers raise AssertionError with descriptive messages on failure.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from caffe_ffi import net_param_from_string, net_from_param


# ──────────────────────────────────────────────────────────────────────
# Net construction
# ──────────────────────────────────────────────────────────────────────

def make_net(prototxt: str):
    """Construct a Net from a prototxt string (convenience wrapper)."""
    return net_from_param(net_param_from_string(prototxt))


# ──────────────────────────────────────────────────────────────────────
# Split counting
# ──────────────────────────────────────────────────────────────────────

def count_splits(net) -> int:
    """Count layers whose name ends with ``_split`` (auto-inserted Split layers)."""
    return sum(1 for n in net.layer_names() if n.endswith("_split"))


# ──────────────────────────────────────────────────────────────────────
# Split structural assertions
# ──────────────────────────────────────────────────────────────────────

def _find_split_idx(names: Sequence[str], pattern: str) -> int:
    """Return the index of the first layer name containing *pattern*.

    Raises AssertionError if not found.
    """
    for i, n in enumerate(names):
        if pattern in n:
            return i
    raise AssertionError(
        f"Split layer matching pattern {pattern!r} not found in layer list: {list(names)}"
    )


def assert_split_exists(names: Sequence[str], pattern: str) -> int:
    """Assert that a split layer matching *pattern* exists; returns its index.

    Uses substring match (``pattern in name``) for flexibility with naming
    conventions like ``data_input_0_split`` vs exact-match requirements.
    """
    return _find_split_idx(names, pattern)


def assert_split_after_producer(
    names: Sequence[str],
    producer_name: str,
    split_pattern: str,
) -> int:
    """Assert that the split layer appears **immediately after** its producer.

    Returns the split index on success.

    Typical use::

        names = list(net.layer_names())
        assert_split_after_producer(names, "cat", "cat_out_cat_0_split")
    """
    producer_idx = names.index(producer_name)
    split_idx = _find_split_idx(names, split_pattern)
    assert split_idx == producer_idx + 1, (
        f"Split '{split_pattern}' (idx {split_idx}) should be immediately "
        f"after producer '{producer_name}' (idx {producer_idx}), "
        f"expected idx {producer_idx + 1}"
    )
    return split_idx


def assert_split_at_position(
    names: Sequence[str],
    split_pattern: str,
    expected_idx: int,
) -> int:
    """Assert that a split layer is at a specific position in the layer list.

    Returns the split index on success.

    Typical use::

        # External input split must be at position 0
        assert_split_at_position(names, "data_input_0_split", 0)
    """
    split_idx = _find_split_idx(names, split_pattern)
    assert split_idx == expected_idx, (
        f"Split '{split_pattern}' should be at position {expected_idx}, "
        f"got idx {split_idx}"
    )
    return split_idx


def assert_split_order(
    names: Sequence[str],
    pattern_a: str,
    pattern_b: str,
    msg: str = "",
) -> tuple[int, int]:
    """Assert that split matching *pattern_a* appears **before** split matching *pattern_b*.

    Returns ``(idx_a, idx_b)`` on success.

    Typical use::

        # Multiple external input splits must appear in declaration order
        assert_split_order(names, "data_input_", "weight_input_",
                           msg="data split should precede weight split")
    """
    idx_a = _find_split_idx(names, pattern_a)
    idx_b = _find_split_idx(names, pattern_b)
    assert idx_a < idx_b, (
        f"{msg + ': ' if msg else ''}"
        f"Split '{pattern_a}' (idx {idx_a}) should appear before "
        f"split '{pattern_b}' (idx {idx_b})"
    )
    return idx_a, idx_b


def assert_no_split(names: Sequence[str], pattern: str) -> None:
    """Assert that **no** split layer matching *pattern* exists.

    Typical use::

        # A single-consumer blob should NOT have a split
        assert_no_split(names, "out1_fc1_0_split")
    """
    matches = [n for n in names if pattern in n]
    assert not matches, (
        f"Expected no split matching '{pattern}', but found: {matches}"
    )


def assert_exact_split_name(names: Sequence[str], exact_name: str) -> int:
    """Assert an exact split layer name exists; returns its index.

    Use this when the full split name is known (e.g. ``"data_data_0_split"``
    for an explicit Input-layer split).
    """
    assert exact_name in names, (
        f"Expected exact split name '{exact_name}' in layer list: {list(names)}"
    )
    return names.index(exact_name)


# ──────────────────────────────────────────────────────────────────────
# Forward pass verification
# ──────────────────────────────────────────────────────────────────────

def assert_forward_shapes(
    outputs: dict,
    expected: dict[str, tuple[int, ...]],
) -> None:
    """Assert that forward outputs have the expected shapes.

    Parameters
    ----------
    outputs : dict
        The dict returned by ``net.Forward(...)``.
    expected : dict[str, tuple]
        Mapping of blob name → expected shape tuple.

    Raises AssertionError if any blob is missing or has a wrong shape.

    Typical use::

        outputs = net.Forward({"data": inp})
        assert_forward_shapes(outputs, {
            "fc_c_out": (2, 3),
            "fc_d_out": (2, 3),
        })
    """
    for blob_name, expected_shape in expected.items():
        assert blob_name in outputs, (
            f"Forward output missing blob '{blob_name}'. "
            f"Available blobs: {list(outputs.keys())}"
        )
        actual_shape = outputs[blob_name].shape
        assert actual_shape == expected_shape, (
            f"Blob '{blob_name}' shape mismatch: "
            f"expected {expected_shape}, got {actual_shape}"
        )


def assert_forward_dtype(outputs: dict, blob_names: list[str], dtype=np.float32) -> None:
    """Assert that output blobs have the expected dtype (default float32)."""
    for name in blob_names:
        assert outputs[name].dtype == dtype, (
            f"Blob '{name}' dtype is {outputs[name].dtype}, expected {dtype}"
        )


# ──────────────────────────────────────────────────────────────────────
# Float / saturation-safe assertions
# ──────────────────────────────────────────────────────────────────────

def assert_finite(arr, label: str = "array") -> None:
    """Assert that *arr* contains no NaN or Inf values.

    Coerces the input to a numpy array first, so it also accepts Blob objects
    returned by ``net.Forward()`` via their ``to_numpy()`` method.

    Typical use after a forward pass::

        out = net.Forward({"data": inp})
        for k, v in out.items():
            assert_finite(v, label=k)
    """
    if hasattr(arr, "to_numpy"):
        arr = arr.to_numpy()
    else:
        arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


def assert_all_between(
    arr: np.ndarray,
    lo: float,
    hi: float,
    label: str = "array",
) -> None:
    """Assert all values in *arr* are in ``[lo, hi]`` (inclusive)."""
    assert_finite(arr, label)
    assert np.all(arr >= lo), f"{label} has values < {lo}: min={arr.min()}"
    assert np.all(arr <= hi), f"{label} has values > {hi}: max={arr.max()}"


def assert_sigmoid_negative_saturated(arr: np.ndarray, label: str = "sigmoid") -> None:
    """Assert sigmoid output is effectively zero for large negative inputs (< 1e-37).

    Uses threshold comparison (``< 1e-37``) instead of ``== 0.0`` to avoid
    subnormal-number failures (see TESTING_GUIDELINES.md §3).
    """
    assert_finite(arr, label)
    assert np.all(arr < 1e-37), (
        f"{label} should be < 1e-37 (negative saturation), got max={arr.max()}"
    )


def assert_sigmoid_positive_saturated(arr: np.ndarray, label: str = "sigmoid") -> None:
    """Assert sigmoid output is exactly 1.0 for large positive inputs (x >= 17).

    In float32, sigmoid(17.0) rounds to exactly 1.0 due to ULP limits.
    """
    assert_finite(arr, label)
    assert np.all(arr == 1.0), (
        f"{label} should be exactly 1.0 (positive saturation), got min={arr.min()}"
    )


def assert_sigmoid_transition(arr: np.ndarray, label: str = "sigmoid") -> None:
    """Assert sigmoid output is strictly in (0, 1) for transition-zone inputs (|x| <= 14)."""
    assert_finite(arr, label)
    assert np.all(arr > 0.0), f"{label} has values <= 0 in transition zone"
    assert np.all(arr < 1.0), f"{label} has values >= 1 in transition zone"


# ──────────────────────────────────────────────────────────────────────
# Numerical gradient stability helpers
# ──────────────────────────────────────────────────────────────────────

def avoid_c1_discontinuity(
    x: np.ndarray,
    h: float = 1e-3,
    kink_points: float | tuple[float, ...] = 0.0,
    margin: float = 2.0,
) -> np.ndarray:
    """Push values near C¹-discontinuity kinks away to prevent finite-difference straddling.

    For piecewise activation functions with C¹-discontinuous kinks (e.g., ReLU, LeakyReLU,
    PReLU at x=0), central differences with step *h* that straddle the kink produce O(h)
    truncation error instead of O(h²), causing numerical gradient checks to fail.

    This helper clamps all points within ``margin * h`` of any kink point to exactly
    ``margin * h`` away (on the same side as the original value), ensuring that the
    x±h perturbation never crosses the kink.

    Parameters
    ----------
    x : np.ndarray
        Input array (modified in-place if copy is not needed; a copy is returned
        regardless).
    h : float
        Finite-difference step size used for the numerical gradient check.
    kink_points : float or tuple of floats
        Location(s) of the C¹-discontinuous kink(s). For most neuron layers this is
        simply ``0.0``.
    margin : float
        Safety margin in units of *h*. Values closer than ``margin * h`` to a kink
        are pushed away. Default is ``2.0``, which guarantees x-h and x+h stay on
        the same side of the kink.

    Returns
    -------
    np.ndarray
        Array with near-kink points pushed away. Same shape and dtype as *x*.

    Notes
    -----
    - For C¹-continuous kinks (e.g., ELU with alpha=1 at x=0), this helper is
      **not** needed—instead, relax ``rtol`` to ~5e-3 to accommodate the O(h)
      second-derivative error across the C¹ kink.
    - This helper is idempotent: applying it twice produces the same result.
    """
    result = x.copy()
    threshold = margin * h
    if isinstance(kink_points, (int, float)):
        kinks = (float(kink_points),)
    else:
        kinks = tuple(float(k) for k in kink_points)
    for kink in kinks:
        # Points on the positive side of the kink (>= kink, but near it)
        pos_mask = (result >= kink) & (result < kink + threshold)
        result[pos_mask] = kink + threshold
        # Points on the negative side of the kink (< kink, but near it)
        neg_mask = (result < kink) & (result > kink - threshold)
        result[neg_mask] = kink - threshold
    return result
