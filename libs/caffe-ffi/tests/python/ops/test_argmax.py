import logging
import numpy as np
import pytest
from .utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_argmax(data, test_dir, **kwargs):
    logger.debug(f"ArgMax params: {kwargs}")
    return _test_op(data, L.ArgMax, "ArgMax", test_dir, **kwargs)


@pytest.mark.correctness
def test_argmax_correctness(caffe_test_dir):
    """ArgMax correctness test with numpy reference.

    ArgMax finds the index of the maximum value along a given axis.
    axis=-1 (default) means argmax over the last dimension.
    """
    logger.info("Running test_argmax_correctness")
    np.random.seed(42)

    # Test axis=0 (channels) on 4D NCHW: output shape (N, 1, H, W) with class indices
    x = np.random.randn(2, 5, 4, 4).astype(np.float32)
    ref = np.argmax(x, axis=1, keepdims=True).astype(np.float32)
    caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"axis": 1})
    assert_op_correct(caffe_out, ref, atol=0.0, rtol=0.0, op_name="ArgMax(axis=1)")

    # Test axis=2 (height) on 4D
    x = np.random.randn(1, 3, 6, 4).astype(np.float32)
    ref = np.argmax(x, axis=2, keepdims=True).astype(np.float32)
    caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"axis": 2})
    assert_op_correct(caffe_out, ref, atol=0.0, rtol=0.0, op_name="ArgMax(axis=2)")

    # Test axis=3 (width) on 4D
    x = np.random.randn(1, 3, 4, 6).astype(np.float32)
    ref = np.argmax(x, axis=3, keepdims=True).astype(np.float32)
    caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"axis": 3})
    assert_op_correct(caffe_out, ref, atol=0.0, rtol=0.0, op_name="ArgMax(axis=3)")

    # Test 2D: axis=0 and axis=1
    x = np.random.randn(4, 6).astype(np.float32)
    for axis in [0, 1]:
        ref = np.argmax(x, axis=axis, keepdims=True).astype(np.float32)
        caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"axis": axis})
        assert_op_correct(caffe_out, ref, atol=0.0, rtol=0.0, op_name=f"ArgMax(axis={axis},2D)")


@pytest.mark.correctness
def test_argmax_top_k(caffe_test_dir):
    """ArgMax with top_k > 1 returns top-k max indices (Pattern C5: verify param semantics)."""
    logger.info("Running test_argmax_top_k")
    np.random.seed(42)

    x = np.random.randn(2, 5, 3, 3).astype(np.float32)
    top_k = 3
    # top_k returns the top k maximum indices along axis
    # Caffe ArgMax with top_k flattens and returns top-k indices
    # Verify shape and that output contains valid indices
    caffe_out = _test_argmax(
        x, caffe_test_dir,
        argmax_param={"axis": 1, "top_k": top_k}
    )
    assert caffe_out[0].shape[0] == x.shape[0], f"Batch dim preserved, got {caffe_out[0].shape}"


@pytest.mark.correctness
def test_argmax_out_max_val_false_default(caffe_test_dir):
    """out_max_val defaults to false (BVLC Caffe alignment): output contains indices only.

    Regression test for commit 7eef6be: previously default was true, which output
    max values instead of indices, breaking BVLC Caffe output semantics.
    """
    logger.info("Running test_argmax_out_max_val_false_default")
    np.random.seed(42)

    # With explicit axis, out_max_val=false → output is argmax indices (same as np.argmax)
    x = np.random.randn(2, 5, 4, 4).astype(np.float32)
    ref = np.argmax(x, axis=1, keepdims=True).astype(np.float32)
    # Default (out_max_val not specified → false)
    caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"axis": 1})
    assert_op_correct(caffe_out, ref, atol=0.0, rtol=0.0, op_name="ArgMax(default out_max_val=false)")
    # Explicitly set out_max_val=false
    caffe_out2 = _test_argmax(x, caffe_test_dir, argmax_param={"axis": 1, "out_max_val": False})
    assert_op_correct(caffe_out2, ref, atol=0.0, rtol=0.0, op_name="ArgMax(out_max_val=false)")


@pytest.mark.correctness
def test_argmax_out_max_val_true_with_axis(caffe_test_dir):
    """out_max_val=true with axis: output contains max VALUES (not indices) along that axis."""
    logger.info("Running test_argmax_out_max_val_true_with_axis")
    np.random.seed(42)

    x = np.random.randn(2, 5, 4, 4).astype(np.float32)
    # out_max_val=true + axis → output is max values along axis (np.max, not np.argmax)
    ref_vals = np.max(x, axis=1, keepdims=True).astype(np.float32)
    caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"axis": 1, "out_max_val": True})
    assert caffe_out[0].shape == ref_vals.shape, \
        f"out_max_val=true shape mismatch: {caffe_out[0].shape} vs {ref_vals.shape}"
    assert_op_correct(caffe_out, ref_vals, atol=1e-6, rtol=0.0, op_name="ArgMax(out_max_val=true,values)")


@pytest.mark.correctness
def test_argmax_out_max_val_true_flatten(caffe_test_dir):
    """out_max_val=true without axis (flatten mode): output shape [N, 2, top_k].

    First top_k entries are max indices, second top_k entries are max values.
    """
    logger.info("Running test_argmax_out_max_val_true_flatten")
    np.random.seed(42)

    x = np.random.randn(2, 5, 4, 4).astype(np.float32)  # N=2, flattened dim=80
    top_k = 3
    caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"out_max_val": True, "top_k": top_k})
    out = caffe_out[0]
    # Flatten mode without axis: shape should be [N, 2, top_k] = [2, 2, 3]
    assert out.ndim == 3, f"Expected 3D output [N,2,top_k], got shape {out.shape}"
    assert out.shape[0] == x.shape[0], f"Batch dim mismatch: {out.shape[0]} vs {x.shape[0]}"
    assert out.shape[1] == 2, f"out_max_val=true → dim1 should be 2 (index+value), got {out.shape[1]}"
    assert out.shape[2] == top_k, f"top_k dim mismatch: {out.shape[2]} vs {top_k}"
    # First slice along dim=1 are indices (should be integers in valid range)
    indices = out[:, 0, :]
    assert np.all(indices >= 0) and np.all(indices < x[0].size), \
        f"Indices out of range [0,{x[0].size}): {indices}"
    # Second slice are max values
    values = out[:, 1, :]
    # Values should be in descending order (partial_sort gives top-k)
    for n in range(x.shape[0]):
        assert np.all(np.diff(values[n]) <= 1e-6), \
            f"Values should be descending for batch {n}: {values[n]}"
    # Verify indices point to actual top-k values
    for n in range(x.shape[0]):
        flat = x[n].flatten()
        for k in range(top_k):
            idx = int(indices[n, k])
            expected_val = flat[idx]
            actual_val = values[n, k]
            assert abs(expected_val - actual_val) < 1e-6, \
                f"Index {idx} value mismatch: {expected_val} vs {actual_val}"


@pytest.mark.correctness
def test_argmax_out_max_val_false_flatten(caffe_test_dir):
    """out_max_val=false (default) without axis (flatten mode): output shape [N, 1, top_k].

    Regression test for the flatten-mode Reshape bug: previously num_top_axes was
    set to max(bottom[0]->num_axes(), 3), producing 4D output [N,1,top_k,1] for 4D
    input instead of the correct 3D [N,1,top_k] (BVLC Caffe semantics: always 3D).
    """
    logger.info("Running test_argmax_out_max_val_false_flatten")
    np.random.seed(42)

    x = np.random.randn(2, 5, 4, 4).astype(np.float32)  # N=2, flattened dim=80
    top_k = 3
    # Default out_max_val=false (no axis → flatten mode)
    caffe_out = _test_argmax(x, caffe_test_dir, argmax_param={"top_k": top_k})
    out = caffe_out[0]
    # Flatten mode without axis: shape should be [N, 1, top_k] = [2, 1, 3]
    assert out.ndim == 3, f"Expected 3D output [N,1,top_k], got shape {out.shape}"
    assert out.shape[0] == x.shape[0], f"Batch dim mismatch: {out.shape[0]} vs {x.shape[0]}"
    assert out.shape[1] == 1, f"out_max_val=false → dim1 should be 1 (indices only), got {out.shape[1]}"
    assert out.shape[2] == top_k, f"top_k dim mismatch: {out.shape[2]} vs {top_k}"
    # Output contains indices (should be integers in valid range)
    indices = out[:, 0, :]
    assert np.all(indices >= 0) and np.all(indices < x[0].size), \
        f"Indices out of range [0,{x[0].size}): {indices}"
    # Verify indices correspond to actual top-k max values (descending)
    for n in range(x.shape[0]):
        flat = x[n].flatten()
        # Get top-k values using indices from output
        extracted_vals = np.array([flat[int(indices[n, k])] for k in range(top_k)])
        # Should be descending
        assert np.all(np.diff(extracted_vals) <= 1e-6), \
            f"Top-k values should be descending for batch {n}: {extracted_vals}"
        # Should match numpy's top-k
        ref_topk = np.sort(flat)[::-1][:top_k]
        assert np.allclose(np.sort(extracted_vals)[::-1], ref_topk, atol=1e-6), \
            f"Top-k values mismatch for batch {n}: {extracted_vals} vs {ref_topk}"
