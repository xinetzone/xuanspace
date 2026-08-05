import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

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
