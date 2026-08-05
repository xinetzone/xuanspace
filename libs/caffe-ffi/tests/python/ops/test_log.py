import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_log(data, test_dir, **kwargs):
    logger.debug(f"Log params: {kwargs}")
    return _test_op(data, L.Log, "Log", test_dir, **kwargs)


def _log_ref(x, base=-1.0, scale=1.0, shift=0.0):
    """Reference: Log(x) = log_base(shift + scale*x).
    If base == -1 (default), uses natural log: ln(shift + scale*x).
    """
    inner = shift + scale * x
    inner = np.maximum(inner, 1e-10)
    if base == -1.0:
        return np.log(inner).astype(np.float32)
    return (np.log(inner) / np.log(base)).astype(np.float32)


@pytest.mark.correctness
def test_log_correctness(caffe_test_dir):
    """Log correctness test with numpy reference."""
    logger.info("Running test_log_correctness")
    np.random.seed(42)
    shapes = [(1, 3, 8, 8), (2, 5), (10,)]
    for shape in shapes:
        x = np.random.rand(*shape).astype(np.float32) * 5 + 0.1
        ref = _log_ref(x)
        caffe_out = _test_log(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, op_name=f"Log(shape={shape})")


@pytest.mark.correctness
def test_log_base_scale_shift(caffe_test_dir):
    """Log with base/scale/shift (Pattern C5: explicit param verification)."""
    logger.info("Running test_log_base_scale_shift")
    np.random.seed(42)
    x = np.random.rand(1, 3, 6, 6).astype(np.float32) * 5 + 0.1

    for base in [2.0, 10.0]:
        ref = _log_ref(x, base=base)
        caffe_out = _test_log(x, caffe_test_dir, log_param={"base": base})
        assert_op_correct(caffe_out, ref, op_name=f"Log(base={base})")

    ref = _log_ref(x, scale=2.0, shift=1.0)
    caffe_out = _test_log(x, caffe_test_dir, log_param={"scale": 2.0, "shift": 1.0})
    assert_op_correct(caffe_out, ref, op_name="Log(scale=2,shift=1)")
