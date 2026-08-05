import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_exp(data, test_dir, **kwargs):
    logger.debug(f"Exp params: {kwargs}")
    return _test_op(data, L.Exp, "Exp", test_dir, **kwargs)


def _exp_ref(x, base=-1.0, scale=1.0, shift=0.0):
    """Reference: Exp(x) = base^(shift + scale*x).
    If base == -1 (default), uses natural exponential: exp(shift + scale*x).
    """
    inner = shift + scale * x
    if base == -1.0:
        return np.exp(inner).astype(np.float32)
    return np.power(base, inner).astype(np.float32)


@pytest.mark.correctness
def test_exp_correctness(caffe_test_dir):
    """Exp correctness test with numpy reference."""
    logger.info("Running test_exp_correctness")
    np.random.seed(42)
    shapes = [(1, 3, 8, 8), (2, 5), (10,)]
    for shape in shapes:
        x = (np.random.rand(*shape).astype(np.float32) - 0.5) * 2
        ref = _exp_ref(x)
        caffe_out = _test_exp(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, op_name=f"Exp(shape={shape})")


@pytest.mark.correctness
def test_exp_base_scale_shift(caffe_test_dir):
    """Exp with base/scale/shift (Pattern C5: explicit param verification)."""
    logger.info("Running test_exp_base_scale_shift")
    np.random.seed(42)
    x = (np.random.rand(1, 3, 6, 6).astype(np.float32) - 0.5) * 2

    for base in [2.0, 10.0]:
        ref = _exp_ref(x, base=base)
        caffe_out = _test_exp(x, caffe_test_dir, exp_param={"base": base})
        assert_op_correct(caffe_out, ref, op_name=f"Exp(base={base})")

    ref = _exp_ref(x, scale=2.0, shift=1.0)
    caffe_out = _test_exp(x, caffe_test_dir, exp_param={"scale": 2.0, "shift": 1.0})
    assert_op_correct(caffe_out, ref, atol=1e-4, op_name="Exp(scale=2,shift=1)")
