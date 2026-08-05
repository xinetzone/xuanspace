import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_elu(data, test_dir, **kwargs):
    logger.debug(f"ELU params: {kwargs}")
    return _test_op(data, L.ELU, "ELU", test_dir, **kwargs)


def _elu_ref(x, alpha=1.0):
    """Reference: ELU(x) = x if x > 0, alpha*(exp(x)-1) if x <= 0."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1)).astype(np.float32)


@pytest.mark.correctness
def test_elu_correctness(caffe_test_dir):
    """ELU correctness test with numpy reference."""
    logger.info("Running test_elu_correctness")
    np.random.seed(42)
    shapes = [(1, 3, 8, 8), (2, 5), (3, 4, 4)]
    for shape in shapes:
        x = np.random.randn(*shape).astype(np.float32) * 2
        ref = _elu_ref(x, alpha=1.0)
        caffe_out = _test_elu(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, op_name=f"ELU(shape={shape})")


@pytest.mark.correctness
def test_elu_alpha(caffe_test_dir):
    """ELU with non-default alpha parameter."""
    logger.info("Running test_elu_alpha")
    np.random.seed(42)
    for alpha in [0.1, 0.5, 2.0]:
        x = np.random.randn(1, 3, 6, 6).astype(np.float32) * 2
        ref = _elu_ref(x, alpha=alpha)
        caffe_out = _test_elu(x, caffe_test_dir, elu_param={"alpha": alpha})
        assert_op_correct(caffe_out, ref, op_name=f"ELU(alpha={alpha})")
