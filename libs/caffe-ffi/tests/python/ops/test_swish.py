import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_swish(data, test_dir, **kwargs):
    logger.debug(f"Swish params: {kwargs}")
    return _test_op(data, L.Swish, "Swish", test_dir, **kwargs)


def _swish_ref(x, beta=1.0):
    """Reference: Swish(x) = x * sigmoid(beta*x)."""
    return (x * (1.0 / (1.0 + np.exp(-beta * x)))).astype(np.float32)


@pytest.mark.correctness
def test_swish_correctness(caffe_test_dir):
    """Swish correctness test with numpy reference."""
    logger.info("Running test_swish_correctness")
    np.random.seed(42)
    shapes = [(1, 3, 8, 8), (2, 5), (3, 4, 4)]
    for shape in shapes:
        x = np.random.randn(*shape).astype(np.float32) * 3
        ref = _swish_ref(x, beta=1.0)
        caffe_out = _test_swish(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, op_name=f"Swish(shape={shape})")


@pytest.mark.correctness
def test_swish_beta(caffe_test_dir):
    """Swish with non-default beta parameter."""
    logger.info("Running test_swish_beta")
    np.random.seed(42)
    for beta in [0.1, 0.5, 2.0]:
        x = np.random.randn(1, 3, 6, 6).astype(np.float32) * 3
        ref = _swish_ref(x, beta=beta)
        caffe_out = _test_swish(x, caffe_test_dir, swish_param={"beta": beta})
        assert_op_correct(caffe_out, ref, op_name=f"Swish(beta={beta})")
