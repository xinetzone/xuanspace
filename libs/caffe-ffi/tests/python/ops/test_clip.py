import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_clip(data, test_dir, **kwargs):
    logger.debug(f"Clip params: {kwargs}")
    return _test_op(data, L.Clip, "Clip", test_dir, **kwargs)


def _clip_ref(x, min_val, max_val):
    """Reference: Clip(x) = min(max(x, min_val), max_val)."""
    return np.clip(x, min_val, max_val).astype(np.float32)


@pytest.mark.correctness
def test_clip_correctness(caffe_test_dir):
    """Clip correctness test with numpy reference."""
    logger.info("Running test_clip_correctness")
    np.random.seed(42)
    test_cases = [
        ((1, 3, 8, 8), -1.0, 1.0),
        ((2, 5), 0.0, 5.0),
        ((3, 4, 4), -2.0, 2.0),
        ((10,), -0.5, 0.5),
    ]
    for shape, min_val, max_val in test_cases:
        x = np.random.randn(*shape).astype(np.float32) * 3
        ref = _clip_ref(x, min_val, max_val)
        caffe_out = _test_clip(x, caffe_test_dir, clip_param={"min": min_val, "max": max_val})
        assert_op_correct(caffe_out, ref, op_name=f"Clip(min={min_val},max={max_val})")
