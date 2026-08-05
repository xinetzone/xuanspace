import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_threshold(data, test_dir, **kwargs):
    logger.debug(f"Threshold params: {kwargs}")
    return _test_op(data, L.Threshold, "Threshold", test_dir, **kwargs)


def _threshold_ref(x, threshold=0.0):
    """Reference: Threshold(x) = 1 if x > threshold else 0.

    IMPORTANT (Pattern C5 verified from source):
    Caffe ThresholdLayer performs BINARY thresholding, NOT pass-through!
    From threshold_layer.cpp line 21:
        top_data[i] = (bottom_data[i] > threshold_) ? Dtype(1) : Dtype(0);
    Output is always 0 or 1, regardless of input magnitude.
    """
    return (x > threshold).astype(np.float32)


@pytest.mark.correctness
def test_threshold_correctness(caffe_test_dir):
    """Threshold correctness test with numpy reference (Pattern C5: source-verified semantics)."""
    logger.info("Running test_threshold_correctness")
    np.random.seed(42)
    for thresh in [0.0, 0.5, 1.0, -0.5]:
        x = np.random.randn(1, 3, 6, 6).astype(np.float32) * 2
        ref = _threshold_ref(x, threshold=thresh)
        caffe_out = _test_threshold(x, caffe_test_dir, threshold_param={"threshold": thresh})
        assert_op_correct(caffe_out, ref, atol=0.0, rtol=0.0,
                          op_name=f"Threshold(threshold={thresh})")
        assert set(np.unique(caffe_out[0])).issubset({0.0, 1.0}), \
            f"Threshold output must be binary 0/1, got values: {np.unique(caffe_out[0])}"
