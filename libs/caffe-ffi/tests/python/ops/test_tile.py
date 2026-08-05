import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_tile(data, test_dir, **kwargs):
    logger.debug(f"Tile params: {kwargs}")
    return _test_op(data, L.Tile, "Tile", test_dir, **kwargs)


@pytest.mark.correctness
def test_tile_correctness(caffe_test_dir):
    """Tile correctness test with numpy reference (Pattern C5: axis broadcasting risk verified).

    Tile repeats the input along a specified axis `tiles` times.
    Unlike broadcasting, tile's axis parameter explicitly selects ONE axis to tile,
    avoiding the Crop-style implicit coordinate broadcast problem.
    axis=0 is batch dim (be careful with batch), axis=1 is channels, etc.
    """
    logger.info("Running test_tile_correctness")
    np.random.seed(42)

    # Tile along axis=1 (channels) - safest axis to test
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    for tiles in [2, 3]:
        ref = np.tile(x, (1, tiles, 1, 1)).astype(np.float32)
        caffe_out = _test_tile(x, caffe_test_dir, tile_param={"axis": 1, "tiles": tiles})
        assert_op_correct(caffe_out, ref, op_name=f"Tile(axis=1,tiles={tiles})")
        assert caffe_out[0].shape[1] == x.shape[1] * tiles

    # Tile along axis=2 (height)
    x = np.random.randn(1, 2, 3, 4).astype(np.float32)
    tiles = 2
    ref = np.tile(x, (1, 1, tiles, 1)).astype(np.float32)
    caffe_out = _test_tile(x, caffe_test_dir, tile_param={"axis": 2, "tiles": tiles})
    assert_op_correct(caffe_out, ref, op_name=f"Tile(axis=2,tiles={tiles})")
    assert caffe_out[0].shape[2] == x.shape[2] * tiles

    # Tile along axis=3 (width)
    x = np.random.randn(1, 2, 4, 3).astype(np.float32)
    tiles = 2
    ref = np.tile(x, (1, 1, 1, tiles)).astype(np.float32)
    caffe_out = _test_tile(x, caffe_test_dir, tile_param={"axis": 3, "tiles": tiles})
    assert_op_correct(caffe_out, ref, op_name=f"Tile(axis=3,tiles={tiles})")
    assert caffe_out[0].shape[3] == x.shape[3] * tiles


@pytest.mark.correctness
def test_tile_no_broadcast_issue(caffe_test_dir):
    """Verify Tile does NOT suffer from Crop-style parameter broadcasting.

    Pattern C5 insight: Crop's axis/offset created ambiguity when scalar offset
    was implicitly broadcast across remaining dims. Tile's axis parameter selects
    exactly ONE axis, and tiles is a scalar count - no multi-dim broadcasting.
    This test verifies explicit axis selection works correctly.
    """
    logger.info("Running test_tile_no_broadcast_issue")
    np.random.seed(42)
    x = np.random.randn(1, 2, 3, 3).astype(np.float32)
    # Tile axis=2 should ONLY affect height, not width
    ref = np.tile(x, (1, 1, 2, 1)).astype(np.float32)
    caffe_out = _test_tile(x, caffe_test_dir, tile_param={"axis": 2, "tiles": 2})
    assert_op_correct(caffe_out, ref, op_name="Tile(axis=2-only-broadcast-check)")
    assert caffe_out[0].shape == (1, 2, 6, 3), f"Width should NOT change: {caffe_out[0].shape}"
