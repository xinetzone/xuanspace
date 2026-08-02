"""Phase 3.1: FFI 绑定集成测试 — set_shape_only / is_lazy_allocated.

验证 caffe_ffi Python 包中 set_shape_only 和 is_lazy_allocated 的
FFI 绑定正确性，覆盖 Blob 级 API 调用流程和生命周期转换。
"""

import pytest
import numpy as np

try:
    import caffe_ffi
    from caffe_ffi import Blob
    from .conftest import require_cpp_extension
except ImportError:
    pytest.skip("caffe_ffi not installed", allow_module_level=True)


# ── Prototxt helper ───────────────────────────────────────────────────

def _dims_str(dims):
    """Convert a dimension tuple to protobuf text format repeated dim entries."""
    return " ".join(f"dim: {d}" for d in dims)

def _make_split_prototxt(num_top, dims):
    dims_field = _dims_str(dims)
    tops = "\n".join(f'  top: "split_{i}"' for i in range(num_top))
    return f"""name: "ffi_test_split"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ {dims_field} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
{tops}
}}
"""


# ============================================================================
# TestSetShapeOnlyFFI — Blob-level FFI binding validation
# ============================================================================

class TestSetShapeOnlyFFI:
    """验证 set_shape_only / is_lazy_allocated FFI 绑定基本功能。"""

    # ── 基础绑定 ──────────────────────────────────────────────────────

    def test_set_shape_only_available(self):
        """set_shape_only 方法可通过 FFI 调用。"""
        blob = Blob()
        assert hasattr(blob, "set_shape_only"), \
            "set_shape_only FFI binding not found"

    def test_is_lazy_allocated_available(self):
        """is_lazy_allocated 方法可通过 FFI 调用。"""
        blob = Blob()
        assert hasattr(blob, "is_lazy_allocated"), \
            "is_lazy_allocated FFI binding not found"

    def test_default_not_lazy(self):
        """新创建的 Blob 默认不是 lazy 状态。"""
        blob = Blob()
        assert not blob.is_lazy_allocated(), \
            "New Blob should not be lazy-allocated"

    # ── set_shape_only 调用流程 ───────────────────────────────────────

    def test_set_shape_only_shape_access(self):
        """set_shape_only 后 shape/shape_at 返回正确的 shape。"""
        blob = Blob()
        blob.set_shape_only([32, 64, 112, 112])
        assert blob.num_axes == 4
        assert blob.shape_at(0) == 32
        assert blob.shape_at(1) == 64
        assert blob.shape_at(2) == 112
        assert blob.shape_at(3) == 112
        assert blob.shape == (32, 64, 112, 112)

    def test_set_shape_only_count(self):
        """set_shape_only 后 count() 返回正确值。"""
        blob = Blob()
        blob.set_shape_only([10, 20, 30])
        assert blob.count() == 10 * 20 * 30
        assert blob.count(1) == 20 * 30
        assert blob.count(0, 2) == 10 * 20

    def test_set_shape_only_no_data_access(self):
        """set_shape_only 后 is_lazy_allocated 为 True（data 未分配）。"""
        blob = Blob()
        blob.set_shape_only([64, 3, 32, 32])
        assert blob.is_lazy_allocated(), \
            "Lazy blob should have is_lazy_allocated() == True"
        # Shape metadata should still be accessible
        assert blob.num_axes == 4

    # ── 生命周期转换 ──────────────────────────────────────────────────

    def test_reshape_clears_lazy(self):
        """Reshape 调用后 lazy 标志被清除，data 可访问。"""
        blob = Blob()
        blob.set_shape_only([2, 10])
        assert blob.is_lazy_allocated()

        blob.Reshape([2, 10])
        assert not blob.is_lazy_allocated(), \
            "Reshape should clear lazy flag"
        assert blob.data_tensor is not None
        assert blob.data_tensor.shape == (2, 10)

    # ── 边界条件 ──────────────────────────────────────────────────────

    def test_set_shape_only_1d(self):
        """1 维 shape。"""
        blob = Blob()
        blob.set_shape_only([100])
        assert blob.num_axes == 1
        assert blob.shape_at(0) == 100
        assert blob.count() == 100

    def test_set_shape_only_empty_raises(self):
        """空 shape 列表应抛出异常。"""
        blob = Blob()
        with pytest.raises((ValueError, RuntimeError, Exception)):
            blob.set_shape_only([])

    def test_set_shape_only_negative_raises(self):
        """负维度应抛出异常。"""
        blob = Blob()
        with pytest.raises((ValueError, RuntimeError, Exception)):
            blob.set_shape_only([-1, 10])

    def test_set_shape_only_zero_raises(self):
        """零维度应抛出异常。"""
        blob = Blob()
        with pytest.raises((ValueError, RuntimeError, Exception)):
            blob.set_shape_only([0, 10])


# ============================================================================
# TestSetShapeOnlyLifecycle — 完整生命周期集成测试
# ============================================================================

class TestSetShapeOnlyLifecycle:
    """验证 SetShapeOnly → ShareData → 下游层 的完整生命周期。"""

    def test_full_lazy_to_shared_cycle(self):
        """完整流程：SetShapeOnly → ShareData → 数据验证。"""
        source = Blob([1, 64, 28, 28])
        src_data = np.random.randn(1, 64, 28, 28).astype(np.float32)
        source.set_data(src_data)
        target = Blob()

        # Phase 1: Lazy allocation
        target.set_shape_only([1, 64, 28, 28])
        assert target.is_lazy_allocated()

        # Phase 2: ShareData replaces lazy tensor
        target.ShareData(source)
        assert not target.is_lazy_allocated()
        assert target.SharesDataWith(source)

        # Phase 3: Shape metadata preserved
        assert target.num_axes == 4
        assert target.shape_at(0) == 1
        assert target.shape_at(1) == 64
        assert target.count() == 1 * 64 * 28 * 28

    def test_lazy_share_data_simulation(self):
        """模拟 Split 层行为：SetShapeOnly → ShareData → 共享验证。"""
        num_top = 64
        bottom = Blob([1, 3, 32, 32])
        bottom_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
        bottom.set_data(bottom_data)
        tops = [Blob() for _ in range(num_top)]

        # Simulate Split::Reshape with lazy allocation
        for top in tops:
            top.set_shape_only([1, 3, 32, 32])
            assert top.is_lazy_allocated()

        # Simulate Split::Forward with ShareData
        for top in tops:
            top.ShareData(bottom)
            assert not top.is_lazy_allocated()
            assert top.SharesDataWith(bottom)

        # Verify all tops share the same data
        for i in range(1, num_top):
            assert tops[i].SharesDataWith(tops[0]), \
                f"top[{i}] should share data with top[0]"

    def test_lazy_blob_mutable_data_triggers_allocation(self):
        """mutable_data_tensor() 在 lazy blob 上应触发防御性分配。"""
        blob = Blob()
        blob.set_shape_only([4, 8])
        assert blob.is_lazy_allocated()

        # mutable_data_tensor() should auto-allocate
        t = blob.mutable_data_tensor()
        assert t is not None, \
            "mutable_data_tensor() should auto-allocate for lazy blob"
        assert not blob.is_lazy_allocated(), \
            "mutable_data_tensor() should clear lazy flag after allocation"
        assert blob.data_tensor is not None
        assert blob.count() == 32


# ============================================================================
# TestSetShapeOnlySplitIntegration — Split 层集成测试（基于 Net/prototxt）
# ============================================================================

@require_cpp_extension
class TestSetShapeOnlySplitIntegration:
    """验证 Split 层在 Phase 3.1 下正确使用 SetShapeOnly（通过 Net API）。"""

    def test_split_n4_not_lazy(self):
        """N=4 不应触发 lazy 分配（低于 kLazyReshapeThreshold=16）。"""
        prototxt = _make_split_prototxt(4, (1, 3, 32, 32))
        net = caffe_ffi.Net(prototxt)

        for i in range(4):
            top = net.blob_by_name(f"split_{i}")
            assert not top.is_lazy_allocated(), \
                f"N=4 split_{i} should not be lazy"

        inp = np.random.randn(1, 3, 32, 32).astype(np.float32)
        out = net.Forward({"data": inp})
        for i in range(4):
            np.testing.assert_array_almost_equal(
                net.blob_by_name(f"split_{i}").to_numpy(), inp
            )

    def test_split_n64_lazy_reshape(self):
        """N=64 应触发 lazy 分配（高于 kLazyReshapeThreshold=16）。"""
        prototxt = _make_split_prototxt(64, (1, 3, 32, 32))
        net = caffe_ffi.Net(prototxt)

        for i in range(64):
            top = net.blob_by_name(f"split_{i}")
            assert top.is_lazy_allocated(), \
                f"N=64 split_{i} should be lazy after Reshape"

        inp = np.random.randn(1, 3, 32, 32).astype(np.float32)
        out = net.Forward({"data": inp})

        for i in range(64):
            top = net.blob_by_name(f"split_{i}")
            assert not top.is_lazy_allocated(), \
                f"split_{i} should not be lazy after Forward"
            np.testing.assert_array_almost_equal(top.to_numpy(), inp)

    def test_split_n64_downstream_relu(self):
        """N=64 Split + 下游 ReLU 层兼容性验证（通过 prototxt）。"""
        # Input shape: (1, 2, 4) — matching np.array shape below
        dims_field = _dims_str((1, 2, 4))
        tops = "\n".join(f'  top: "split_{i}"' for i in range(64))
        prototxt = f"""name: "ffi_test_split_relu"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ {dims_field} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
{tops}
}}
layer {{
  name: "relu"
  type: "ReLU"
  bottom: "split_0"
  top: "relu_out"
}}
"""
        net = caffe_ffi.Net(prototxt)
        inp = np.array([[[-1.0, 2.0, -3.0, 4.0],
                         [5.0, -6.0, 7.0, -8.0]]], dtype=np.float32)
        out = net.Forward({"data": inp})

        assert "relu_out" in out
        relu_data = net.blob_by_name("relu_out").to_numpy()
        expected = np.maximum(inp, 0)
        np.testing.assert_array_almost_equal(relu_data, expected)


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
