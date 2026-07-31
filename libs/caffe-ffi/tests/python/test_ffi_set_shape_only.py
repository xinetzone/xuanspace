"""Phase 3.1: FFI 绑定集成测试 — set_shape_only / is_lazy_allocated.

验证 caffe_ffi Python 包中 set_shape_only 和 is_lazy_allocated 的
FFI 绑定正确性，覆盖 Blob 级 API 调用流程和生命周期转换。
"""

import pytest

try:
    import caffe_ffi
except ImportError:
    pytest.skip("caffe_ffi not installed", allow_module_level=True)


# ============================================================================
# TestSetShapeOnlyFFI — Blob-level FFI binding validation
# ============================================================================

class TestSetShapeOnlyFFI:
    """验证 set_shape_only / is_lazy_allocated FFI 绑定基本功能。"""

    # ── 基础绑定 ──────────────────────────────────────────────────────

    def test_set_shape_only_available(self):
        """set_shape_only 方法可通过 FFI 调用。"""
        blob = caffe_ffi.Blob()
        assert hasattr(blob, "set_shape_only"), \
            "set_shape_only FFI binding not found"

    def test_is_lazy_allocated_available(self):
        """is_lazy_allocated 方法可通过 FFI 调用。"""
        blob = caffe_ffi.Blob()
        assert hasattr(blob, "is_lazy_allocated"), \
            "is_lazy_allocated FFI binding not found"

    def test_default_not_lazy(self):
        """新创建的 Blob 默认不是 lazy 状态。"""
        blob = caffe_ffi.Blob()
        assert not blob.is_lazy_allocated(), \
            "New Blob should not be lazy-allocated"

    # ── set_shape_only 调用流程 ───────────────────────────────────────

    def test_set_shape_only_basic(self):
        """set_shape_only 设置 shape 后 is_lazy_allocated 返回 True。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([2, 3, 224, 224])
        assert blob.is_lazy_allocated(), \
            "set_shape_only should set is_lazy_allocated=True"

    def test_set_shape_only_shape_access(self):
        """set_shape_only 后 shape() 返回正确的 shape。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([32, 64, 112, 112])
        assert blob.num_axes() == 4
        assert blob.shape(0) == 32
        assert blob.shape(1) == 64
        assert blob.shape(2) == 112
        assert blob.shape(3) == 112

    def test_set_shape_only_count(self):
        """set_shape_only 后 count() 返回正确值。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([10, 20, 30])
        assert blob.count() == 10 * 20 * 30
        assert blob.count(1) == 20 * 30
        assert blob.count(0, 2) == 10 * 20

    def test_set_shape_only_cpu_data_none(self):
        """set_shape_only 后 cpu_data() 返回 None（未分配内存）。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([64, 3, 32, 32])
        assert blob.cpu_data() is None, \
            "cpu_data() should return None for lazy blob (data_tensor_ undefined)"

    def test_set_shape_only_cpu_diff_none(self):
        """set_shape_only 后 cpu_diff() 返回 None。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([64, 3, 32, 32])
        assert blob.cpu_diff() is None, \
            "cpu_diff() should return None for lazy blob (diff_tensor_ undefined)"

    # ── 生命周期转换 ──────────────────────────────────────────────────

    def test_reshape_clears_lazy(self):
        """Reshape 调用后 lazy 标志被清除。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([10, 20])
        assert blob.is_lazy_allocated()
        blob.reshape([5, 5])
        assert not blob.is_lazy_allocated(), \
            "Reshape should clear lazy flag"
        assert blob.cpu_data() is not None

    def test_share_data_clears_lazy(self):
        """ShareData 调用后 lazy 标志被清除。"""
        source = caffe_ffi.Blob([3, 4])
        target = caffe_ffi.Blob()
        target.set_shape_only([3, 4])
        assert target.is_lazy_allocated()
        target.share_data(source)
        assert not target.is_lazy_allocated(), \
            "ShareData should clear lazy flag"
        assert target.shares_data_with(source)
        assert target.cpu_data() == source.cpu_data()

    def test_from_proto_clears_lazy(self):
        """FromProto 调用后 lazy 标志被清除（通过 Reshape 间接触发）。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([2, 10])
        assert blob.is_lazy_allocated()

        # Construct a minimal BlobProto
        proto = caffe_ffi.BlobProto()
        proto.shape.extend([2, 10])
        proto.data.extend([0.0] * 20)
        blob.from_proto(proto)
        assert not blob.is_lazy_allocated(), \
            "FromProto should clear lazy flag (via Reshape)"
        assert blob.cpu_data() is not None

    # ── 双重 set_shape_only ───────────────────────────────────────────

    def test_double_set_shape_only(self):
        """连续两次 set_shape_only 覆盖前一次 shape。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([10, 20])
        assert blob.count() == 200
        blob.set_shape_only([5, 5, 3])
        assert blob.is_lazy_allocated()
        assert blob.count() == 75
        assert blob.num_axes() == 3
        assert blob.shape(0) == 5

    # ── 边界条件 ──────────────────────────────────────────────────────

    def test_set_shape_only_1d(self):
        """1 维 shape。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([100])
        assert blob.num_axes() == 1
        assert blob.shape(0) == 100
        assert blob.count() == 100

    def test_set_shape_only_empty_raises(self):
        """空 shape 列表应抛出异常。"""
        blob = caffe_ffi.Blob()
        with pytest.raises((ValueError, RuntimeError)):
            blob.set_shape_only([])

    def test_set_shape_only_negative_raises(self):
        """负维度应抛出异常。"""
        blob = caffe_ffi.Blob()
        with pytest.raises((ValueError, RuntimeError)):
            blob.set_shape_only([-1, 10])

    def test_set_shape_only_zero_raises(self):
        """零维度应抛出异常。"""
        blob = caffe_ffi.Blob()
        with pytest.raises((ValueError, RuntimeError)):
            blob.set_shape_only([0, 10])


# ============================================================================
# TestSetShapeOnlyLifecycle — 完整生命周期集成测试
# ============================================================================

class TestSetShapeOnlyLifecycle:
    """验证 SetShapeOnly → ShareData → 下游层 的完整生命周期。"""

    def test_full_lazy_to_shared_cycle(self):
        """完整流程：SetShapeOnly → ShareData → 下游层读取。"""
        source = caffe_ffi.Blob([1, 64, 28, 28])
        target = caffe_ffi.Blob()

        # Phase 1: Lazy allocation
        target.set_shape_only([1, 64, 28, 28])
        assert target.is_lazy_allocated()
        assert target.cpu_data() is None

        # Phase 2: ShareData replaces lazy tensor
        target.share_data(source)
        assert not target.is_lazy_allocated()
        assert target.cpu_data() is not None
        assert target.cpu_data() == source.cpu_data()
        assert target.shares_data_with(source)

        # Phase 3: Shape metadata preserved
        assert target.num_axes() == 4
        assert target.shape(0) == 1
        assert target.shape(1) == 64
        assert target.count() == 1 * 64 * 28 * 28

    def test_lazy_reshape_then_forward_like_split(self):
        """模拟 Split 层行为：SetShapeOnly → Forward(ShareData) → 下游读取。"""
        num_top = 64
        bottom = caffe_ffi.Blob([1, 3, 32, 32])
        tops = [caffe_ffi.Blob() for _ in range(num_top)]

        # Simulate Split::Reshape with lazy allocation
        for top in tops:
            top.set_shape_only([1, 3, 32, 32])
            assert top.is_lazy_allocated()

        # Simulate Split::Forward with ShareData
        for top in tops:
            top.share_data(bottom)
            assert not top.is_lazy_allocated()
            assert top.cpu_data() == bottom.cpu_data()

        # Verify all tops share the same data
        for i in range(1, num_top):
            assert tops[i].shares_data_with(tops[0]), \
                f"top[{i}] should share data with top[0]"

    def test_lazy_blob_cpu_mutable_data_triggers_allocation(self):
        """cpu_mutable_data() 在 lazy blob 上应触发防御性分配。"""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([4, 8])
        assert blob.is_lazy_allocated()
        assert blob.cpu_data() is None

        # cpu_mutable_data() should auto-allocate
        data = blob.cpu_mutable_data()
        assert data is not None, \
            "cpu_mutable_data() should auto-allocate for lazy blob"
        assert not blob.is_lazy_allocated(), \
            "cpu_mutable_data() should clear lazy flag after allocation"
        assert blob.cpu_data() is not None
        assert blob.count() == 32


# ============================================================================
# TestSetShapeOnlySplitIntegration — Split 层集成测试
# ============================================================================

class TestSetShapeOnlySplitIntegration:
    """验证 Split 层在 Phase 3.1 下正确使用 SetShapeOnly。"""

    def test_split_layer_available(self):
        """Split 层可通过 FFI 创建。"""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "test_split", num_top=4)
        assert split is not None

    def test_split_n4_reshape_small(self):
        """N=4 不应触发 lazy 分配（低于 kLazyReshapeThreshold=16）。"""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split4", num_top=4)
        bottom = caffe_ffi.Blob([1, 3, 32, 32])
        tops = [caffe_ffi.Blob() for _ in range(4)]

        split.reshape([bottom], tops)

        for top in tops:
            assert not top.is_lazy_allocated(), \
                "N=4 should not trigger lazy allocation"
            assert top.cpu_data() is not None

    def test_split_n64_reshape_large(self):
        """N=64 应触发 lazy 分配（高于 kLazyReshapeThreshold=16）。"""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split64", num_top=64)
        bottom = caffe_ffi.Blob([1, 3, 32, 32])
        tops = [caffe_ffi.Blob() for _ in range(64)]

        split.reshape([bottom], tops)

        for top in tops:
            assert top.is_lazy_allocated(), \
                f"N=64 should trigger lazy allocation, but top is not lazy"
            assert top.cpu_data() is None, \
                "Lazy blob should have cpu_data() == None"

    def test_split_n64_forward_transitions_to_normal(self):
        """N=64 Split Forward 后所有 top 转为 normal 状态。"""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split64", num_top=64)
        bottom = caffe_ffi.Blob([1, 3, 32, 32])
        tops = [caffe_ffi.Blob() for _ in range(64)]

        split.reshape([bottom], tops)
        split.forward([bottom], tops)

        for top in tops:
            assert not top.is_lazy_allocated(), \
                "After Forward, lazy flag should be cleared by ShareData"
            assert top.shares_data_with(bottom), \
                "After Forward, top should share data with bottom"
            assert top.cpu_data() is not None

    def test_split_n64_forward_data_correctness(self):
        """N=64 Split Forward 后数据正确性验证。"""
        import numpy as np

        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split64", num_top=64)

        # Create bottom with known data
        bottom = caffe_ffi.Blob([2, 4])
        data = np.array([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        bottom.cpu_mutable_data()[:] = data.ravel()

        tops = [caffe_ffi.Blob() for _ in range(64)]

        split.reshape([bottom], tops)
        split.forward([bottom], tops)

        # All tops should reference the same data
        for i, top in enumerate(tops):
            top_data = np.array(top.cpu_data())
            top_data = top_data.reshape(2, 4)
            assert np.allclose(top_data, data), \
                f"top[{i}] data mismatch"

    def test_split_n64_downstream_relu_compatibility(self):
        """N=64 Split + 下游 ReLU 层兼容性验证。"""
        import numpy as np

        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split64", num_top=64)
        relu = net.add_layer("ReLU", "relu")

        bottom = caffe_ffi.Blob([2, 4])
        data = np.array([[-1.0, 2.0, -3.0, 4.0],
                         [5.0, -6.0, 7.0, -8.0]], dtype=np.float32)
        bottom.cpu_mutable_data()[:] = data.ravel()

        tops = [caffe_ffi.Blob() for _ in range(64)]

        split.reshape([bottom], tops)
        split.forward([bottom], tops)

        # Downstream ReLU on top[0]
        relu_out = caffe_ffi.Blob([2, 4])
        relu.reshape([tops[0]], [relu_out])
        relu.forward([tops[0]], [relu_out])

        relu_data = np.array(relu_out.cpu_data()).reshape(2, 4)
        expected = np.maximum(data, 0)
        assert np.allclose(relu_data, expected), \
            "ReLU output mismatch after lazy Split"

        # Verify top[0] is still shared with bottom (COW triggered on write)
        # top[0] should still share with bottom because ReLU reads only
        assert tops[0].shares_data_with(bottom), \
            "top[0] should still share with bottom after ReLU (read-only)"


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])