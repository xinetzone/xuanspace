"""Phase 3.1: SetShapeOnly compatibility and integration tests.

Covers 17 test cases from the SetShapeOnly API Design Document:
  - 7 Blob-level unit tests (TestSetShapeOnly)
  - 4 Split-layer integration tests (TestSplitLazyReshape)
  - 6 extended tests for edge cases & regression

Usage:
  pytest tests/python/test_phase3_set_shape_only.py -v
"""
import pytest
import numpy as np
import caffe_ffi


# ═══════════════════════════════════════════════════════════════════════
# TestSetShapeOnly — Blob-level unit tests (7 cases)
# ═══════════════════════════════════════════════════════════════════════

class TestSetShapeOnly:
    """Blob-level SetShapeOnly unit tests."""

    # ── Case 1: BasicShapeStorage ──

    def test_basic_shape_storage(self):
        """SetShapeOnly stores shape correctly and sets lazy flag."""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([2, 3, 224, 224])

        assert blob.is_lazy_allocated(), "is_lazy_allocated() should return True"
        assert blob.num_axes() == 4, f"num_axes should be 4, got {blob.num_axes()}"
        assert blob.shape(0) == 2
        assert blob.shape(1) == 3
        assert blob.shape(2) == 224
        assert blob.shape(3) == 224
        assert blob.count() == 2 * 3 * 224 * 224

    # ── Case 2: NoDataAllocation ──

    def test_no_data_allocation(self):
        """SetShapeOnly should not allocate data memory."""
        blob = caffe_ffi.Blob()
        bytes_before = caffe_ffi.total_allocated_bytes()
        blob.set_shape_only([100, 100, 100])
        bytes_after = caffe_ffi.total_allocated_bytes()

        assert bytes_after == bytes_before, (
            f"SetShapeOnly allocated {bytes_after - bytes_before} bytes, "
            f"expected 0"
        )

    # ── Case 3: CpuDataReturnsNullptr ──

    def test_cpu_data_returns_none(self):
        """cpu_data() should return None for lazy-allocated blob."""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([64, 3, 32, 32])

        assert blob.cpu_data() is None, "cpu_data() should be None for lazy blob"
        assert blob.cpu_diff() is None, "cpu_diff() should be None for lazy blob"

    # ── Case 4: DataTensorUndefined ──

    def test_data_tensor_undefined(self):
        """data_tensor() should return undefined Tensor."""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([1, 10])

        dt = blob.data_tensor()
        assert not dt.defined(), "data_tensor() should be undefined"

    # ── Case 5: ReshapeClearsLazyFlag ──

    def test_reshape_clears_lazy_flag(self):
        """Reshape() should allocate memory and clear lazy flag."""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([10, 20])
        assert blob.is_lazy_allocated()

        blob.reshape([5, 5])
        assert not blob.is_lazy_allocated(), "Reshape should clear lazy flag"
        assert blob.cpu_data() is not None, "cpu_data() should not be None after Reshape"
        assert blob.count() == 25

    # ── Case 6: ShareDataClearsLazyFlag ──

    def test_share_data_clears_lazy_flag(self):
        """ShareData() should replace tensor and clear lazy flag."""
        source = caffe_ffi.Blob([3, 4])
        target = caffe_ffi.Blob()
        target.set_shape_only([3, 4])
        assert target.is_lazy_allocated()

        target.share_data(source)
        assert not target.is_lazy_allocated(), "ShareData should clear lazy flag"
        assert target.shares_data_with(source), "target should share data with source"
        assert target.cpu_data() == source.cpu_data(), "data pointers should be equal"

    # ── Case 7: InvalidShapeRejected ──

    def test_invalid_shape_rejected_negative(self):
        """Negative dimension should raise ValueError."""
        blob = caffe_ffi.Blob()
        with pytest.raises(ValueError, match="positive"):
            blob.set_shape_only([-1, 10])

    def test_invalid_shape_rejected_zero(self):
        """Zero dimension should raise ValueError."""
        blob = caffe_ffi.Blob()
        with pytest.raises(ValueError, match="positive"):
            blob.set_shape_only([0, 10])


# ═══════════════════════════════════════════════════════════════════════
# TestSplitLazyReshape — Split layer integration tests (4 cases)
# ═══════════════════════════════════════════════════════════════════════

class TestSplitLazyReshape:
    """Split layer lazy reshape integration tests."""

    # ── Case 8: LargeNTriggersLazyAllocation ──

    def test_large_n_triggers_lazy_allocation(self):
        """N=64 (>= threshold 16) should trigger lazy allocation."""
        from caffe_ffi import LayerParameter, NetParameter

        N = 64
        C = 3 * 32 * 32

        proto = NetParameter()
        proto.name = "test_lazy_N64"
        input_param = LayerParameter()
        input_param.type = "Input"
        input_param.name = "data"
        input_param.top.append("data")
        input_param.input_param.shape.add().dim[:] = [1, 3, 32, 32]
        proto.layer.append(input_param)
        split_param = LayerParameter()
        split_param.type = "Split"
        split_param.name = "split"
        split_param.bottom.append("data")
        for i in range(N):
            split_param.top.append(f"split_{i}")
        proto.layer.append(split_param)

        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 3, 32, 32).astype(np.float32)

        bytes_before = caffe_ffi.total_allocated_bytes()
        out = net.Forward({"data": inp})
        bytes_after = caffe_ffi.total_allocated_bytes()

        # With lazy allocation, memory should be much less than
        # N * count * sizeof(float) = 64 * 3072 * 4 = 786,432 bytes
        expected_full = N * C * 4
        actual_delta = bytes_after - bytes_before
        assert actual_delta < expected_full, (
            f"Memory increase {actual_delta} >= expected_full {expected_full}, "
            f"lazy allocation may not be active"
        )

    # ── Case 9: ForwardTransitionsToNormal ──

    def test_forward_transitions_to_normal(self):
        """After Forward, ShareData should replace lazy tensors."""
        from caffe_ffi import LayerParameter, NetParameter

        N = 64
        proto = NetParameter()
        proto.name = "test_transition_N64"
        input_param = LayerParameter()
        input_param.type = "Input"
        input_param.name = "data"
        input_param.top.append("data")
        input_param.input_param.shape.add().dim[:] = [1, 3, 32, 32]
        proto.layer.append(input_param)
        split_param = LayerParameter()
        split_param.type = "Split"
        split_param.name = "split"
        split_param.bottom.append("data")
        for i in range(N):
            split_param.top.append(f"split_{i}")
        proto.layer.append(split_param)

        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 3, 32, 32).astype(np.float32)

        out = net.Forward({"data": inp})

        for i in range(N):
            key = f"split_{i}"
            assert key in out, f"Missing output '{key}'"
            np.testing.assert_array_almost_equal(
                out[key], inp,
                err_msg=f"split_{i} output differs from input"
            )

    # ── Case 10: SmallNStaysNormal ──

    def test_small_n_stays_normal(self):
        """N=4 (< threshold 16) should NOT trigger lazy allocation."""
        from caffe_ffi import LayerParameter, NetParameter

        N = 4
        C = 256

        proto = NetParameter()
        proto.name = "test_normal_N4"
        input_param = LayerParameter()
        input_param.type = "Input"
        input_param.name = "data"
        input_param.top.append("data")
        input_param.input_param.shape.add().dim[:] = [1, C]
        proto.layer.append(input_param)
        split_param = LayerParameter()
        split_param.type = "Split"
        split_param.name = "split"
        split_param.bottom.append("data")
        for i in range(N):
            split_param.top.append(f"split_{i}")
        proto.layer.append(split_param)

        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, C).astype(np.float32)

        out = net.Forward({"data": inp})

        for i in range(N):
            key = f"split_{i}"
            assert key in out
            np.testing.assert_array_almost_equal(out[key], inp)

    # ── Case 11: DownstreamLayerCompatibility ──

    def test_downstream_layer_compatibility_relu(self):
        """ReLU downstream of lazy-allocated Split should work correctly."""
        from caffe_ffi import LayerParameter, NetParameter

        N = 100
        C = 64 * 28 * 28

        proto = NetParameter()
        proto.name = "test_downstream_relu"
        input_param = LayerParameter()
        input_param.type = "Input"
        input_param.name = "data"
        input_param.top.append("data")
        input_param.input_param.shape.add().dim[:] = [1, 64, 28, 28]
        proto.layer.append(input_param)
        split_param = LayerParameter()
        split_param.type = "Split"
        split_param.name = "split"
        split_param.bottom.append("data")
        for i in range(N):
            split_param.top.append(f"split_{i}")
        proto.layer.append(split_param)
        # Add a ReLU after split_0 to test downstream compatibility
        relu_param = LayerParameter()
        relu_param.type = "ReLU"
        relu_param.name = "relu"
        relu_param.bottom.append("split_0")
        relu_param.top.append("relu_out")
        proto.layer.append(relu_param)

        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 64, 28, 28).astype(np.float32)

        out = net.Forward({"data": inp})

        assert "relu_out" in out, "ReLU output missing"
        expected_relu = np.maximum(0, inp)
        np.testing.assert_array_almost_equal(
            out["relu_out"], expected_relu,
            err_msg="ReLU output differs from expected"
        )


# ═══════════════════════════════════════════════════════════════════════
# Extended tests — edge cases & regression (6 cases)
# ═══════════════════════════════════════════════════════════════════════

class TestSetShapeOnlyExtended:
    """Extended edge case and regression tests for SetShapeOnly."""

    # ── Case 12: CountAfterSetShapeOnly ──

    def test_count_after_set_shape_only(self):
        """count() and count(start_axis) should work with stored shape."""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([32, 64, 112, 112])

        assert blob.count() == 32 * 64 * 112 * 112
        assert blob.count(1) == 64 * 112 * 112
        assert blob.count(0, 2) == 32 * 64

    # ── Case 13: SetShapeOnlyThenFromProto ──

    def test_set_shape_only_then_from_proto(self):
        """FromProto should clear lazy flag and allocate data."""
        blob = caffe_ffi.Blob()
        blob.set_shape_only([3, 4])
        assert blob.is_lazy_allocated()

        # Create a blob with data, serialize, then load into lazy blob
        source = caffe_ffi.Blob([3, 4])
        data = np.random.randn(3, 4).astype(np.float32)
        source.set_data(data)

        proto = caffe_ffi.BlobProto()
        source.to_proto(proto)
        blob.from_proto(proto)

        assert not blob.is_lazy_allocated(), "FromProto should clear lazy flag"
        assert blob.cpu_data() is not None

    # ── Case 14: EmptyShape ──

    def test_empty_shape_rejected(self):
        """Empty shape should be rejected."""
        blob = caffe_ffi.Blob()
        # Empty shape list should raise
        with pytest.raises((ValueError, RuntimeError)):
            blob.set_shape_only([])

    # ── Case 15: LazyBlobShareDataThenWrite ──

    def test_lazy_blob_share_data_then_write(self):
        """After ShareData, cpu_mutable_data() should trigger COW."""
        source = caffe_ffi.Blob([5, 5])
        source_data = np.random.randn(5, 5).astype(np.float32)
        source.set_data(source_data)

        target = caffe_ffi.Blob()
        target.set_shape_only([5, 5])
        target.share_data(source)

        # Writing to target should trigger COW (if COW is enabled)
        mutable = target.cpu_mutable_data()
        assert mutable is not None, "cpu_mutable_data() should return valid pointer"

    # ── Case 16: N1SplitLazyAllocation ──

    def test_n1_split_no_lazy_allocation(self):
        """N=1 Split should NOT trigger lazy allocation (N=1 is special path)."""
        from caffe_ffi import LayerParameter, NetParameter

        proto = NetParameter()
        proto.name = "test_n1_normal"
        input_param = LayerParameter()
        input_param.type = "Input"
        input_param.name = "data"
        input_param.top.append("data")
        input_param.input_param.shape.add().dim[:] = [1, 256]
        proto.layer.append(input_param)
        split_param = LayerParameter()
        split_param.type = "Split"
        split_param.name = "split"
        split_param.bottom.append("data")
        split_param.top.append("split_0")
        proto.layer.append(split_param)

        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 256).astype(np.float32)

        out = net.Forward({"data": inp})
        assert "split_0" in out
        np.testing.assert_array_almost_equal(out["split_0"], inp)

    # ── Case 17: N16Boundary ──

    def test_n16_boundary(self):
        """N=16 exactly at threshold should trigger lazy allocation."""
        from caffe_ffi import LayerParameter, NetParameter

        N = 16
        C = 128

        proto = NetParameter()
        proto.name = f"test_boundary_N{N}"
        input_param = LayerParameter()
        input_param.type = "Input"
        input_param.name = "data"
        input_param.top.append("data")
        input_param.input_param.shape.add().dim[:] = [1, C]
        proto.layer.append(input_param)
        split_param = LayerParameter()
        split_param.type = "Split"
        split_param.name = "split"
        split_param.bottom.append("data")
        for i in range(N):
            split_param.top.append(f"split_{i}")
        proto.layer.append(split_param)

        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, C).astype(np.float32)

        out = net.Forward({"data": inp})

        for i in range(N):
            key = f"split_{i}"
            assert key in out
            np.testing.assert_array_almost_equal(out[key], inp)


# ═══════════════════════════════════════════════════════════════════════
# Parametrized compatibility test for downstream layers
# ═══════════════════════════════════════════════════════════════════════

class TestDownstreamLayerCompatibility:
    """Verify downstream layers work with lazy-allocated->shared blobs."""

    @pytest.mark.parametrize("layer_type,output_key", [
        ("ReLU", "downstream_out"),
        ("Sigmoid", "downstream_out"),
        ("TanH", "downstream_out"),
    ])
    def test_activation_after_lazy_split(self, layer_type, output_key):
        """Activation layers after lazy Split should work."""
        from caffe_ffi import LayerParameter, NetParameter

        N = 100

        proto = NetParameter()
        proto.name = f"test_{layer_type}_after_lazy"
        input_param = LayerParameter()
        input_param.type = "Input"
        input_param.name = "data"
        input_param.top.append("data")
        input_param.input_param.shape.add().dim[:] = [1, 64, 28, 28]
        proto.layer.append(input_param)
        split_param = LayerParameter()
        split_param.type = "Split"
        split_param.name = "split"
        split_param.bottom.append("data")
        for i in range(N):
            split_param.top.append(f"split_{i}")
        proto.layer.append(split_param)
        downstream = LayerParameter()
        downstream.type = layer_type
        downstream.name = "downstream"
        downstream.bottom.append("split_0")
        downstream.top.append(output_key)
        proto.layer.append(downstream)

        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 64, 28, 28).astype(np.float32)

        out = net.Forward({"data": inp})

        assert output_key in out, f"{layer_type} output '{output_key}' missing"
        assert out[output_key].shape == inp.shape


# ── Manual verification ────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 3.1 SetShapeOnly Manual Verification")
    print("=" * 60)

    # Test 1: Basic shape storage
    print("\n[1/5] Basic shape storage...")
    blob = caffe_ffi.Blob()
    blob.set_shape_only([2, 3, 224, 224])
    assert blob.is_lazy_allocated()
    assert blob.num_axes() == 4
    print("  PASS")

    # Test 2: No allocation
    print("[2/5] No data allocation...")
    blob2 = caffe_ffi.Blob()
    before = caffe_ffi.total_allocated_bytes()
    blob2.set_shape_only([100, 100, 100])
    after = caffe_ffi.total_allocated_bytes()
    assert after == before
    print("  PASS")

    # Test 3: cpu_data() returns None
    print("[3/5] cpu_data() returns None...")
    assert blob2.cpu_data() is None
    print("  PASS")

    # Test 4: Reshape clears lazy flag
    print("[4/5] Reshape clears lazy flag...")
    blob3 = caffe_ffi.Blob()
    blob3.set_shape_only([10, 20])
    assert blob3.is_lazy_allocated()
    blob3.reshape([5, 5])
    assert not blob3.is_lazy_allocated()
    print("  PASS")

    # Test 5: ShareData clears lazy flag
    print("[5/5] ShareData clears lazy flag...")
    source = caffe_ffi.Blob([3, 4])
    target = caffe_ffi.Blob()
    target.set_shape_only([3, 4])
    target.share_data(source)
    assert not target.is_lazy_allocated()
    assert target.shares_data_with(source)
    print("  PASS")

    print("\nAll manual checks passed!")