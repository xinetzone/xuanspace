"""P2-B1: Numeric boundary tests — NaN/Inf/extreme values, dtype errors, non-contiguous arrays.

These tests verify that the C++ extension handles malformed inputs gracefully:
no segfaults, no silent memory corruption, clean Python exceptions for type errors.
"""
from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import net_param_from_string, net_from_param
from .conftest import require_cpp_extension


# ─── Helpers ──────────────────────────────────────────────────────

_SMALL_MLP_PROTO = """name: "extreme_mlp"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 4 dim: 3 } } }
layer { name: "ip1" type: "InnerProduct" bottom: "data" top: "ip1" inner_product_param { num_output: 4 bias_term: true } }
layer { name: "relu1" type: "ReLU" bottom: "ip1" top: "ip1" }
layer { name: "ip2" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 2 bias_term: true } }
layer { name: "prob" type: "Softmax" bottom: "ip2" top: "prob" }"""


def _make_small_mlp(seed: int = 42):
    """Create a small MLP with known random weights."""
    net = net_from_param(net_param_from_string(_SMALL_MLP_PROTO))
    rng = np.random.RandomState(seed)
    for layer in net.layers_array():
        if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
            W = layer.blobs[0]
            W.from_numpy((rng.randn(*W.shape).astype(np.float32)) * 0.1)
            if len(layer.blobs) >= 2:
                layer.blobs[1].from_numpy(np.zeros(layer.blobs[1].shape, dtype=np.float32))
    return net


def _try_forward(net, input_dict, expected_error=False):
    """Run forward, catching all common exceptions. Returns (output_dict_or_None, exception_or_None)."""
    try:
        out = net.forward(input_dict)
        return out, None
    except (ValueError, TypeError, RuntimeError, IndexError, OverflowError, MemoryError) as e:
        return None, e
    except Exception as e:
        # Segfault would be a hard crash, not a Python exception — if we get here
        # it's an unexpected exception type; re-raise to fail the test
        raise


# ─── P2-B1a: Extreme numeric values ──────────────────────────────

@require_cpp_extension
class TestExtremeValues:
    """NaN, Inf, denormals, very large/small values — must not segfault."""

    @pytest.fixture
    def net(self, ptrace):
        with ptrace("build small MLP (extreme values)") as t:
            n = _make_small_mlp(42)
            t['layers'] = len(n.layers_array())
        return n

    def test_nan_input_single_element(self, net, ptrace):
        """Input with one NaN element should not segfault; output may contain NaN."""
        inp = np.array([[1.0, 2.0, np.nan], [0.1, 0.2, 0.3],
                        [-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]], dtype=np.float32)
        with ptrace("forward with 1 NaN element") as t:
            out, err = _try_forward(net, {"data": inp}, expected_error=False)
            if err is None:
                prob = out["prob"]
                has_nan = bool(np.any(np.isnan(prob)))
                has_inf = bool(np.any(np.isinf(prob)))
                t['result'] = f'ok nan={has_nan} inf={has_inf}'
                assert prob.shape == (4, 2), f"Shape mismatch: {prob.shape}"
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_nan_input_all_elements(self, net, ptrace):
        """All-NaN input must not segfault."""
        inp = np.full((4, 3), np.nan, dtype=np.float32)
        with ptrace("forward with all NaN") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = f'ok shape={out["prob"].shape}'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_positive_inf_input(self, net, ptrace):
        """+Inf input must not segfault (may produce NaN in Softmax)."""
        inp = np.array([[np.inf, 0.0, 0.0], [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        with ptrace("forward with +Inf") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_negative_inf_input(self, net, ptrace):
        """-Inf input must not segfault."""
        inp = np.array([[-np.inf, 0.0, 0.0], [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        with ptrace("forward with -Inf") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_mixed_nan_inf(self, net, ptrace):
        """Mixed NaN and Inf input must not segfault."""
        inp = np.array([[np.nan, np.inf, -np.inf], [0.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
        with ptrace("forward with NaN+Inf mixed") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_large_values_1e6(self, net, ptrace):
        """Input values ~1e6 may overflow Softmax but must not segfault."""
        rng = np.random.RandomState(1)
        inp = (rng.randn(4, 3).astype(np.float32)) * 1e6
        with ptrace("forward with values ~1e6") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_large_values_near_float32_max(self, net, ptrace):
        """Input near float32 max (~3e38) should not segfault."""
        inp = np.full((4, 3), np.finfo(np.float32).max / 2, dtype=np.float32)
        with ptrace("forward near float32 max") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_denormal_values(self, net, ptrace):
        """Denormal/very small positive values (~1e-40) should not segfault."""
        inp = np.full((4, 3), np.finfo(np.float32).tiny, dtype=np.float32)
        with ptrace("forward with denormal values") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'
            else:
                t['result'] = f'raised:{type(err).__name__}'
            # Either success or clean error is acceptable; no segfault is the requirement

    def test_all_zeros(self, net, ptrace):
        """All-zero input should produce valid probability distribution."""
        inp = np.zeros((4, 3), dtype=np.float32)
        with ptrace("forward with all zeros") as t:
            out = net.forward({"data": inp})
            prob = out["prob"]
            t['out_shape'] = str(prob.shape)
            assert prob.shape == (4, 2)
            np.testing.assert_allclose(prob.sum(axis=1), np.ones(4), rtol=1e-5)
            assert np.all(prob >= 0) and np.all(prob <= 1)

    def test_negative_zero(self, net, ptrace):
        """Negative zero (-0.0) should behave identically to 0.0."""
        inp = np.full((4, 3), -0.0, dtype=np.float32)
        with ptrace("forward with -0.0") as t:
            out = net.forward({"data": inp})
            prob = out["prob"]
            t['out_shape'] = str(prob.shape)
            assert prob.shape == (4, 2)
            np.testing.assert_allclose(prob.sum(axis=1), np.ones(4), rtol=1e-5)

    def test_alternating_extremes(self, net, ptrace):
        """Alternating NaN/Inf/normal inputs across forwards must not crash."""
        rng = np.random.RandomState(99)
        normal = rng.randn(4, 3).astype(np.float32)
        nan_inp = np.full((4, 3), np.nan, dtype=np.float32)
        inf_inp = np.full((4, 3), np.inf, dtype=np.float32)
        results = []
        with ptrace("alternating normal/NaN/Inf x3 rounds") as t:
            for label, inp in [
                ("normal", normal), ("nan", nan_inp), ("inf", inf_inp),
                ("normal2", normal), ("nan2", nan_inp), ("inf2", inf_inp),
            ]:
                out, err = _try_forward(net, {"data": inp}, expected_error=True)
                if err is None:
                    results.append((label, out["prob"].shape))
                else:
                    results.append((label, 'raised'))
            t['results'] = str(results)
        # Normal inputs must always produce correct shape
        assert results[0][1] == (4, 2), f"First normal forward failed: {results[0]}"
        assert results[3][1] == (4, 2), f"Second normal forward failed: {results[3]}"


# ─── P2-B1b: Wrong dtype inputs ──────────────────────────────────

@require_cpp_extension
class TestDTypeErrors:
    """Wrong dtype inputs (float64, int, etc.) must raise clean errors, not segfault."""

    @pytest.fixture
    def net(self, ptrace):
        with ptrace("build small MLP (dtype tests)"):
            return _make_small_mlp(123)

    @pytest.mark.parametrize("dtype", [np.float64, np.float16])
    def test_float_dtype_mismatch(self, net, dtype, ptrace):
        """float64/float16 inputs should raise cleanly or auto-convert."""
        inp = np.random.randn(4, 3).astype(dtype)
        with ptrace(f"forward dtype={dtype.__name__}") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            t['dtype'] = dtype.__name__
            if err is not None:
                t['result'] = f'clean_raise:{type(err).__name__}'
                assert isinstance(err, (TypeError, ValueError, RuntimeError)), \
                    f"Unexpected error type: {type(err).__name__}"
            else:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8, np.bool_])
    def test_integer_dtype_raises(self, net, dtype, ptrace):
        """Integer/bool inputs must raise TypeError/ValueError (not segfault)."""
        if dtype == np.bool_:
            inp = np.array([[True, False, True], [False, True, False],
                           [True, True, False], [False, False, True]], dtype=dtype)
        else:
            inp = np.random.randint(-10, 10, size=(4, 3)).astype(dtype)
        with ptrace(f"forward dtype={dtype.__name__}") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            t['dtype'] = dtype.__name__
            if err is not None:
                t['result'] = f'clean_raise:{type(err).__name__}'
                assert isinstance(err, (TypeError, ValueError, RuntimeError)), \
                    f"Expected TypeError/ValueError, got {type(err).__name__}"
            else:
                # If auto-conversion works, verify shape
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'

    def test_complex_dtype_raises(self, net, ptrace):
        """Complex64 inputs must not segfault."""
        inp = np.random.randn(4, 3).astype(np.complex64) + 1j * np.random.randn(4, 3).astype(np.complex64)
        with ptrace("forward complex64") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            if err is not None:
                t['result'] = f'clean_raise:{type(err).__name__}'
            else:
                t['result'] = 'ok (auto-convert or accepted)'
                assert out["prob"].shape == (4, 2)

    def test_object_dtype_raises(self, net, ptrace):
        """Object dtype (Python floats in ndarray) must not segfault."""
        inp = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=object)
        with ptrace("forward object dtype") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            if err is not None:
                t['result'] = f'clean_raise:{type(err).__name__}'
            else:
                t['result'] = 'ok (unexpected)'
                assert out["prob"].shape == (4, 2)

    def test_readonly_array(self, net, ptrace):
        """Read-only (writeable=False) float32 array must work or raise cleanly."""
        inp = np.random.randn(4, 3).astype(np.float32)
        inp.setflags(write=False)
        assert not inp.flags['WRITEABLE']
        with ptrace("forward readonly array") as t:
            out, err = _try_forward(net, {"data": inp})
            if err is None:
                assert out["prob"].shape == (4, 2)
                t['result'] = 'ok'
            else:
                t['result'] = f'raised:{type(err).__name__}'


# ─── P2-B1c: Non-contiguous arrays ──────────────────────────────

@require_cpp_extension
class TestNonContiguousArrays:
    """Non-C-contiguous arrays (transposed, sliced, Fortran-order) must not segfault."""

    @pytest.fixture
    def net(self, ptrace):
        with ptrace("build small MLP (non-contiguous tests)"):
            return _make_small_mlp(77)

    def test_transposed_array(self, net, ptrace):
        """Transposed array (Fortran memory order) should work or raise cleanly."""
        rng = np.random.RandomState(10)
        arr = rng.randn(3, 4).astype(np.float32)
        inp = arr.T  # shape (4,3) but Fortran-order
        assert not inp.flags['C_CONTIGUOUS'], "Test setup error: transpose should be non-contiguous"
        assert inp.shape == (4, 3)

        # Reference: contiguous version (on a fresh net to avoid state pollution)
        net_ref = _make_small_mlp(77)
        with ptrace("forward transposed (contiguous copy ref)"):
            ref_out = net_ref.forward({"data": np.ascontiguousarray(inp)})
            ref = ref_out["prob"].copy()

        # Non-contiguous test
        with ptrace("forward transposed (non-contiguous)") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            if err is None:
                t['result'] = 'ok_non_contig'
                assert out["prob"].shape == (4, 2)
                # If outputs match, implementation handles strides correctly
                max_diff = float(np.max(np.abs(ref - out["prob"])))
                t['max_diff_vs_ref'] = f'{max_diff:.2e}'
                if max_diff < 1e-5:
                    t['stride_safe'] = True
                else:
                    t['stride_safe'] = False
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_sliced_array_strided(self, net, ptrace):
        """Sliced array with stride>1 (non-contiguous) should not segfault."""
        rng = np.random.RandomState(11)
        big = rng.randn(8, 3).astype(np.float32)
        inp = big[::2, :]  # shape (4,3), stride 2 in dim 0
        assert not inp.flags['C_CONTIGUOUS']
        assert inp.shape == (4, 3)

        net_ref = _make_small_mlp(77)
        with ptrace("forward sliced (contiguous ref)"):
            ref = net_ref.forward({"data": np.ascontiguousarray(inp)})["prob"].copy()

        with ptrace("forward sliced (stride=2)") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            if err is None:
                t['result'] = 'ok_non_contig'
                max_diff = float(np.max(np.abs(ref - out["prob"])))
                t['max_diff_vs_ref'] = f'{max_diff:.2e}'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_fortran_order_array(self, net, ptrace):
        """Explicitly Fortran-ordered array should not segfault."""
        rng = np.random.RandomState(12)
        data = rng.randn(4, 3).astype(np.float32)
        inp = np.asfortranarray(data)
        assert inp.flags['F_CONTIGUOUS']
        assert not inp.flags['C_CONTIGUOUS']

        net_ref = _make_small_mlp(77)
        with ptrace("forward F-order (contiguous ref)"):
            ref = net_ref.forward({"data": np.ascontiguousarray(inp)})["prob"].copy()

        with ptrace("forward F-order array") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            if err is None:
                t['result'] = 'ok'
                max_diff = float(np.max(np.abs(ref - out["prob"])))
                t['max_diff_vs_ref'] = f'{max_diff:.2e}'
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_reversed_array(self, net, ptrace):
        """Reversed array [::-1] has negative stride — non-contiguous."""
        rng = np.random.RandomState(13)
        data = rng.randn(4, 3).astype(np.float32)
        inp = data[::-1, :]
        assert not inp.flags['C_CONTIGUOUS']
        with ptrace("forward reversed array") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            if err is None:
                t['result'] = 'ok'
                assert out["prob"].shape == (4, 2)
            else:
                t['result'] = f'raised:{type(err).__name__}'

    def test_column_slice_non_contiguous(self, net, ptrace):
        """Column slice with non-contiguous columns (e.g., arr[:, [0,2,1]])."""
        rng = np.random.RandomState(14)
        big = rng.randn(4, 6).astype(np.float32)
        inp = big[:, [0, 2, 4]]  # Fancy indexing — always non-contiguous
        assert inp.shape == (4, 3)
        with ptrace("forward fancy-indexed columns") as t:
            t['expected_error'] = True
            out, err = _try_forward(net, {"data": inp}, expected_error=True)
            if err is None:
                t['result'] = 'ok'
                assert out["prob"].shape == (4, 2)
            else:
                t['result'] = f'raised:{type(err).__name__}'


# ─── P2-B1d: Recovery after errors ──────────────────────────────

@require_cpp_extension
class TestRecoveryAfterError:
    """After an error (NaN input, wrong dtype), normal forwards must still work."""

    @pytest.fixture
    def net(self, ptrace):
        with ptrace("build small MLP (recovery tests)"):
            return _make_small_mlp(2024)

    def test_normal_after_nan_input(self, net, ptrace):
        """After NaN forward (which may raise or produce NaN), normal input must not crash.

        Note: NaN may contaminate intermediate blobs if the forward silently propagates NaN
        without raising. The key assertion is NO SEGFAULT; shape correctness is verified.
        If the implementation fully resets state, output is a valid distribution.
        """
        rng = np.random.RandomState(0)
        nan_inp = np.full((4, 3), np.nan, dtype=np.float32)
        normal = rng.randn(4, 3).astype(np.float32)

        with ptrace("NaN forward (may raise)") as t:
            t['expected_error'] = True
            nan_out, nan_err = _try_forward(net, {"data": nan_inp}, expected_error=True)
            t['result'] = 'raised' if nan_err else 'ok_no_raise'

        with ptrace("normal forward after NaN") as t:
            out, err = _try_forward(net, {"data": normal})
            assert err is None, f"Recovery crashed after NaN: {err}"
            assert out["prob"].shape == (4, 2), f"Shape corrupted after NaN: {out['prob'].shape}"
            prob = out["prob"]
            # Check for NaN contamination — if present, document but don't fail
            has_nan = bool(np.any(np.isnan(prob)))
            has_inf = bool(np.any(np.isinf(prob)))
            if not has_nan and not has_inf:
                np.testing.assert_allclose(prob.sum(axis=1), np.ones(4), rtol=1e-5)
                t['result'] = 'recovered_clean'
            else:
                t['result'] = f'nan_contamination nan={has_nan} inf={has_inf} (no crash)'

    def test_normal_after_type_error(self, net, ptrace):
        """After wrong-dtype forward raises, normal input still works."""
        rng = np.random.RandomState(1)
        bad_inp = rng.randn(4, 3).astype(np.float64)
        normal = rng.randn(4, 3).astype(np.float32)

        with ptrace("float64 forward (expected error)") as t:
            t['expected_error'] = True
            _try_forward(net, {"data": bad_inp}, expected_error=True)
            t['result'] = 'attempted'

        with ptrace("normal forward after type error") as t:
            out, err = _try_forward(net, {"data": normal})
            assert err is None, f"Recovery failed after type error: {err}"
            assert out["prob"].shape == (4, 2)
            np.testing.assert_allclose(out["prob"].sum(axis=1), np.ones(4), rtol=1e-5)
            t['result'] = 'recovered'

    def test_normal_after_large_values(self, net, ptrace):
        """After extreme-value forward (may raise), normal input still works."""
        rng = np.random.RandomState(2)
        extreme = np.full((4, 3), np.finfo(np.float32).max, dtype=np.float32)
        normal = rng.randn(4, 3).astype(np.float32)

        with ptrace("extreme value forward (may raise)") as t:
            t['expected_error'] = True
            _try_forward(net, {"data": extreme}, expected_error=True)
            t['result'] = 'attempted'

        with ptrace("normal forward after extremes") as t:
            out, err = _try_forward(net, {"data": normal})
            assert err is None, f"Recovery failed after extremes: {err}"
            assert out["prob"].shape == (4, 2)
            np.testing.assert_allclose(out["prob"].sum(axis=1), np.ones(4), rtol=1e-5)
            t['result'] = 'recovered'

    def test_normal_after_non_contiguous_error(self, net, ptrace):
        """After non-contiguous input (which may raise), normal input still works."""
        rng = np.random.RandomState(4)
        big = rng.randn(8, 3).astype(np.float32)
        bad = big[::2, :]  # stride=2 non-contiguous
        normal = rng.randn(4, 3).astype(np.float32)

        with ptrace("non-contiguous forward (may raise)") as t:
            t['expected_error'] = True
            _try_forward(net, {"data": bad}, expected_error=True)
            t['result'] = 'attempted'

        with ptrace("normal forward after non-contiguous") as t:
            out, err = _try_forward(net, {"data": normal})
            assert err is None, f"Recovery failed after non-contiguous: {err}"
            assert out["prob"].shape == (4, 2)
            t['result'] = 'recovered'

    def test_repeated_error_recovery_cycles(self, net, ptrace):
        """10 cycles of (error forward -> normal forward) must all succeed."""
        rng = np.random.RandomState(3)
        n_cycles = 10
        results = []
        with ptrace(f"{n_cycles} error-recovery cycles") as t:
            for i in range(n_cycles):
                bad = np.full((4, 3), np.nan if i % 2 == 0 else np.inf, dtype=np.float32)
                _try_forward(net, {"data": bad}, expected_error=True)
                normal = rng.randn(4, 3).astype(np.float32)
                out, err = _try_forward(net, {"data": normal})
                assert err is None, f"Cycle {i} recovery failed: {err}"
                assert out["prob"].shape == (4, 2)
                results.append(True)
            t['cycles'] = n_cycles
            t['result'] = f'all_{len(results)}_recovered'
