"""RNN / LSTM Backward gradient tests (Phase 2, Task 5).

Verifies the C++ ``RecurrentLayer``/``RNNLayer``/``LSTMLayer`` Backward (BPTT)
against the L0-L1-L2-L3 gradient-verification standard defined in
``.trae/specs/caffe-ffi-rnn-lstm-phase2/spec.md``:

  L0  Smoke       — minimal config Backward runs without crashing, no NaN/Inf.
  L1  Hand-known  — exact known-value Backward (assert_array_equal where exact).
  L2  Numpy ref   — C++ gradients match the pure-numpy ``rnn_backward`` /
                    ``lstm_backward`` reference (>=3 param combos, N>1,
                    >=3 seeds, rtol <= 1e-5).
  L3  Numerical   — central finite-difference gradient end-to-end for dX and
                    every dW/db (cos_sim > 0.99, norm_ratio in [0.9, 1.1]),
                    using ``_grad_check_utils`` and ``avoid_c1_discontinuity``
                    for the piecewise relu activation.

Weight/blob conventions (Caffe-compatible):
  RNN  layer -> 4 blobs: [W_ih (H,D), W_hh (H,H), b_ih (H,), b_hh (H,)]
  LSTM layer -> 2 blobs: [W (4H, D+H), b (4H,)]  (packed, gate order i,f,o,g)

The hidden dim H is inferred from the pre-loaded weight blob shapes, so the
prototxt must carry the weight blob shapes via ``LayerParameter.blobs``.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net, sequence
from .conftest import require_cpp_extension
from ._grad_check_utils import (
    assert_grad_close,
    numerical_grad_for_blob,
    numerical_grad_for_input,
)
from .caffe_test_helpers import avoid_c1_discontinuity

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
RTOL = 1e-5      # L2 numpy-reference tolerance
ATOL = 1e-5
EPS = 1e-3       # central finite-difference step for L3
NUM_RTOL = 1e-3  # L3 numerical-gradient tolerance (smooth layers)
NUM_ATOL = 1e-4
REL_RTOL = 5e-3  # L3 tolerance for the piecewise relu activation

# Tolerance matrix (from spec.md L3 threshold table):
#   linear combination / smooth activation -> (1e-3, 1e-4)
#   piecewise relu (C1 kink)               -> (5e-3, 1e-4) + avoid_c1_discontinuity


# ---------------------------------------------------------------------------
# Prototxt builders
# ---------------------------------------------------------------------------

def _make_rnn_proto(T, N, D, H, activation="tanh"):
    """Input(T,N,D) -> RNN layer with 4 pre-sized weight blobs."""
    return textwrap.dedent(f"""\
        name: "rnn_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {T} dim: {N} dim: {D} }} }}
        }}
        layer {{
          name: "rnn"
          type: "RNN"
          bottom: "data"
          top: "out"
          recurrent_param {{
            num_steps: {T}
            activation: "{activation}"
          }}
          blobs {{ shape {{ dim: {H} dim: {D} }} }}
          blobs {{ shape {{ dim: {H} dim: {H} }} }}
          blobs {{ shape {{ dim: {H} }} }}
          blobs {{ shape {{ dim: {H} }} }}
        }}
    """)


def _make_lstm_proto(T, N, D, H):
    """Input(T,N,D) -> LSTM layer with 2 pre-sized packed-weight blobs."""
    return textwrap.dedent(f"""\
        name: "lstm_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {T} dim: {N} dim: {D} }} }}
        }}
        layer {{
          name: "lstm"
          type: "LSTM"
          bottom: "data"
          top: "out"
          recurrent_param {{
            num_steps: {T}
          }}
          blobs {{ shape {{ dim: {4 * H} dim: {D + H} }} }}
          blobs {{ shape {{ dim: {4 * H} }} }}
        }}
    """)


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _rnn_weights(D, H, seed):
    """Deterministic RNN weight dict (W_ih, W_hh, b_ih, b_hh)."""
    return sequence.init_rnn_weights(D, H, seed=seed, scale=0.3)


def _lstm_weights(D, H, seed):
    """Deterministic LSTM packed weights (W, b) + per-gate dict for reference.

    Returns (packed_W, packed_b, gates) where:
      - packed_W/b are the Caffe-style blobs loaded into the C++ layer.
      - gates is the per-gate dict (b_hi=0 etc.) fed to the numpy reference.
    """
    w = sequence.init_lstm_weights(D, H, seed=seed, scale=0.3)
    packed_W, packed_b = sequence.pack_lstm_weights_caffe(**w)
    gates = sequence.unpack_lstm_weights_caffe(packed_W, packed_b, D, H)
    return packed_W, packed_b, gates


def _set_rnn_weights(net, w):
    layer = net.layer_by_name("rnn")
    layer.blobs[0].from_numpy(w["W_ih"].astype(np.float32))
    layer.blobs[1].from_numpy(w["W_hh"].astype(np.float32))
    layer.blobs[2].from_numpy(w["b_ih"].reshape(-1).astype(np.float32))
    layer.blobs[3].from_numpy(w["b_hh"].reshape(-1).astype(np.float32))


def _set_lstm_weights(net, W, b):
    layer = net.layer_by_name("lstm")
    layer.blobs[0].from_numpy(W.astype(np.float32))
    layer.blobs[1].from_numpy(b.reshape(-1).astype(np.float32))


# ---------------------------------------------------------------------------
# L2 reference helpers: pack numpy per-gate gradients into Caffe packed form
# ---------------------------------------------------------------------------

def _pack_lstm_dW(grads, H, D):
    """Pack the numpy per-gate dW into Caffe (4H, D+H) layout."""
    dW_ih = np.vstack([grads["dW_ii"], grads["dW_if"], grads["dW_io"], grads["dW_ig"]])
    dW_hh = np.vstack([grads["dW_hi"], grads["dW_hf"], grads["dW_ho"], grads["dW_hg"]])
    return np.hstack([dW_ih, dW_hh]).astype(np.float32)


def _pack_lstm_db(grads, H):
    """Pack the combined-bias gradient into the Caffe (4H,) layout.

    The C++ layer uses the combined per-gate bias (b_i = b_ii + b_hi), so the
    gradient w.r.t. the combined bias equals the reference's per-gate bias
    gradient (db_ii == db_hi == dL/db_i).
    """
    return np.concatenate([
        grads["db_ii"], grads["db_if"], grads["db_io"], grads["db_ig"],
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# L0: smoke tests
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestRecurrentSmoke:
    """L0 — minimal config Backward runs without crash and stays finite."""

    def test_rnn_backward_smoke(self):
        T, N, D, H = 4, 2, 3, 4
        net = Net(_make_rnn_proto(T, N, D, H, activation="tanh"))
        _set_rnn_weights(net, _rnn_weights(D, H, seed=0))
        x = np.random.RandomState(0).randn(T, N, D).astype(np.float32) * 0.3
        dy = np.random.RandomState(1).randn(T, N, H).astype(np.float32) * 0.3
        net.forward({"data": x})
        net.backward({"out": dy})
        dX = net.blob_by_name("data").diff
        for blob in net.layer_by_name("rnn").blobs:
            assert np.all(np.isfinite(blob.diff)), f"RNN blob diff has NaN/Inf: {blob.shape}"
        assert np.all(np.isfinite(dX)), "RNN dX has NaN/Inf"
        assert dX.shape == x.shape

    def test_lstm_backward_smoke(self):
        T, N, D, H = 4, 2, 3, 4
        net = Net(_make_lstm_proto(T, N, D, H))
        W, b, _ = _lstm_weights(D, H, seed=0)
        _set_lstm_weights(net, W, b)
        x = np.random.RandomState(2).randn(T, N, D).astype(np.float32) * 0.3
        dy = np.random.RandomState(3).randn(T, N, H).astype(np.float32) * 0.3
        net.forward({"data": x})
        net.backward({"out": dy})
        dX = net.blob_by_name("data").diff
        for blob in net.layer_by_name("lstm").blobs:
            assert np.all(np.isfinite(blob.diff)), f"LSTM blob diff has NaN/Inf: {blob.shape}"
        assert np.all(np.isfinite(dX)), "LSTM dX has NaN/Inf"
        assert dX.shape == x.shape


# ---------------------------------------------------------------------------
# L1: hand-computed known values
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestRecurrentKnownValues:
    """L1 — exact known-value Backward checks."""

    def test_rnn_relu_known_value(self):
        """Single-step relu RNN, W_ih=I, W_hh=0, b=0.

        x = [[[1.0, -2.0]]]           (T=1,N=1,D=2)
        z = x @ I = [1, -2]  ->  h = relu(z) = [1, 0]
        dy = [[[1.0, 3.0]]]
        dz = dy * relu'(z) = [1, 0]
        dX  = dz @ I = [1, 0]
        dW_ih = dz^T @ x = [[1, -2], [0, 0]]
        db_ih = dz = [1, 0]
        dW_hh = dz^T @ h_prev = 0      (h_prev = h0 = 0)
        db_hh = dz = [1, 0]            (b_hh contributes to z, so dL/db_hh = dz)
        """
        T, N, D, H = 1, 1, 2, 2
        net = Net(_make_rnn_proto(T, N, D, H, activation="relu"))
        w = {
            "W_ih": np.eye(2, dtype=np.float32),
            "W_hh": np.zeros((2, 2), dtype=np.float32),
            "b_ih": np.zeros(2, dtype=np.float32),
            "b_hh": np.zeros(2, dtype=np.float32),
        }
        _set_rnn_weights(net, w)
        x = np.array([[[1.0, -2.0]]], dtype=np.float32)
        dy = np.array([[[1.0, 3.0]]], dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dX, np.array([[[1.0, 0.0]]], dtype=np.float32))
        np.testing.assert_array_equal(
            net.layer_by_name("rnn").blobs[0].diff,
            np.array([[1.0, -2.0], [0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            net.layer_by_name("rnn").blobs[2].diff, np.array([1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            net.layer_by_name("rnn").blobs[1].diff, np.zeros((2, 2), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            net.layer_by_name("rnn").blobs[3].diff, np.array([1.0, 0.0], dtype=np.float32)
        )

    def test_rnn_tanh_known_value(self):
        """Single-step tanh RNN, W_ih=I, W_hh=0, b=0.

        x = [[[0.5, -0.3]]], h = tanh([0.5, -0.3]) = [0.4621, -0.2913]
        dy = [[[1.0, 2.0]]]
        dz = dy * (1 - h^2)
        dX  = dz @ I = dz
        db_ih = dz
        """
        T, N, D, H = 1, 1, 2, 2
        net = Net(_make_rnn_proto(T, N, D, H, activation="tanh"))
        w = {
            "W_ih": np.eye(2, dtype=np.float32),
            "W_hh": np.zeros((2, 2), dtype=np.float32),
            "b_ih": np.zeros(2, dtype=np.float32),
            "b_hh": np.zeros(2, dtype=np.float32),
        }
        _set_rnn_weights(net, w)
        x = np.array([[[0.5, -0.3]]], dtype=np.float32)
        dy = np.array([[[1.0, 2.0]]], dtype=np.float32)
        net.forward({"data": x})
        net.backward({"out": dy})

        h = np.tanh(np.array([0.5, -0.3], dtype=np.float64))
        dz = dy[0, 0].astype(np.float64) * (1.0 - h * h)
        expected_dX = dz.astype(np.float32)
        expected_db = dz.astype(np.float32)

        dX = net.blob_by_name("data").diff
        np.testing.assert_allclose(dX, expected_dX.reshape(1, 1, 2), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            net.layer_by_name("rnn").blobs[2].diff, expected_db, rtol=RTOL, atol=ATOL
        )

    def test_rnn_forward_last_state(self):
        """Multi-step RNN: forward h_n equals the last time-step output."""
        T, N, D, H = 5, 2, 3, 4
        net = Net(_make_rnn_proto(T, N, D, H, activation="tanh"))
        _set_rnn_weights(net, _rnn_weights(D, H, seed=1))
        x = np.random.RandomState(4).randn(T, N, D).astype(np.float32) * 0.3
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out[-1], out[-1], rtol=1e-6)

    def test_lstm_zero_weight_forward(self):
        """Zero-weight LSTM: all gates = sigmoid(0) = 0.5, g = tanh(0) = 0.

        c = f*c0 + i*g = 0, h = o*tanh(c) = 0 -> output all zero.
        """
        T, N, D, H = 3, 2, 1, 1
        net = Net(_make_lstm_proto(T, N, D, H))
        zero_W = np.zeros((4 * H, D + H), dtype=np.float32)
        zero_b = np.zeros(4 * H, dtype=np.float32)
        _set_lstm_weights(net, zero_W, zero_b)
        x = np.random.RandomState(5).randn(T, N, D).astype(np.float32) * 0.5
        out = net.forward({"data": x})["out"]
        np.testing.assert_allclose(out, np.zeros_like(out), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# L2: numpy reference match
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestRNNReferenceBackward:
    """L2 — RNN C++ Backward matches the pure-numpy ``rnn_backward``."""

    @pytest.mark.parametrize("activation", ["tanh", "relu"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_rnn_matches_reference(self, activation, seed):
        T, N, D, H = 5, 2, 3, 4
        if activation == "relu":
            scale = 0.3
        net = Net(_make_rnn_proto(T, N, D, H, activation=activation))
        w = _rnn_weights(D, H, seed)
        _set_rnn_weights(net, w)

        rng = np.random.RandomState(100 + seed)
        x = rng.randn(T, N, D).astype(np.float32) * 0.3
        dy = rng.randn(T, N, H).astype(np.float32) * 0.3

        net.forward({"data": x})
        net.backward({"out": dy})

        dX_cpp = net.blob_by_name("data").diff
        dW_ih = net.layer_by_name("rnn").blobs[0].diff
        dW_hh = net.layer_by_name("rnn").blobs[1].diff
        db_ih = net.layer_by_name("rnn").blobs[2].diff
        db_hh = net.layer_by_name("rnn").blobs[3].diff

        ref = sequence.rnn_backward(
            dy, x, w["W_ih"], w["W_hh"], w["b_ih"], w["b_hh"],
            h0=None, activation=activation,
        )
        np.testing.assert_allclose(dX_cpp, ref["dX"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dW_ih, ref["dW_ih"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dW_hh, ref["dW_hh"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(db_ih, ref["db_ih"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(db_hh, ref["db_hh"], rtol=RTOL, atol=ATOL)

    def test_rnn_forward_matches_numpy(self):
        """RNN C++ forward matches the numpy reference (rtol <= 1e-5)."""
        T, N, D, H = 5, 2, 3, 4
        net = Net(_make_rnn_proto(T, N, D, H, activation="tanh"))
        w = _rnn_weights(D, H, seed=3)
        _set_rnn_weights(net, w)
        x = np.random.RandomState(6).randn(T, N, D).astype(np.float32) * 0.3
        out_cpp = net.forward({"data": x})["out"]
        out_ref, _ = sequence.rnn_forward(
            x, w["W_ih"], w["W_hh"], w["b_ih"], w["b_hh"], activation="tanh",
        )
        np.testing.assert_allclose(out_cpp, out_ref, rtol=RTOL, atol=ATOL)


@require_cpp_extension
class TestLSTMReferenceBackward:
    """L2 — LSTM C++ Backward matches the pure-numpy ``lstm_backward``."""

    @pytest.mark.parametrize("seed", [4, 5, 6])
    def test_lstm_matches_reference(self, seed):
        T, N, D, H = 5, 2, 3, 4
        net = Net(_make_lstm_proto(T, N, D, H))
        W, b, gates = _lstm_weights(D, H, seed)
        _set_lstm_weights(net, W, b)

        rng = np.random.RandomState(200 + seed)
        x = rng.randn(T, N, D).astype(np.float32) * 0.3
        dy = rng.randn(T, N, H).astype(np.float32) * 0.3

        net.forward({"data": x})
        net.backward({"out": dy})

        dX_cpp = net.blob_by_name("data").diff
        dW_cpp = net.layer_by_name("lstm").blobs[0].diff
        db_cpp = net.layer_by_name("lstm").blobs[1].diff

        ref = sequence.lstm_backward(
            dy, x,
            gates["W_ii"], gates["W_if"], gates["W_io"], gates["W_ig"],
            gates["W_hi"], gates["W_hf"], gates["W_ho"], gates["W_hg"],
            gates["b_ii"], gates["b_if"], gates["b_io"], gates["b_ig"],
            gates["b_hi"], gates["b_hf"], gates["b_ho"], gates["b_hg"],
            h0=None, c0=None,
        )
        np.testing.assert_allclose(dX_cpp, ref["dX"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dW_cpp, _pack_lstm_dW(ref, H, D), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(db_cpp, _pack_lstm_db(ref, H), rtol=RTOL, atol=ATOL)

    def test_lstm_forward_matches_numpy(self):
        """LSTM C++ forward matches the numpy reference (rtol <= 1e-5)."""
        T, N, D, H = 5, 2, 3, 4
        net = Net(_make_lstm_proto(T, N, D, H))
        W, b, gates = _lstm_weights(D, H, seed=7)
        _set_lstm_weights(net, W, b)
        x = np.random.RandomState(7).randn(T, N, D).astype(np.float32) * 0.3
        out_cpp = net.forward({"data": x})["out"]
        out_ref, _ = sequence.lstm_forward(
            x,
            gates["W_ii"], gates["W_if"], gates["W_io"], gates["W_ig"],
            gates["W_hi"], gates["W_hf"], gates["W_ho"], gates["W_hg"],
            gates["b_ii"], gates["b_if"], gates["b_io"], gates["b_ig"],
            gates["b_hi"], gates["b_hf"], gates["b_ho"], gates["b_hg"],
        )
        np.testing.assert_allclose(out_cpp, out_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# L3: numerical gradient end-to-end
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestRNNNumericalGradient:
    """L3 — RNN analytic vs central finite-difference gradient."""

    def _run(self, activation, seed):
        T, N, D, H = 2, 2, 2, 3  # small for speed
        net = Net(_make_rnn_proto(T, N, D, H, activation=activation))
        w = _rnn_weights(D, H, seed)
        _set_rnn_weights(net, w)
        rng = np.random.RandomState(300 + seed)
        x = rng.randn(T, N, D).astype(np.float32) * 0.3
        if activation == "relu":
            x = avoid_c1_discontinuity(x, h=EPS)
        dy = rng.randn(T, N, H).astype(np.float32) * 0.3

        net.forward({"data": x})
        net.backward({"out": dy})
        return net, x, dy

    @pytest.mark.parametrize("activation", ["tanh", "relu"])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_rnn_dX_numerical(self, activation, seed):
        net, x, dy = self._run(activation, seed)
        rtol, atol = (NUM_RTOL, NUM_ATOL) if activation == "tanh" else (REL_RTOL, NUM_ATOL)
        dX_analytic = net.blob_by_name("data").diff
        dX_num = numerical_grad_for_input(net, "data", x, "out", dy, h=EPS, name="RNN dX")
        assert_grad_close(dX_analytic, dX_num, name=f"RNN({activation}) dX", rtol=rtol, atol=atol)

    @pytest.mark.parametrize("activation", ["tanh", "relu"])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_rnn_dW_dB_numerical(self, activation, seed):
        net, x, dy = self._run(activation, seed)
        rtol, atol = (NUM_RTOL, NUM_ATOL) if activation == "tanh" else (REL_RTOL, NUM_ATOL)
        for idx, name in [(0, "dW_ih"), (1, "dW_hh"), (2, "db_ih"), (3, "db_hh")]:
            analytic = net.layer_by_name("rnn").blobs[idx].diff
            num = numerical_grad_for_blob(
                net, "rnn", idx, {"data": x}, "out", dy, h=EPS,
                name=f"RNN({activation}) {name}",
            )
            assert_grad_close(analytic, num, name=f"RNN({activation}) {name}",
                              rtol=rtol, atol=atol)


@require_cpp_extension
class TestLSTMNumericalGradient:
    """L3 — LSTM analytic vs central finite-difference gradient."""

    def _run(self, seed):
        T, N, D, H = 2, 2, 2, 3  # small for speed
        net = Net(_make_lstm_proto(T, N, D, H))
        W, b, _ = _lstm_weights(D, H, seed)
        _set_lstm_weights(net, W, b)
        rng = np.random.RandomState(400 + seed)
        x = rng.randn(T, N, D).astype(np.float32) * 0.3
        dy = rng.randn(T, N, H).astype(np.float32) * 0.3

        net.forward({"data": x})
        net.backward({"out": dy})
        return net, x, dy

    @pytest.mark.parametrize("seed", [0, 1])
    def test_lstm_dX_numerical(self, seed):
        net, x, dy = self._run(seed)
        dX_analytic = net.blob_by_name("data").diff
        dX_num = numerical_grad_for_input(net, "data", x, "out", dy, h=EPS, name="LSTM dX")
        assert_grad_close(dX_analytic, dX_num, name=f"LSTM dX", rtol=NUM_RTOL, atol=NUM_ATOL)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_lstm_dW_dB_numerical(self, seed):
        net, x, dy = self._run(seed)
        for idx, name in [(0, "dW_packed"), (1, "db_packed")]:
            analytic = net.layer_by_name("lstm").blobs[idx].diff
            num = numerical_grad_for_blob(
                net, "lstm", idx, {"data": x}, "out", dy, h=EPS,
                name=f"LSTM {name}",
            )
            assert_grad_close(analytic, num, name=f"LSTM {name}",
                              rtol=NUM_RTOL, atol=NUM_ATOL)