"""Tests for the pure-Python ``caffe_ffi.sequence`` RNN/LSTM forward API.

Covers:
  1. API surface / import checks (``RNN``, ``LSTM``, re-exported helpers)
  2. Hand-computed known-value tests (RNN tanh, RNN relu, LSTM zero-weight)
  3. Output / final-state shape tests (time-first, batch-first, bidirectional)
  4. Layout consistency between ``batch_first=False`` and ``True``
  5. Caffe-style packed weight loading consistency
  6. Final-state correctness (h_n equals last time-step states)
  7. numpy self-consistency (class API vs direct ``rnn_forward``/``lstm_forward``)

These tests are pure numpy and do NOT require the C++ extension. Importing
``caffe_ffi`` may optionally load the native extension (or fall back to
Python-only mode); the ``sequence`` subpackage is pure Python either way.
"""
from __future__ import annotations

import numpy as np
import pytest

from caffe_ffi import sequence

# ---------------------------------------------------------------------------
# Tolerance: reference computes in float64, casts to float32.
# ---------------------------------------------------------------------------
RTOL = 1e-5
ATOL = 1e-5


# ---------------------------------------------------------------------------
# 1. API surface / import
# ---------------------------------------------------------------------------

def test_api_surface():
    """``from caffe_ffi import sequence`` exposes RNN/LSTM and all helpers."""
    assert callable(sequence.RNN)
    assert callable(sequence.LSTM)
    for name in (
        "rnn_forward",
        "lstm_forward",
        "pack_lstm_weights_caffe",
        "unpack_lstm_weights_caffe",
        "init_rnn_weights",
        "init_lstm_weights",
    ):
        assert callable(getattr(sequence, name)), f"missing re-export: {name}"


# ---------------------------------------------------------------------------
# 2. Known-value tests (hand-computed)
# ---------------------------------------------------------------------------

def test_rnn_known_value():
    """RNN single-step tanh: h = tanh(W_ih @ x) with identity W_ih, zero bias.

    x = [[[0.5, -0.3]]], W_ih = eye(2), W_hh = 0, b = 0
    → h = tanh([0.5, -0.3]) = [0.4621..., -0.2913...]
    """
    rnn = sequence.RNN(2, 2)
    rnn.load_weights({
        "W_ih": np.eye(2, dtype=np.float32),
        "W_hh": np.zeros((2, 2), dtype=np.float32),
        "b_ih": np.zeros(2, dtype=np.float32),
        "b_hh": np.zeros(2, dtype=np.float32),
    })
    x = np.array([[[0.5, -0.3]]], dtype=np.float32)  # (1, 1, 2)
    out, h_n = rnn.forward(x)
    expected = np.tanh(np.array([0.5, -0.3], dtype=np.float64)).astype(np.float32)

    assert out.shape == (1, 1, 2)
    np.testing.assert_allclose(out[0, 0], expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(h_n[0], expected, rtol=RTOL, atol=ATOL)


def test_rnn_relu_activation():
    """RNN relu activation: h_t = relu(W_ih @ x_t) with identity W_ih.

    x = [[[1.0, -2.0]]] → relu([1.0, -2.0]) = [1.0, 0.0].
    """
    rnn = sequence.RNN(2, 2, activation="relu")
    rnn.load_weights({
        "W_ih": np.eye(2, dtype=np.float32),
        "W_hh": np.zeros((2, 2), dtype=np.float32),
        "b_ih": np.zeros(2, dtype=np.float32),
        "b_hh": np.zeros(2, dtype=np.float32),
    })
    x = np.array([[[1.0, -2.0]]], dtype=np.float32)  # (1, 1, 2)
    out, h_n = rnn.forward(x)
    expected = np.array([1.0, 0.0], dtype=np.float32)

    np.testing.assert_allclose(out[0, 0], expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(h_n[0], expected, rtol=RTOL, atol=ATOL)


def test_lstm_known_value():
    """LSTM zero-weight known value: all gates = sigmoid(0) = 0.5, g = tanh(0) = 0.

    c_1 = f*c0 + i*g = 0; h_1 = o*tanh(c_1) = 0 → h_n = c_n = 0.
    """
    lstm = sequence.LSTM(1, 1)
    H, D = 1, 1
    weights = {g: np.zeros((H, D), dtype=np.float32) for g in ("W_ii", "W_if", "W_io", "W_ig")}
    weights.update({g: np.zeros((H, H), dtype=np.float32) for g in ("W_hi", "W_hf", "W_ho", "W_hg")})
    weights.update({g: np.zeros(H, dtype=np.float32) for g in (
        "b_ii", "b_if", "b_io", "b_ig", "b_hi", "b_hf", "b_ho", "b_hg")})
    lstm.load_weights(weights)

    x = np.array([[[0.5]]], dtype=np.float32)  # (1, 1, 1)
    out, (h_n, c_n) = lstm.forward(x)

    assert out.shape == (1, 1, 1)
    np.testing.assert_allclose(h_n, np.zeros((1, 1), dtype=np.float32), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(c_n, np.zeros((1, 1), dtype=np.float32), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 3. Shape tests
# ---------------------------------------------------------------------------

def test_rnn_forward_shape():
    """RNN (T, N, D) → output (T, N, H), h_n (N, H)."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(0)
    x = rng.randn(T, N, D).astype(np.float32)
    rnn = sequence.RNN(D, H)
    rnn.load_weights(sequence.init_rnn_weights(D, H, seed=1))
    out, h_n = rnn.forward(x)
    assert out.shape == (T, N, H)
    assert h_n.shape == (N, H)
    assert np.all(np.isfinite(out))


def test_rnn_batch_first_shape():
    """RNN batch_first: x (N, T, D) → output (N, T, H)."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(2)
    x = rng.randn(N, T, D).astype(np.float32)
    rnn = sequence.RNN(D, H)
    rnn.load_weights(sequence.init_rnn_weights(D, H, seed=3))
    out, h_n = rnn.forward(x, batch_first=True)
    assert out.shape == (N, T, H)
    assert h_n.shape == (N, H)


def test_lstm_forward_shape():
    """LSTM (T, N, D) → output (T, N, H), h_n (N, H), c_n (N, H)."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(4)
    x = rng.randn(T, N, D).astype(np.float32)
    lstm = sequence.LSTM(D, H)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=5))
    out, (h_n, c_n) = lstm.forward(x)
    assert out.shape == (T, N, H)
    assert h_n.shape == (N, H)
    assert c_n.shape == (N, H)
    assert np.all(np.isfinite(out))


def test_bidirectional_shape():
    """Bidirectional LSTM: output (T, N, 2H), h_n/c_n (2, N, H)."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(6)
    x = rng.randn(T, N, D).astype(np.float32)
    lstm = sequence.LSTM(D, H, bidirectional=True)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=7))
    out, (h_n, c_n) = lstm.forward(x)
    assert out.shape == (T, N, 2 * H)
    assert h_n.shape == (2, N, H)
    assert c_n.shape == (2, N, H)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 4. Layout consistency (batch_first vs time-first)
# ---------------------------------------------------------------------------

def test_layout_consistency_rnn():
    """Same weights: batch_first=False equals batch_first=True (transposed)."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(8)
    x = rng.randn(T, N, D).astype(np.float32)
    rnn = sequence.RNN(D, H)
    rnn.load_weights(sequence.init_rnn_weights(D, H, seed=9))

    out_tf, _ = rnn.forward(x, batch_first=False)
    out_bf, _ = rnn.forward(x.transpose(1, 0, 2), batch_first=True)
    np.testing.assert_allclose(out_bf.transpose(1, 0, 2), out_tf, rtol=RTOL, atol=ATOL)


def test_layout_consistency_lstm():
    """Same weights: LSTM batch_first=False equals batch_first=True (transposed)."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(10)
    x = rng.randn(T, N, D).astype(np.float32)
    lstm = sequence.LSTM(D, H)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=11))

    out_tf, (hn_tf, cn_tf) = lstm.forward(x, batch_first=False)
    out_bf, (hn_bf, cn_bf) = lstm.forward(x.transpose(1, 0, 2), batch_first=True)
    np.testing.assert_allclose(out_bf.transpose(1, 0, 2), out_tf, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(hn_bf, hn_tf, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(cn_bf, cn_tf, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 5. Caffe-style packed weight loading consistency
# ---------------------------------------------------------------------------

def test_lstm_weights_consistency_caffe_packed():
    """Loading Caffe packed (W, b) matches loading the dict form directly."""
    D, H = 4, 8
    rng = np.random.RandomState(12)
    x = rng.randn(5, 3, D).astype(np.float32)

    w = sequence.init_lstm_weights(D, H, seed=13)
    packed_W, packed_b = sequence.pack_lstm_weights_caffe(**w)

    lstm_dict = sequence.LSTM(D, H)
    lstm_dict.load_weights(w)
    out_dict, (hn_dict, cn_dict) = lstm_dict.forward(x)

    lstm_packed = sequence.LSTM(D, H)
    # The packed form is passed via the W/b keyword args (the first positional
    # ``weights`` parameter is reserved for the dict form).
    lstm_packed.load_weights(W=packed_W, b=packed_b, fmt="caffe")
    out_packed, (hn_packed, cn_packed) = lstm_packed.forward(x)

    np.testing.assert_allclose(out_packed, out_dict, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(hn_packed, hn_dict, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(cn_packed, cn_dict, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 6. Final-state correctness
# ---------------------------------------------------------------------------

def test_rnn_final_state():
    """Unidirectional RNN: h_n equals the last time-step output."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(14)
    x = rng.randn(T, N, D).astype(np.float32)
    rnn = sequence.RNN(D, H)
    rnn.load_weights(sequence.init_rnn_weights(D, H, seed=15))
    out, h_n = rnn.forward(x)
    np.testing.assert_allclose(h_n, out[-1], rtol=RTOL, atol=ATOL)


def test_lstm_final_state():
    """Unidirectional LSTM: h_n equals the last time-step output."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(16)
    x = rng.randn(T, N, D).astype(np.float32)
    lstm = sequence.LSTM(D, H)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=17))
    out, (h_n, _c_n) = lstm.forward(x)
    np.testing.assert_allclose(h_n, out[-1], rtol=RTOL, atol=ATOL)


def test_bidirectional_final_state():
    """Bidirectional: h_n[0] = last forward state, h_n[1] = last backward state.

    Forward direction runs x[0..T-1] → its final state sits at output[-1, :, :H].
    Backward direction runs x[T-1..0] → its final state sits at output[0, :, H:].
    """
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(18)
    x = rng.randn(T, N, D).astype(np.float32)
    lstm = sequence.LSTM(D, H, bidirectional=True)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=19))
    out, (h_n, _c_n) = lstm.forward(x)

    np.testing.assert_allclose(h_n[0], out[-1, :, :H], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(h_n[1], out[0, :, H:], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# 7. numpy self-consistency (class API vs direct reference functions)
# ---------------------------------------------------------------------------

def test_rnn_numpy_consistency():
    """RNN.forward equals rnn_forward called directly with the same weights."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(20)
    x = rng.randn(T, N, D).astype(np.float32)
    w = sequence.init_rnn_weights(D, H, seed=21)

    rnn = sequence.RNN(D, H)
    rnn.load_weights(w)
    out_cls, h_cls = rnn.forward(x)

    out_ref, h_ref = sequence.rnn_forward(
        x,
        w["W_ih"], w["W_hh"], w["b_ih"], w["b_hh"],
        activation="tanh",
    )
    np.testing.assert_allclose(out_cls, out_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(h_cls, h_ref, rtol=RTOL, atol=ATOL)


def test_lstm_numpy_consistency():
    """LSTM.forward equals lstm_forward called directly with the same weights."""
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(22)
    x = rng.randn(T, N, D).astype(np.float32)
    w = sequence.init_lstm_weights(D, H, seed=23)

    lstm = sequence.LSTM(D, H)
    lstm.load_weights(w)
    out_cls, (hn_cls, cn_cls) = lstm.forward(x)

    out_ref, (hn_ref, cn_ref) = sequence.lstm_forward(
        x,
        w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
        w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
        w["b_ii"], w["b_if"], w["b_io"], w["b_ig"],
        w["b_hi"], w["b_hf"], w["b_ho"], w["b_hg"],
    )
    np.testing.assert_allclose(out_cls, out_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(hn_cls, hn_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(cn_cls, cn_ref, rtol=RTOL, atol=ATOL)