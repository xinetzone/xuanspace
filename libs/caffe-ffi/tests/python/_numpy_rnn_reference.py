"""Pure-numpy RNN/LSTM forward reference implementation.

This module provides numpy-only RNN and LSTM forward computations
for quick network structure verification WITHOUT requiring the C++ extension.

Use cases:
    1. Validate LSTM/RNN network topology in prototxt before implementing
       C++ RecurrentLayer (e.g., verify shapes, connections, outputs exist)
    2. Generate reference outputs for testing future C++ LSTM implementation
    3. Lightweight prototyping of sequential models

Shape convention:
    Follows Caffe convention — T×N×D (timesteps × batch × feature_dim).
    Batch-first (N×T×D) is also supported via batch_first=True.

Weight layout (Caffe-compatible):
    W_ih: (hidden_dim, input_dim)      — input-to-hidden weights
    W_hh: (hidden_dim, hidden_dim)     — hidden-to-hidden weights
    b_ih: (hidden_dim,)                — input-to-hidden bias
    b_hh: (hidden_dim,)                — hidden-to-hidden bias (often zero)

    For LSTM there are 4 gate versions (i, f, o, g):
        W_ii, W_if, W_io, W_ig  — each (H, D)
        W_hi, W_hf, W_ho, W_hg  — each (H, H)
        b_ii, b_if, b_io, b_ig  — each (H,)
        b_hi, b_hf, b_ho, b_hg  — each (H,)

    Convenience: use concat_weights() to pack into Caffe-style
    blob[0] shape (4*H, D+H), blob[1] shape (4*H,).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
#  Vanilla RNN: h_t = act( W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh )
# ---------------------------------------------------------------------------

def rnn_forward(
    x: np.ndarray,
    W_ih: np.ndarray,
    W_hh: np.ndarray,
    b_ih: Optional[np.ndarray] = None,
    b_hh: Optional[np.ndarray] = None,
    h0: Optional[np.ndarray] = None,
    activation: str = "tanh",
    batch_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vanilla RNN forward pass.

    Args:
        x: Input sequence. Shape (T, N, D) if batch_first=False else (N, T, D).
        W_ih: Input-to-hidden weights. Shape (H, D).
        W_hh: Hidden-to-hidden weights. Shape (H, H).
        b_ih: Input bias. Shape (H,). Default zeros.
        b_hh: Hidden bias. Shape (H,). Default zeros.
        h0: Initial hidden state. Shape (N, H). Default zeros.
        activation: 'tanh' or 'relu'.
        batch_first: If True, x is (N, T, D); otherwise (T, N, D).

    Returns:
        output: All hidden states. Shape matches x layout.
        h_n: Final hidden state. Shape (N, H).
    """
    if batch_first:
        x = x.transpose(1, 0, 2)  # → (T, N, D)

    T, N, D = x.shape
    H = W_ih.shape[0]
    assert W_ih.shape == (H, D), f"W_ih must be ({H},{D}), got {W_ih.shape}"
    assert W_hh.shape == (H, H), f"W_hh must be ({H},{H}), got {W_hh.shape}"

    b_ih = _ensure_1d(b_ih, H, "b_ih")
    b_hh = _ensure_1d(b_hh, H, "b_hh")

    h = np.zeros((N, H), dtype=np.float64) if h0 is None else h0.astype(np.float64)
    outputs = np.zeros((T, N, H), dtype=np.float64)

    act_fn = _get_activation(activation)

    for t in range(T):
        x_t = x[t].astype(np.float64)
        h = act_fn(x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        outputs[t] = h

    if batch_first:
        outputs = outputs.transpose(1, 0, 2)

    return outputs.astype(np.float32), h.astype(np.float32)


# ---------------------------------------------------------------------------
#  LSTM:  i_t=σ(W_ii·x_t + W_hi·h_{t-1} + b_ii + b_hi)
#         f_t=σ(W_if·x_t + W_hf·h_{t-1} + b_if + b_hf)
#         g_t=tanh(W_ig·x_t + W_hg·h_{t-1} + b_ig + b_hg)
#         o_t=σ(W_io·x_t + W_ho·h_{t-1} + b_io + b_ho)
#         c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
#         h_t = o_t ⊙ tanh(c_t)
# ---------------------------------------------------------------------------

def lstm_forward(
    x: np.ndarray,
    W_ii: np.ndarray, W_if: np.ndarray, W_io: np.ndarray, W_ig: np.ndarray,
    W_hi: np.ndarray, W_hf: np.ndarray, W_ho: np.ndarray, W_hg: np.ndarray,
    b_ii: Optional[np.ndarray] = None,
    b_if: Optional[np.ndarray] = None,
    b_io: Optional[np.ndarray] = None,
    b_ig: Optional[np.ndarray] = None,
    b_hi: Optional[np.ndarray] = None,
    b_hf: Optional[np.ndarray] = None,
    b_ho: Optional[np.ndarray] = None,
    b_hg: Optional[np.ndarray] = None,
    h0: Optional[np.ndarray] = None,
    c0: Optional[np.ndarray] = None,
    batch_first: bool = False,
    bidirectional: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """LSTM forward pass (single-direction).

    Args:
        x: Input sequence (T,N,D) or (N,T,D) with batch_first=True.
        W_ii/W_if/W_io/W_ig: Input-to-hidden weights for gates (i,f,o,g). Shape (H,D).
        W_hi/W_hf/W_ho/W_hg: Hidden-to-hidden weights (H,H).
        b_** : Optional biases (H,). Default zeros.
        h0: Initial hidden state (N,H). Default zeros.
        c0: Initial cell state (N,H). Default zeros.
        batch_first: x shape convention.
        bidirectional: If True, runs forward + backward directions and
                       concatenates hidden dim (output dim = 2*H).

    Returns:
        output: All time-step hidden states. (T,N,H) or (N,T,H);
                bidirectional → (T,N,2*H).
        (h_n, c_n): Final hidden/cell states. Each (N,H) unidirectional,
                    (2,N,H) bidirectional [forward_layer, backward_layer].
    """
    if not bidirectional:
        return _lstm_unidirectional(
            x,
            W_ii, W_if, W_io, W_ig,
            W_hi, W_hf, W_ho, W_hg,
            b_ii, b_if, b_io, b_ig,
            b_hi, b_hf, b_ho, b_hg,
            h0, c0, batch_first, reverse=False,
        )

    # Bidirectional: forward + backward directions
    # Forward pass uses h0[:H], c0[:H]; backward uses h0[H:], c0[H:]
    N = x.shape[0] if batch_first else x.shape[1]
    H = W_ii.shape[0]
    h0_f, h0_b = _split_initial(h0, N, H)
    c0_f, c0_b = _split_initial(c0, N, H)

    out_f, (hn_f, cn_f) = _lstm_unidirectional(
        x, W_ii, W_if, W_io, W_ig,
        W_hi, W_hf, W_ho, W_hg,
        b_ii, b_if, b_io, b_ig,
        b_hi, b_hf, b_ho, b_hg,
        h0_f, c0_f, batch_first, reverse=False,
    )
    out_b, (hn_b, cn_b) = _lstm_unidirectional(
        x, W_ii, W_if, W_io, W_ig,
        W_hi, W_hf, W_ho, W_hg,
        b_ii, b_if, b_io, b_ig,
        b_hi, b_hf, b_ho, b_hg,
        h0_b, c0_b, batch_first, reverse=True,
    )
    # Concatenate forward and backward along hidden dimension
    output = np.concatenate([out_f, out_b], axis=-1)
    h_n = np.stack([hn_f, hn_b], axis=0)  # (2, N, H)
    c_n = np.stack([cn_f, cn_b], axis=0)
    return output.astype(np.float32), (h_n.astype(np.float32), c_n.astype(np.float32))


def _lstm_unidirectional(
    x, W_ii, W_if, W_io, W_ig,
    W_hi, W_hf, W_ho, W_hg,
    b_ii, b_if, b_io, b_ig,
    b_hi, b_hf, b_ho, b_hg,
    h0, c0, batch_first, reverse=False,
):
    """Single-direction LSTM forward."""
    if batch_first:
        x = x.transpose(1, 0, 2)

    T, N, D = x.shape
    H = W_ii.shape[0]

    for name, w, shape in [
        ("W_ii", W_ii, (H, D)), ("W_if", W_if, (H, D)),
        ("W_io", W_io, (H, D)), ("W_ig", W_ig, (H, D)),
        ("W_hi", W_hi, (H, H)), ("W_hf", W_hf, (H, H)),
        ("W_ho", W_ho, (H, H)), ("W_hg", W_hg, (H, H)),
    ]:
        assert w.shape == shape, f"{name} must be {shape}, got {w.shape}"

    b_ii = _ensure_1d(b_ii, H, "b_ii")
    b_if = _ensure_1d(b_if, H, "b_if")
    b_io = _ensure_1d(b_io, H, "b_io")
    b_ig = _ensure_1d(b_ig, H, "b_ig")
    b_hi = _ensure_1d(b_hi, H, "b_hi")
    b_hf = _ensure_1d(b_hf, H, "b_hf")
    b_ho = _ensure_1d(b_ho, H, "b_ho")
    b_hg = _ensure_1d(b_hg, H, "b_hg")

    # Combine biases: b_i = b_ii + b_hi
    b_i = (b_ii + b_hi).astype(np.float64)
    b_f = (b_if + b_hf).astype(np.float64)
    b_o = (b_io + b_ho).astype(np.float64)
    b_g = (b_ig + b_hg).astype(np.float64)

    h = np.zeros((N, H), dtype=np.float64) if h0 is None else h0.astype(np.float64)
    c = np.zeros((N, H), dtype=np.float64) if c0 is None else c0.astype(np.float64)
    outputs = np.zeros((T, N, H), dtype=np.float64)

    time_steps = range(T - 1, -1, -1) if reverse else range(T)
    out_indices = range(T - 1, -1, -1) if reverse else range(T)

    for t, out_idx in zip(time_steps, out_indices):
        x_t = x[t].astype(np.float64)
        # Gates
        i_t = _sigmoid(x_t @ W_ii.T + h @ W_hi.T + b_i)
        f_t = _sigmoid(x_t @ W_if.T + h @ W_hf.T + b_f)
        o_t = _sigmoid(x_t @ W_io.T + h @ W_ho.T + b_o)
        g_t = np.tanh(x_t @ W_ig.T + h @ W_hg.T + b_g)
        # Cell state
        c = f_t * c + i_t * g_t
        # Hidden state
        h = o_t * np.tanh(c)
        outputs[out_idx] = h

    if batch_first:
        outputs = outputs.transpose(1, 0, 2)

    return outputs, (h, c)


# ---------------------------------------------------------------------------
#  Caffe-style packed weights helper
# ---------------------------------------------------------------------------

def pack_lstm_weights_caffe(
    W_ii, W_if, W_io, W_ig,
    W_hi, W_hf, W_ho, W_hg,
    b_ii=None, b_if=None, b_io=None, b_ig=None,
    b_hi=None, b_hf=None, b_ho=None, b_hg=None,
):
    """Pack LSTM weights into Caffe-style blob format.

    In Caffe's LSTMLayer, weights are stored as:
        blob[0] shape (4*H, D+H): [W_ih | W_hh] concatenated for all 4 gates
                  row order: input gate, forget gate, output gate, cell gate
        blob[1] shape (4*H,):     concatenated biases [b_i; b_f; b_o; b_g]

    Returns:
        (W_blobs, b_blob) where W_blobs is (4*H, D+H), b_blob is (4*H,).
    """
    H, D = W_ii.shape
    # Stack input weights: (4*H, D)
    W_ih = np.vstack([W_ii, W_if, W_io, W_ig])
    # Stack recurrent weights: (4*H, H)
    W_hh = np.vstack([W_hi, W_hf, W_ho, W_hg])
    # Concat: (4*H, D+H)
    W = np.hstack([W_ih, W_hh]).astype(np.float32)

    b_ii = _ensure_1d(b_ii, H, "b_ii")
    b_if = _ensure_1d(b_if, H, "b_if")
    b_io = _ensure_1d(b_io, H, "b_io")
    b_ig = _ensure_1d(b_ig, H, "b_ig")
    b_hi = _ensure_1d(b_hi, H, "b_hi")
    b_hf = _ensure_1d(b_hf, H, "b_hf")
    b_ho = _ensure_1d(b_ho, H, "b_ho")
    b_hg = _ensure_1d(b_hg, H, "b_hg")

    b = np.concatenate([
        b_ii + b_hi,
        b_if + b_hf,
        b_io + b_ho,
        b_ig + b_hg,
    ]).astype(np.float32)

    return W, b


def unpack_lstm_weights_caffe(W: np.ndarray, b: np.ndarray, D: int, H: int):
    """Unpack Caffe-style blob weights into individual gate matrices.

    Args:
        W: (4*H, D+H) packed weight.
        b: (4*H,) packed bias.
        D: input feature dimension.
        H: hidden dimension.

    Returns:
        Dict with keys W_ii/W_if/W_io/W_ig/W_hi/W_hf/W_ho/W_hg/b_i/b_f/b_o/b_g.
    """
    assert W.shape == (4 * H, D + H), f"Expected ({4*H},{D+H}), got {W.shape}"
    assert b.shape == (4 * H,), f"Expected ({4*H},), got {b.shape}"

    W_ih = W[:, :D]   # (4*H, D)
    W_hh = W[:, D:]   # (4*H, H)

    return {
        "W_ii": W_ih[0*H:1*H], "W_if": W_ih[1*H:2*H],
        "W_io": W_ih[2*H:3*H], "W_ig": W_ih[3*H:4*H],
        "W_hi": W_hh[0*H:1*H], "W_hf": W_hh[1*H:2*H],
        "W_ho": W_hh[2*H:3*H], "W_hg": W_hh[3*H:4*H],
        "b_ii": b[0*H:1*H], "b_if": b[1*H:2*H],
        "b_io": b[2*H:3*H], "b_ig": b[3*H:4*H],
        # b_hi/b_hf/b_ho/b_hg are zero when unpacked from Caffe format
        # (Caffe combines biases into single per-gate bias)
        "b_hi": np.zeros(H, dtype=np.float32),
        "b_hf": np.zeros(H, dtype=np.float32),
        "b_ho": np.zeros(H, dtype=np.float32),
        "b_hg": np.zeros(H, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
#  LSTM Cell (single timestep) — useful for unrolled network debugging
# ---------------------------------------------------------------------------

def lstm_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    W_ii, W_if, W_io, W_ig,
    W_hi, W_hf, W_ho, W_hg,
    b_i=None, b_f=None, b_o=None, b_g=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single LSTM cell step.

    Args:
        x_t: Input at time t. Shape (N, D).
        h_prev: Previous hidden state (N, H).
        c_prev: Previous cell state (N, H).
        W_** : Weight matrices as in lstm_forward.
        b_i/b_f/b_o/b_g: Combined bias (input + recurrent) (H,).

    Returns:
        h_t: New hidden state (N, H).
        c_t: New cell state (N, H).
        cache: Dict of intermediate values (gates, etc.) for debugging.
    """
    H = W_ii.shape[0]
    b_i = _ensure_1d(b_i, H, "b_i")
    b_f = _ensure_1d(b_f, H, "b_f")
    b_o = _ensure_1d(b_o, H, "b_o")
    b_g = _ensure_1d(b_g, H, "b_g")

    x_t = x_t.astype(np.float64)
    h_prev = h_prev.astype(np.float64)
    c_prev = c_prev.astype(np.float64)

    i_t = _sigmoid(x_t @ W_ii.T + h_prev @ W_hi.T + b_i)
    f_t = _sigmoid(x_t @ W_if.T + h_prev @ W_hf.T + b_f)
    o_t = _sigmoid(x_t @ W_io.T + h_prev @ W_ho.T + b_o)
    g_t = np.tanh(x_t @ W_ig.T + h_prev @ W_hg.T + b_g)

    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t)

    cache = {
        "i_t": i_t.astype(np.float32),
        "f_t": f_t.astype(np.float32),
        "o_t": o_t.astype(np.float32),
        "g_t": g_t.astype(np.float32),
        "c_t": c_t.astype(np.float32),
    }
    return h_t.astype(np.float32), c_t.astype(np.float32), cache


# ---------------------------------------------------------------------------
#  Weight initialization helpers
# ---------------------------------------------------------------------------

def init_rnn_weights(D: int, H: int, seed: int = 42, scale: float = 0.1):
    """Initialize vanilla RNN weights with small random values.

    Returns dict with W_ih, W_hh, b_ih, b_hh.
    """
    rng = np.random.RandomState(seed)
    return {
        "W_ih": (rng.randn(H, D) * scale).astype(np.float32),
        "W_hh": (rng.randn(H, H) * scale / np.sqrt(H)).astype(np.float32),
        "b_ih": np.zeros(H, dtype=np.float32),
        "b_hh": np.zeros(H, dtype=np.float32),
    }


def init_lstm_weights(D: int, H: int, seed: int = 42, scale: float = 0.1):
    """Initialize LSTM weights with Xavier/Glorot-like scaling.

    Returns dict with all 4×2 weight matrices and 4×2 bias vectors.
    Biases initialized to zero except forget gate bias = 1.0
    (Jozefowicz et al., 2015 — helps learn long-term dependencies).
    """
    rng = np.random.RandomState(seed)
    s = scale / np.sqrt(D + H)
    W = {}
    for gate in ("i", "f", "o", "g"):
        W[f"W_i{gate}"] = (rng.randn(H, D) * s).astype(np.float32)  # input-to-gate
        W[f"W_h{gate}"] = (rng.randn(H, H) * s).astype(np.float32)  # hidden-to-gate
        W[f"b_i{gate}"] = np.zeros(H, dtype=np.float32)
        W[f"b_h{gate}"] = np.zeros(H, dtype=np.float32)
    # Forget gate bias = 1.0 (combined b_if + b_hf = 1.0)
    W["b_if"] = np.ones(H, dtype=np.float32) * 0.5
    W["b_hf"] = np.ones(H, dtype=np.float32) * 0.5
    return W


# ---------------------------------------------------------------------------
#  Known-value verification (self-test)
# ---------------------------------------------------------------------------

def _known_values_rnn():
    """Return known inputs/weights/outputs for RNN verification.

    Uses a minimal 1-timestep, batch=1, D=2, H=2 case computed manually.
    """
    # x_t = [0.5, -0.3], h0 = [0, 0]
    # W_ih = [[1, 0], [0, 1]] (identity), W_hh = [[0,0],[0,0]], b=0
    # tanh(W_ih @ x_t) = tanh([0.5, -0.3]) = [0.4621..., -0.2913...]
    x = np.array([[[0.5, -0.3]]], dtype=np.float32)  # (1,1,2)
    W_ih = np.eye(2, dtype=np.float32)
    W_hh = np.zeros((2, 2), dtype=np.float32)
    b_ih = np.zeros(2, dtype=np.float32)
    b_hh = np.zeros(2, dtype=np.float32)
    expected_h = np.tanh(np.array([0.5, -0.3], dtype=np.float64)).astype(np.float32)
    return x, W_ih, W_hh, b_ih, b_hh, expected_h


def _known_values_lstm():
    """Return known inputs/weights/outputs for LSTM verification.

    Single timestep with all weights = 0 and biases = 0.
    x_t = [0.5], h0 = [0], c0 = [0]
    All gates: sigmoid(0) = 0.5 for i,f,o; tanh(0) = 0 for g
    c_1 = f*c0 + i*g = 0.5*0 + 0.5*0 = 0
    h_1 = o*tanh(c_1) = 0.5*tanh(0) = 0
    """
    x = np.array([[[0.5]]], dtype=np.float32)  # (1,1,1)
    H = 1
    D = 1
    W = {g: np.zeros((H, D), dtype=np.float32) for g in ("W_ii", "W_if", "W_io", "W_ig")}
    W.update({g: np.zeros((H, H), dtype=np.float32) for g in ("W_hi", "W_hf", "W_ho", "W_hg")})
    b = {g: np.zeros(H, dtype=np.float32) for g in (
        "b_ii", "b_if", "b_io", "b_ig", "b_hi", "b_hf", "b_ho", "b_hg")}
    # With zero weights/biases: all gates = sigmoid(0)=0.5, g=tanh(0)=0
    # c1 = 0.5*0 + 0.5*0 = 0 → h1 = 0.5*tanh(0) = 0
    expected_h = np.zeros((1, 1), dtype=np.float32)
    expected_c = np.zeros((1, 1), dtype=np.float32)
    return x, W, b, expected_h, expected_c


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _ensure_1d(b, H, name):
    if b is None:
        return np.zeros(H, dtype=np.float32)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    assert b.shape == (H,), f"{name} must be ({H},), got {b.shape}"
    return b


def _sigmoid(x):
    """Numerically stable sigmoid."""
    pos = x >= 0
    neg = ~pos
    z = np.zeros_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    z[neg] = exp_x / (1.0 + exp_x)
    return z


def _get_activation(name: str):
    if name == "tanh":
        return np.tanh
    elif name == "relu":
        return lambda x: np.maximum(0, x)
    else:
        raise ValueError(f"Unknown activation: {name}")


def _split_initial(state, N, H):
    """Split bidirectional initial state into forward/backward halves."""
    if state is None:
        return None, None
    s = np.asarray(state, dtype=np.float64)
    if s.shape == (2, N, H):
        return s[0], s[1]
    elif s.shape == (N, H):
        return s, s
    else:
        raise ValueError(f"Expected initial state (2,{N},{H}) or ({N},{H}), got {s.shape}")


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Run built-in verification."""
    import sys

    print("=== numpy RNN/LSTM self-test ===\n")

    # Test 1: RNN known value
    print("[1] RNN known-value check ...", end=" ")
    x, W_ih, W_hh, b_ih, b_hh, expected = _known_values_rnn()
    out, h_n = rnn_forward(x, W_ih, W_hh, b_ih, b_hh)
    assert np.allclose(out[0, 0], expected, atol=1e-6), f"RNN failed: {out[0,0]} vs {expected}"
    assert np.allclose(h_n[0], expected, atol=1e-6)
    print("OK")

    # Test 2: LSTM known value
    print("[2] LSTM known-value (zero weights) ...", end=" ")
    x, W, b, exp_h, exp_c = _known_values_lstm()
    out, (h_n, c_n) = lstm_forward(
        x, W["W_ii"], W["W_if"], W["W_io"], W["W_ig"],
        W["W_hi"], W["W_hf"], W["W_ho"], W["W_hg"],
        **b)
    assert np.allclose(h_n, exp_h, atol=1e-6), f"LSTM h_n failed: {h_n}"
    assert np.allclose(c_n, exp_c, atol=1e-6), f"LSTM c_n failed: {c_n}"
    print("OK")

    # Test 3: LSTM random weights, shape check
    print("[3] LSTM random weights shape check ...", end=" ")
    T, N, D, H = 5, 3, 4, 8
    rng = np.random.RandomState(123)
    x = rng.randn(T, N, D).astype(np.float32)
    w = init_lstm_weights(D, H, seed=42)
    out, (h_n, c_n) = lstm_forward(
        x, w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
        w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
        w["b_ii"], w["b_if"], w["b_io"], w["b_ig"],
        w["b_hi"], w["b_hf"], w["b_ho"], w["b_hg"],
    )
    assert out.shape == (T, N, H), f"Expected ({T},{N},{H}), got {out.shape}"
    assert h_n.shape == (N, H), f"Expected ({N},{H}), got {h_n.shape}"
    assert c_n.shape == (N, H)
    assert np.all(np.isfinite(out)), "Non-finite values in output"
    print(f"OK (output shape {out.shape}, h_n {h_n.shape})")

    # Test 4: Bidirectional LSTM
    print("[4] Bidirectional LSTM shape check ...", end=" ")
    out_bi, (hn_bi, cn_bi) = lstm_forward(
        x, w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
        w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
        w["b_ii"], w["b_if"], w["b_io"], w["b_ig"],
        w["b_hi"], w["b_hf"], w["b_ho"], w["b_hg"],
        bidirectional=True,
    )
    assert out_bi.shape == (T, N, 2 * H)
    assert hn_bi.shape == (2, N, H)
    assert cn_bi.shape == (2, N, H)
    print(f"OK (output {out_bi.shape}, h_n {hn_bi.shape})")

    # Test 5: Caffe weight pack/unpack roundtrip
    print("[5] Caffe weight pack/unpack roundtrip ...", end=" ")
    W_caffe, b_caffe = pack_lstm_weights_caffe(
        w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
        w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
        w["b_ii"], w["b_if"], w["b_io"], w["b_ig"],
        w["b_hi"], w["b_hf"], w["b_ho"], w["b_hg"],
    )
    assert W_caffe.shape == (4 * H, D + H), f"W_caffe shape {W_caffe.shape}"
    assert b_caffe.shape == (4 * H,)
    unpacked = unpack_lstm_weights_caffe(W_caffe, b_caffe, D, H)
    for key in ["W_ii", "W_if", "W_io", "W_ig", "W_hi", "W_hf", "W_ho", "W_hg"]:
        assert np.allclose(unpacked[key], w[key], atol=1e-6), f"Weight mismatch: {key}"
    # Biases are combined, check sum
    assert np.allclose(unpacked["b_ii"], w["b_ii"] + w["b_hi"], atol=1e-6)
    print("OK")

    # Test 6: Batch-first mode
    print("[6] Batch-first RNN shape check ...", end=" ")
    x_bf = rng.randn(N, T, D).astype(np.float32)
    w_rnn = init_rnn_weights(D, H, seed=99)
    out_bf, hn_bf = rnn_forward(x_bf, w_rnn["W_ih"], w_rnn["W_hh"],
                                 w_rnn["b_ih"], w_rnn["b_hh"],
                                 batch_first=True)
    assert out_bf.shape == (N, T, H)
    assert hn_bf.shape == (N, H)
    print("OK")

    # Test 7: LSTM cell step
    print("[7] LSTM single-cell step ...", end=" ")
    x_t = rng.randn(N, D).astype(np.float32)
    h0 = np.zeros((N, H), dtype=np.float32)
    c0 = np.zeros((N, H), dtype=np.float32)
    h1, c1, cache = lstm_cell(
        x_t, h0, c0,
        w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
        w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
    )
    assert h1.shape == (N, H)
    assert c1.shape == (N, H)
    assert "i_t" in cache and "f_t" in cache and "o_t" in cache and "g_t" in cache
    # Gate values should be in (0,1) for sigmoid gates
    assert np.all(cache["i_t"] > 0) and np.all(cache["i_t"] < 1)
    assert np.all(cache["f_t"] > 0) and np.all(cache["f_t"] < 1)
    assert np.all(cache["o_t"] > 0) and np.all(cache["o_t"] < 1)
    print("OK")

    # Test 8: Determinism (same inputs → same outputs)
    print("[8] Determinism check ...", end=" ")
    out1, (hn1, cn1) = lstm_forward(
        x, w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
        w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
        w["b_ii"], w["b_if"], w["b_io"], w["b_ig"],
        w["b_hi"], w["b_hf"], w["b_ho"], w["b_hg"],
    )
    out2, (hn2, cn2) = lstm_forward(
        x, w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
        w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
        w["b_ii"], w["b_if"], w["b_io"], w["b_ig"],
        w["b_hi"], w["b_hf"], w["b_ho"], w["b_hg"],
    )
    assert np.array_equal(out1, out2), "LSTM not deterministic"
    assert np.array_equal(hn1, hn2)
    print("OK")

    print(f"\n=== All 8 self-tests PASSED ===")
    return True


if __name__ == "__main__":
    _self_test()
