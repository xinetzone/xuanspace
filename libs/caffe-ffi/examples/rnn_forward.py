"""
RNN / LSTM forward-inference example using the pure-Python ``caffe_ffi.sequence`` API.

Demonstrates:
    1. Vanilla RNN forward (tanh)          -> output (T, N, H), final h_n (N, H)
    2. One-directional LSTM forward        -> output (T, N, H), final (h_n, c_n)
    3. Bidirectional LSTM forward           -> output (T, N, 2*H), final states (2, N, H)
    4. ``batch_first=True`` LSTM           -> input (N, T, D), output (N, T, H)
    5. Caffe-style packed weight loading   -> pack per-gate weights, load via fmt="caffe"

Run from the repo root:
    python examples/rnn_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from caffe_ffi import sequence

SEED = 42


def sec(title: str) -> None:
    print("-" * 60)
    print(title)
    print("-" * 60)


def demo_rnn(D: int, H: int, T: int, N: int) -> None:
    """Vanilla RNN forward inference."""
    sec("1. Vanilla RNN forward (tanh activation)")
    rnn = sequence.RNN(input_dim=D, hidden_dim=H, activation="tanh")
    rnn.load_weights(sequence.init_rnn_weights(D, H, seed=SEED))

    rng = np.random.RandomState(SEED)
    x = rng.randn(T, N, D).astype(np.float32)
    output, h_n = rnn.forward(x)

    print(f"input  x   : shape {x.shape}   (T, N, D) = ({T}, {N}, {D})")
    print(f"output     : shape {output.shape}   expects (T, N, H) = ({T}, {N}, {H})")
    print(f"final h_n  : shape {h_n.shape}   expects (N, H) = ({N}, {H})")
    print(f"last output[0] = {output[-1, 0, :3]}")
    print(f"h_n[0]         = {h_n[0, :3]}")
    print()


def demo_lstm(D: int, H: int, T: int, N: int) -> None:
    """One-directional LSTM forward inference."""
    sec("2. One-directional LSTM forward")
    lstm = sequence.LSTM(input_dim=D, hidden_dim=H, bidirectional=False)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=SEED))

    rng = np.random.RandomState(SEED)
    x = rng.randn(T, N, D).astype(np.float32)
    output, (h_n, c_n) = lstm.forward(x)

    print(f"input  x   : shape {x.shape}   (T, N, D) = ({T}, {N}, {D})")
    print(f"output     : shape {output.shape}   expects (T, N, H) = ({T}, {N}, {H})")
    print(f"final h_n  : shape {h_n.shape}   expects (N, H) = ({N}, {H})")
    print(f"final c_n  : shape {c_n.shape}   expects (N, H) = ({N}, {H})")
    print(f"last output[0] = {output[-1, 0, :3]}")
    print(f"h_n[0]         = {h_n[0, :3]}")
    print(f"c_n[0]         = {c_n[0, :3]}")
    print()


def demo_bidirectional(D: int, H: int, T: int, N: int) -> None:
    """Bidirectional LSTM forward inference."""
    sec("3. Bidirectional LSTM forward")
    lstm = sequence.LSTM(input_dim=D, hidden_dim=H, bidirectional=True)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=SEED))

    rng = np.random.RandomState(SEED)
    x = rng.randn(T, N, D).astype(np.float32)
    output, (h_n, c_n) = lstm.forward(x)

    print(f"input  x   : shape {x.shape}   (T, N, D) = ({T}, {N}, {D})")
    print(f"output     : shape {output.shape}   expects (T, N, 2*H) = ({T}, {N}, {2 * H})")
    print(f"final h_n  : shape {h_n.shape}   expects (2, N, H) = (2, {N}, {H})")
    print(f"final c_n  : shape {c_n.shape}   expects (2, N, H) = (2, {N}, {H})")
    print(f"last output[-1, 0] = {output[-1, 0, :3]}  (forward | backward concat)")
    print(f"h_n[0, 0] (fwd)    = {h_n[0, 0, :3]}")
    print(f"h_n[1, 0] (bwd)    = {h_n[1, 0, :3]}")
    print()


def demo_batch_first(D: int, H: int, T: int, N: int) -> None:
    """LSTM forward with ``batch_first=True`` (input N, T, D)."""
    sec("4. LSTM forward with batch_first=True")
    lstm = sequence.LSTM(input_dim=D, hidden_dim=H)
    lstm.load_weights(sequence.init_lstm_weights(D, H, seed=SEED))

    rng = np.random.RandomState(SEED)
    x = rng.randn(N, T, D).astype(np.float32)
    output, (h_n, c_n) = lstm.forward(x, batch_first=True)

    print(f"input  x   : shape {x.shape}   (N, T, D) = ({N}, {T}, {D})")
    print(f"output     : shape {output.shape}   expects (N, T, H) = ({N}, {T}, {H})")
    print(f"final h_n  : shape {h_n.shape}   expects (N, H) = ({N}, {H})")
    print(f"last output[0, -1] = {output[0, -1, :3]}")
    print(f"h_n[0]            = {h_n[0, :3]}")
    print()


def demo_caffe_packed(D: int, H: int, T: int, N: int) -> None:
    """Caffe-style packed weight loading for LSTM."""
    sec("5. LSTM with Caffe-style packed weights (fmt='caffe')")
    gates = sequence.init_lstm_weights(D, H, seed=SEED)
    W, b = sequence.pack_lstm_weights_caffe(
        gates["W_ii"], gates["W_if"], gates["W_io"], gates["W_ig"],
        gates["W_hi"], gates["W_hf"], gates["W_ho"], gates["W_hg"],
        gates["b_ii"], gates["b_if"], gates["b_io"], gates["b_ig"],
        gates["b_hi"], gates["b_hf"], gates["b_ho"], gates["b_hg"],
    )
    print(f"packed W shape: {W.shape}   expects (4*H, D+H) = ({4 * H}, {D + H})")
    print(f"packed b shape: {b.shape}   expects (4*H,) = ({4 * H},)")

    lstm = sequence.LSTM(input_dim=D, hidden_dim=H)
    lstm.load_weights(W=W, b=b, fmt="caffe")

    rng = np.random.RandomState(SEED)
    x = rng.randn(T, N, D).astype(np.float32)
    output, (h_n, c_n) = lstm.forward(x)

    print(f"input  x   : shape {x.shape}   (T, N, D) = ({T}, {N}, {D})")
    print(f"output     : shape {output.shape}   expects (T, N, H) = ({T}, {N}, {H})")
    print(f"final h_n  : shape {h_n.shape}   expects (N, H) = ({N}, {H})")
    print(f"last output[0] = {output[-1, 0, :3]}")
    print(f"h_n[0]         = {h_n[0, :3]}")
    print()


def main() -> None:
    print("=" * 60)
    print("Caffe-FFI RNN / LSTM Forward Inference Demo")
    print("=" * 60)
    print()

    D, H, T, N = 4, 8, 6, 3

    demo_rnn(D, H, T, N)
    demo_lstm(D, H, T, N)
    demo_bidirectional(D, H, T, N)
    demo_batch_first(D, H, T, N)
    demo_caffe_packed(D, H, T, N)

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()