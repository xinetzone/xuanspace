"""Generate deterministic mock data + reference outputs for RNN/LSTM forward.

This script produces a fixed, reproducible set of RNN/LSTM inputs, weights and
forward reference outputs using the Phase 1 pure-Python ``caffe_ffi.sequence``
implementation. The emitted artifacts are the ground-truth baseline for:

  1. Validating the Phase 2 C++ ``RecurrentLayer``/``RNNLayer``/
     ``LSTMUnit``/``LSTMLayer`` forward pass (rtol <= 1e-5 vs these values).
  2. Building the L2 numpy-backward reference test harness (Task 2 / Task 5).

The script is pure numpy and does NOT require the C++ extension. It is the
"mock data" harness the user asked to create BEFORE writing C++ code.

Usage::

    python examples/mock_sequence_data.py                  # run + self-check
    python examples/mock_sequence_data.py --save out.json  # also dump JSON

Artifacts (JSON) contain, per case:
    - config:  input_dim / hidden_dim / num_steps / num_batch / activation
    - input:   x (T, N, D), float32
    - weights: packed Caffe-style (W, b) for LSTM; dict for RNN
    - output:  all hidden states (T, N, H)
    - final:   h_n (and c_n for LSTM)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from caffe_ffi import sequence


def _tolist(a: np.ndarray) -> list:
    return np.asarray(a, dtype=np.float32).tolist()


def make_rnn_case(seed: int = 0, T: int = 4, N: int = 2, D: int = 3, H: int = 4,
                  activation: str = "tanh") -> dict:
    """Deterministic RNN forward case with reference output."""
    rng = np.random.RandomState(seed)
    x = (rng.randn(T, N, D) * 0.5).astype(np.float32)
    w = sequence.init_rnn_weights(D, H, seed=seed + 1, scale=0.2)
    h0 = (rng.randn(N, H) * 0.1).astype(np.float32)

    rnn = sequence.RNN(D, H, activation=activation)
    rnn.load_weights(w)
    out, h_n = rnn.forward(x, h0=h0)

    return {
        "type": "rnn",
        "config": {
            "input_dim": D, "hidden_dim": H, "num_steps": T,
            "num_batch": N, "activation": activation,
        },
        "input": _tolist(x),
        "h0": _tolist(h0),
        "weights": {k: _tolist(v) for k, v in w.items()},
        "output": _tolist(out),
        "final": _tolist(h_n),
    }


def make_lstm_case(seed: int = 0, T: int = 4, N: int = 2, D: int = 3, H: int = 4,
                   bidirectional: bool = False) -> dict:
    """Deterministic LSTM forward case with reference output (packed Caffe weights)."""
    rng = np.random.RandomState(seed)
    x = (rng.randn(T, N, D) * 0.5).astype(np.float32)
    w = sequence.init_lstm_weights(D, H, seed=seed + 1, scale=0.2)
    h0 = (rng.randn(2, N, H) * 0.1).astype(np.float32) if bidirectional else \
        (rng.randn(N, H) * 0.1).astype(np.float32)
    c0 = (rng.randn(N, H) * 0.1).astype(np.float32)

    lstm = sequence.LSTM(D, H, bidirectional=bidirectional)
    lstm.load_weights(w)  # dict form
    out, (h_n, c_n) = lstm.forward(x, h0=h0, c0=c0)

    packed_W, packed_b = sequence.pack_lstm_weights_caffe(**w)

    return {
        "type": "lstm",
        "config": {
            "input_dim": D, "hidden_dim": H, "num_steps": T,
            "num_batch": N, "bidirectional": bidirectional,
        },
        "input": _tolist(x),
        "h0": _tolist(h0),
        "c0": _tolist(c0),
        "weights": {"W": _tolist(packed_W), "b": _tolist(packed_b)},
        "output": _tolist(out),
        "final": {"h_n": _tolist(h_n), "c_n": _tolist(c_n)},
    }


def self_check(cases: list[dict]) -> None:
    """Re-run forward from the JSON-serialized artifacts to confirm round-trip."""
    print("=" * 60)
    print("Self-check: round-trip forward from serialized mock data")
    print("=" * 60)
    ok = True
    for i, case in enumerate(cases):
        cfg = case["config"]
        x = np.asarray(case["input"], dtype=np.float32)
        if case["type"] == "rnn":
            obj = sequence.RNN(cfg["input_dim"], cfg["hidden_dim"],
                               activation=cfg["activation"])
            obj.load_weights(case["weights"])
            out, h_n = obj.forward(x, h0=np.asarray(case["h0"], dtype=np.float32))
            ref_out, ref_h = np.asarray(case["output"], np.float32), \
                np.asarray(case["final"], np.float32)
            rtol_out = np.max(np.abs(out - ref_out)) / (np.max(np.abs(ref_out)) + 1e-12)
            rtol_h = np.max(np.abs(h_n - ref_h)) / (np.max(np.abs(ref_h)) + 1e-12)
            status = "PASS" if max(rtol_out, rtol_h) <= 1e-5 else "FAIL"
            print(f"  [{i}] RNN out={out.shape} h_n={h_n.shape} "
                  f"max_rtol={max(rtol_out, rtol_h):.2e} {status}")
            ok = ok and status == "PASS"
        else:
            obj = sequence.LSTM(cfg["input_dim"], cfg["hidden_dim"],
                                bidirectional=cfg["bidirectional"])
            obj.load_weights(W=case["weights"]["W"], b=case["weights"]["b"],
                             fmt="caffe")
            out, (h_n, c_n) = obj.forward(
                x, h0=np.asarray(case["h0"], np.float32),
                c0=np.asarray(case["c0"], np.float32))
            ref_out = np.asarray(case["output"], np.float32)
            ref_h = np.asarray(case["final"]["h_n"], np.float32)
            ref_c = np.asarray(case["final"]["c_n"], np.float32)
            rm = max(
                np.max(np.abs(out - ref_out)),
                np.max(np.abs(h_n - ref_h)),
                np.max(np.abs(c_n - ref_c)),
            )
            status = "PASS" if rm / (np.max(np.abs(ref_out)) + 1e-12) <= 1e-5 else "FAIL"
            print(f"  [{i}] LSTM out={out.shape} h_n={h_n.shape} c_n={c_n.shape} "
                  f"max_abs_rtol={rm:.2e} {status}")
            ok = ok and status == "PASS"
    print("=" * 60)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=str, default=None,
                        help="Path to JSON file to dump mock data (+ reference).")
    args = parser.parse_args()

    cases = [
        make_rnn_case(seed=0, activation="tanh"),
        make_rnn_case(seed=1, activation="relu"),
        make_lstm_case(seed=2),
        make_lstm_case(seed=3, bidirectional=True),
    ]

    ok = self_check(cases)

    if args.save:
        dest = Path(args.save)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(cases, indent=2), encoding="utf-8")
        print(f"Mock data saved to {dest}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())