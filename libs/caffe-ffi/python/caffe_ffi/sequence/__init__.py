"""Pure-Python RNN / LSTM forward inference.

This subpackage encapsulates the numpy-only RNN/LSTM reference implementation
(``_numpy_rnn_reference``) behind a small ``RNN`` / ``LSTM`` class API. It is
pure Python and does NOT require the C++ extension: useful for validating
network topology, generating reference outputs, and lightweight prototyping.

Example::

    from caffe_ffi import sequence

    lstm = sequence.LSTM(input_dim=4, hidden_dim=8, bidirectional=True)
    lstm.load_weights(sequence.init_lstm_weights(4, 8))

    import numpy as np
    x = np.random.randn(5, 3, 4).astype(np.float32)
    out, (h_n, c_n) = lstm.forward(x)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._numpy_rnn_reference import (
    rnn_forward,
    lstm_forward,
    pack_lstm_weights_caffe,
    unpack_lstm_weights_caffe,
    init_rnn_weights,
    init_lstm_weights,
)

__all__ = [
    "RNN",
    "LSTM",
    "rnn_forward",
    "lstm_forward",
    "pack_lstm_weights_caffe",
    "unpack_lstm_weights_caffe",
    "init_rnn_weights",
    "init_lstm_weights",
]


class RNN:
    """Vanilla RNN forward-inference wrapper.

    Args:
        input_dim: Input feature dimension (D).
        hidden_dim: Hidden state dimension (H).
        activation: 'tanh' or 'relu'.
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "tanh") -> None:
        if activation not in ("tanh", "relu"):
            raise ValueError(f"activation must be 'tanh' or 'relu', got {activation!r}")
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._activation = activation
        self._weights: Optional[dict[str, np.ndarray]] = None

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def activation(self) -> str:
        return self._activation

    def load_weights(self, weights: Optional[dict] = None, W: Any = None, b: Any = None, fmt: Optional[str] = None) -> None:
        """Load RNN weights.

        Two call forms are supported:

        * dict form: ``load_weights({"W_ih": ..., "W_hh": ..., "b_ih": ..., "b_hh": ...})``
        * Caffe packed form: ``load_weights(W, b, fmt="caffe")``

        RNN has no Caffe packed format in this reference implementation, so
        passing ``fmt="caffe"`` raises ``ValueError``.

        Args:
            weights: Dict of per-weight arrays.
            W: Packed weight matrix (only valid with ``fmt="caffe"``).
            b: Packed bias vector (only valid with ``fmt="caffe"``).
            fmt: Packed format name. Only ``"caffe"`` is recognized.

        Raises:
            ValueError: If ``fmt="caffe"`` is passed (RNN has no Caffe packed
                format) or if the required keys are missing.
        """
        if fmt is not None:
            if fmt == "caffe":
                raise ValueError(
                    "RNN has no Caffe packed format in this reference; "
                    "pass a dict of {'W_ih', 'W_hh', 'b_ih', 'b_hh'} instead."
                )
            raise ValueError(f"Unknown weight format: {fmt!r}")

        if weights is None:
            raise ValueError("load_weights requires a dict of weights for RNN")

        required = ("W_ih", "W_hh")
        missing = [k for k in required if k not in weights]
        if missing:
            raise ValueError(f"Missing required RNN weights: {missing}")

        self._weights = {
            "W_ih": np.asarray(weights["W_ih"], dtype=np.float32),
            "W_hh": np.asarray(weights["W_hh"], dtype=np.float32),
            "b_ih": np.asarray(weights.get("b_ih", np.zeros(self._hidden_dim, dtype=np.float32)), dtype=np.float32),
            "b_hh": np.asarray(weights.get("b_hh", np.zeros(self._hidden_dim, dtype=np.float32)), dtype=np.float32),
        }

    def forward(self, x: np.ndarray, batch_first: bool = False, h0: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Run RNN forward pass.

        Args:
            x: Input sequence (T, N, D) or (N, T, D) with ``batch_first=True``.
            batch_first: Whether ``x`` is batch-first.
            h0: Optional initial hidden state (N, H).

        Returns:
            (output, h_n): All hidden states and the final hidden state.
        """
        if self._weights is None:
            raise RuntimeError("weights not loaded; call load_weights() first")
        w = self._weights
        return rnn_forward(
            x,
            w["W_ih"],
            w["W_hh"],
            w["b_ih"],
            w["b_hh"],
            h0=h0,
            activation=self._activation,
            batch_first=batch_first,
        )


class LSTM:
    """LSTM forward-inference wrapper.

    Args:
        input_dim: Input feature dimension (D).
        hidden_dim: Hidden state dimension (H).
        bidirectional: If True, run forward + backward directions and
            concatenate hidden dim (output dim = 2*H).
    """

    def __init__(self, input_dim: int, hidden_dim: int, bidirectional: bool = False) -> None:
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._bidirectional = bidirectional
        self._weights: Optional[dict[str, np.ndarray]] = None

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    def load_weights(self, weights: Optional[dict] = None, W: Any = None, b: Any = None, fmt: Optional[str] = None) -> None:
        """Load LSTM weights.

        Two call forms are supported:

        * dict form: ``load_weights({"W_ii": ..., "W_if": ..., ..., "b_hg": ...})``
          providing all 4x2 per-gate weights and 4x2 biases.
        * Caffe packed form: ``load_weights(W, b, fmt="caffe")`` where ``W`` is
          ``(4*H, D+H)`` and ``b`` is ``(4*H,)``. Internally the packed blob is
          unpacked via :func:`unpack_lstm_weights_caffe`.

        Args:
            weights: Dict of per-gate weight arrays.
            W: Packed weight matrix (only valid with ``fmt="caffe"``).
            b: Packed bias vector (only valid with ``fmt="caffe"``).
            fmt: Packed format name. Only ``"caffe"`` is recognized.

        Raises:
            ValueError: If ``fmt`` is unsupported or the required keys are missing.
        """
        if fmt is not None:
            if fmt != "caffe":
                raise ValueError(f"Unknown weight format: {fmt!r}")
            # Support BOTH positional ``load_weights(W, b, fmt="caffe")`` and
            # keyword ``load_weights(W=W, b=b, fmt="caffe")``.
            # In the positional form, ``weights`` holds the packed matrix and
            # ``W`` holds the packed bias vector (``b`` stays None).
            if weights is not None and W is not None and b is None:
                W, b = weights, W
            if W is None or b is None:
                raise ValueError("Caffe packed form requires both W and b")
            weights = unpack_lstm_weights_caffe(
                np.asarray(W, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
                self._input_dim,
                self._hidden_dim,
            )

        if weights is None:
            raise ValueError("load_weights requires a dict of weights or W/b with fmt='caffe'")

        required = (
            "W_ii", "W_if", "W_io", "W_ig",
            "W_hi", "W_hf", "W_ho", "W_hg",
        )
        missing = [k for k in required if k not in weights]
        if missing:
            raise ValueError(f"Missing required LSTM weights: {missing}")

        self._weights = {
            k: np.asarray(weights[k], dtype=np.float32)
            for k in (
                "W_ii", "W_if", "W_io", "W_ig",
                "W_hi", "W_hf", "W_ho", "W_hg",
                "b_ii", "b_if", "b_io", "b_ig",
                "b_hi", "b_hf", "b_ho", "b_hg",
            )
        }

    def forward(self, x: np.ndarray, batch_first: bool = False, h0: Optional[np.ndarray] = None, c0: Optional[np.ndarray] = None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Run LSTM forward pass.

        Args:
            x: Input sequence (T, N, D) or (N, T, D) with ``batch_first=True``.
            batch_first: Whether ``x`` is batch-first.
            h0: Optional initial hidden state. For bidirectional, (2, N, H) or (N, H).
            c0: Optional initial cell state.

        Returns:
            (output, (h_n, c_n)): All hidden states and the final (hidden, cell).
        """
        if self._weights is None:
            raise RuntimeError("weights not loaded; call load_weights() first")
        w = self._weights
        return lstm_forward(
            x,
            w["W_ii"], w["W_if"], w["W_io"], w["W_ig"],
            w["W_hi"], w["W_hf"], w["W_ho"], w["W_hg"],
            w["b_ii"], w["b_if"], w["b_io"], w["b_ig"],
            w["b_hi"], w["b_hf"], w["b_ho"], w["b_hg"],
            h0=h0,
            c0=c0,
            batch_first=batch_first,
            bidirectional=self._bidirectional,
        )