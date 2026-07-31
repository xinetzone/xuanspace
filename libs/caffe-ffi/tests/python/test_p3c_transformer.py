"""P3-C: Transformer Components — Self-Attention and Positional Encoding Tests.

Comprehensive tests covering the building blocks of Transformer architectures
using existing caffe-ffi layers (no new C++ layers required):

Positional Encoding:
- Sinusoidal PE via pre-computed Input + Eltwise SUM (same-shape, pre-broadcast in numpy)
- Learnable PE via Bias layer (per-position bias with broadcasting)
- Deterministic repeated forward

Self-Attention Components:
- Q/K/V linear projections via InnerProduct
- Attention scaling via Scale layer (1/sqrt(d_k))
- Attention softmax normalization via Softmax
- Residual connections via implicit fan-out + Eltwise SUM
- Attention-weighted sum via InnerProduct (V as weight matrix)

End-to-end:
- Scaled Dot-Product Attention (SDP Attention)
- Multi-head projection via multiple InnerProduct + Concat
- Simplified Encoder Block integration

Each test includes numpy reference implementations and detailed perf_trace
logging of forward time, RSS memory peaks, and exception details.

Run with:
    pytest tests/python/test_p3c_transformer.py -v
    pytest tests/python/test_p3c_transformer.py -v -s  # verbose with [PERF] logs
"""
from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import net_param_from_string, net_from_param
from .conftest import require_cpp_extension, perf_trace


# ═══════════════════════════════════════════════════════════════════════
# Numpy Reference Implementations
# ═══════════════════════════════════════════════════════════════════════

def sinusoidal_pe_np(seq_len: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length
        d_model: Model dimension (must be even)
    Returns:
        PE array of shape (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(0, seq_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                      (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def sinusoidal_pe_broadcast(seq_len: int, d_model: int, batch: int) -> np.ndarray:
    """Compute sinusoidal PE broadcast to (batch, seq_len, d_model)."""
    pe_2d = sinusoidal_pe_np(seq_len, d_model)  # (S, D)
    return np.broadcast_to(pe_2d[np.newaxis, :, :], (batch, seq_len, d_model)).copy()


def linear_projection_np(x: np.ndarray, W: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Numpy reference for linear projection: y = x @ W^T + b.

    Args:
        x: Input tensor (..., K)
        W: Weight matrix (N, K)
        b: Optional bias (N,) or None
    Returns:
        Output tensor (..., N)
    """
    out = x @ W.T.astype(x.dtype)
    if b is not None:
        out = out + b.astype(x.dtype)
    return out.astype(np.float32)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numpy reference for numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return (e_x / np.sum(e_x, axis=axis, keepdims=True)).astype(np.float32)


def scale_np(x: np.ndarray, factor: float) -> np.ndarray:
    """Numpy reference for uniform scaling: y = x * factor."""
    return (x * factor).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Helper: build net from prototxt string
# ═══════════════════════════════════════════════════════════════════════

def _make_net(prototxt: str):
    """Create a net from prototxt string."""
    return net_from_param(net_param_from_string(prototxt))


# ═══════════════════════════════════════════════════════════════════════
# Positional Encoding Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPositionalEncoding:
    """Tests for Positional Encoding using Eltwise SUM and Bias layers."""

    def test_sinusoidal_pe_eltwise_sum(self, ptrace):
        """Sinusoidal PE (pre-broadcast to same shape) added via Eltwise SUM."""
        batch, seq_len, d_model = 2, 8, 16
        # Pre-broadcast PE to match embedding shape since Eltwise requires same shape
        pe = sinusoidal_pe_broadcast(seq_len, d_model, batch)  # (N, S, D)

        prototxt = f"""name: "sinusoidal_pe"
layer {{
  name: "embeddings" type: "Input" top: "embeddings"
  input_param {{ shape {{ dim: {batch} dim: {seq_len} dim: {d_model} }} }}
}}
layer {{
  name: "pe" type: "Input" top: "pe"
  input_param {{ shape {{ dim: {batch} dim: {seq_len} dim: {d_model} }} }}
}}
layer {{
  name: "add_pe" type: "Eltwise" bottom: "embeddings" bottom: "pe" top: "output"
  eltwise_param {{ operation: SUM }}
}}
"""
        with ptrace("Net(sinusoidal PE)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(42)
        embeddings = rng.randn(batch, seq_len, d_model).astype(np.float32)

        with ptrace("sinusoidal PE forward") as t:
            out = net.forward({"embeddings": embeddings, "pe": pe})
            t['shape'] = f"emb={embeddings.shape} pe={pe.shape}"

        expected = (embeddings + pe).astype(np.float32)
        np.testing.assert_allclose(out["output"], expected, rtol=1e-5, atol=1e-5)

    def test_learnable_pe_bias_layer(self, ptrace):
        """Learnable positional encoding via Bias layer (axis=1, 2D bias).

        Bias layer with axis=1, num_axes=2 adds a learned bias of shape (S, D)
        that broadcasts over the batch dimension. This implements learnable PE.
        """
        batch, seq_len, d_model = 2, 6, 12

        prototxt = f"""name: "learnable_pe"
layer {{
  name: "embeddings" type: "Input" top: "embeddings"
  input_param {{ shape {{ dim: {batch} dim: {seq_len} dim: {d_model} }} }}
}}
layer {{
  name: "pe_bias" type: "Bias" bottom: "embeddings" top: "output"
  bias_param {{ axis: 1 num_axes: 2 }}
}}
"""
        with ptrace("Net(learnable PE via Bias)"):
            net = _make_net(prototxt)

        # Learned positional encoding: shape (seq_len, d_model)
        rng = np.random.RandomState(123)
        pe_weights = rng.randn(seq_len, d_model).astype(np.float32) * 0.02
        bias_layer = net.layer_by_name("pe_bias")
        with ptrace("load PE bias weights"):
            # Bias layer stores bias as a flat blob matching the bias shape
            bias_layer.blobs[0].from_numpy(pe_weights)

        embeddings = rng.randn(batch, seq_len, d_model).astype(np.float32)

        with ptrace("learnable PE forward") as t:
            out = net.forward({"embeddings": embeddings})
            t['shape'] = f"emb={embeddings.shape} bias={pe_weights.shape}"

        expected = (embeddings + pe_weights[np.newaxis, :, :]).astype(np.float32)
        np.testing.assert_allclose(out["output"], expected, rtol=1e-5, atol=1e-5)

    def test_pe_addition_2d_flattened(self, ptrace):
        """PE addition in 2D flattened space (M, D) — typical for caffe-ffi usage.

        Real Transformers flatten (batch, seq, d_model) to (batch*seq, d_model)
        for linear layers. PE is pre-broadcast and flattened to match.
        """
        batch, seq_len, d_model = 4, 6, 16
        M = batch * seq_len
        pe = sinusoidal_pe_broadcast(seq_len, d_model, batch).reshape(M, d_model)

        prototxt = f"""name: "pe_2d"
layer {{
  name: "emb" type: "Input" top: "emb"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "pe" type: "Input" top: "pe"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "add" type: "Eltwise" bottom: "emb" bottom: "pe" top: "output"
  eltwise_param {{ operation: SUM }}
}}
"""
        with ptrace("Net(PE 2D flattened)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(55)
        emb = rng.randn(M, d_model).astype(np.float32)

        with ptrace("PE 2D forward"):
            out = net.forward({"emb": emb, "pe": pe})

        expected = (emb + pe).astype(np.float32)
        np.testing.assert_allclose(out["output"], expected, rtol=1e-5, atol=1e-5)

    def test_pe_repeated_forward_deterministic(self, ptrace):
        """PE addition is deterministic across repeated forwards."""
        batch, seq_len, d_model = 3, 10, 16
        M = batch * seq_len
        pe = sinusoidal_pe_broadcast(seq_len, d_model, batch).reshape(M, d_model)

        prototxt = f"""name: "pe_det"
layer {{
  name: "emb" type: "Input" top: "emb"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "pe" type: "Input" top: "pe"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "add" type: "Eltwise" bottom: "emb" bottom: "pe" top: "out"
  eltwise_param {{ operation: SUM }}
}}
"""
        with ptrace("Net(PE deterministic)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(99)
        emb = rng.randn(M, d_model).astype(np.float32)
        results = []
        for i in range(5):
            with ptrace(f"PE forward #{i+1}"):
                out = net.forward({"emb": emb, "pe": pe})
            results.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(results[0], results[i])


# ═══════════════════════════════════════════════════════════════════════
# Self-Attention Component Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSelfAttentionComponents:
    """Tests for individual Self-Attention building blocks (2D operations)."""

    def test_qkv_linear_projection_dimensions(self, ptrace):
        """Q/K/V linear projections produce correct output dimensions and values.

        Uses explicit Split layer to fan out input to Q, K, V projections
        (caffe-ffi requires explicit Split for multi-consumer blobs).
        """
        M, d_model, d_k = 24, 16, 8
        d_v = 8

        prototxt = f"""name: "qkv_proj"
layer {{
  name: "input" type: "Input" top: "input"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "split" type: "Split" bottom: "input" top: "q_in" top: "k_in" top: "v_in"
}}
layer {{
  name: "q_proj" type: "InnerProduct" bottom: "q_in" top: "Q"
  inner_product_param {{ num_output: {d_k} bias_term: false }}
}}
layer {{
  name: "k_proj" type: "InnerProduct" bottom: "k_in" top: "K"
  inner_product_param {{ num_output: {d_k} bias_term: false }}
}}
layer {{
  name: "v_proj" type: "InnerProduct" bottom: "v_in" top: "V"
  inner_product_param {{ num_output: {d_v} bias_term: false }}
}}
"""
        with ptrace("Net(QKV projection)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(42)
        Wq = rng.randn(d_k, d_model).astype(np.float32) * 0.1
        Wk = rng.randn(d_k, d_model).astype(np.float32) * 0.1
        Wv = rng.randn(d_v, d_model).astype(np.float32) * 0.1

        net.layer_by_name("q_proj").blobs[0].from_numpy(Wq)
        net.layer_by_name("k_proj").blobs[0].from_numpy(Wk)
        net.layer_by_name("v_proj").blobs[0].from_numpy(Wv)

        x = rng.randn(M, d_model).astype(np.float32)

        with ptrace("QKV projection forward") as t:
            out = net.forward({"input": x})
            t['shape_Q'] = f"{out['Q'].shape}"
            t['shape_K'] = f"{out['K'].shape}"
            t['shape_V'] = f"{out['V'].shape}"

        assert out["Q"].shape == (M, d_k), f"Q shape mismatch: {out['Q'].shape}"
        assert out["K"].shape == (M, d_k), f"K shape mismatch: {out['K'].shape}"
        assert out["V"].shape == (M, d_v), f"V shape mismatch: {out['V'].shape}"

        np.testing.assert_allclose(out["Q"], linear_projection_np(x, Wq), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out["K"], linear_projection_np(x, Wk), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out["V"], linear_projection_np(x, Wv), rtol=1e-5, atol=1e-5)

    def test_attention_scale_factor(self, ptrace):
        """Scale layer correctly applies 1/sqrt(d_k) factor to attention scores."""
        M, S = 6, 4
        d_k = 16
        scale_factor = 1.0 / np.sqrt(d_k)

        prototxt = f"""name: "attn_scale"
layer {{
  name: "scores" type: "Input" top: "scores"
  input_param {{ shape {{ dim: {M} dim: {S} }} }}
}}
layer {{
  name: "scale" type: "Scale" bottom: "scores" top: "scaled"
  scale_param {{ bias_term: false }}
}}
"""
        with ptrace("Net(attention scale)"):
            net = _make_net(prototxt)

        # Scale on 2D (M, S): axis=1 (default), scale shape = (S,)
        # Fill all channels with same factor for uniform scaling
        scale_weights = np.full(S, scale_factor, dtype=np.float32)
        net.layer_by_name("scale").blobs[0].from_numpy(scale_weights)

        rng = np.random.RandomState(7)
        scores = rng.randn(M, S).astype(np.float32)

        with ptrace("attention scale forward") as t:
            out = net.forward({"scores": scores})
            t['scale_factor'] = f"{scale_factor:.4f}"

        expected = scale_np(scores, scale_factor)
        np.testing.assert_allclose(out["scaled"], expected, rtol=1e-5, atol=1e-5)

    def test_softmax_attention_weights(self, ptrace):
        """Softmax on axis=1 normalizes attention scores to valid probability distributions."""
        M, S = 8, 6

        prototxt = f"""name: "attn_softmax"
layer {{
  name: "scores" type: "Input" top: "scores"
  input_param {{ shape {{ dim: {M} dim: {S} }} }}
}}
layer {{
  name: "softmax" type: "Softmax" bottom: "scores" top: "weights"
  softmax_param {{ axis: 1 }}
}}
"""
        with ptrace("Net(attention softmax)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(55)
        scores = rng.randn(M, S).astype(np.float32) * 3

        with ptrace("attention softmax forward") as t:
            out = net.forward({"scores": scores})
            t['shape'] = f"{out['weights'].shape}"

        weights = out["weights"]
        row_sums = np.sum(weights, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(M), rtol=1e-5, atol=1e-5)
        assert np.all(weights >= -1e-7), f"Negative softmax outputs: min={weights.min()}"
        expected = softmax_np(scores, axis=1)
        np.testing.assert_allclose(weights, expected, rtol=1e-5, atol=1e-5)

    def test_residual_connection_split_eltwise(self, ptrace):
        """Residual connection: x + sublayer(x) via explicit Split + Eltwise SUM."""
        M, D = 10, 16

        prototxt = f"""name: "residual"
layer {{
  name: "input" type: "Input" top: "input"
  input_param {{ shape {{ dim: {M} dim: {D} }} }}
}}
layer {{
  name: "split" type: "Split" bottom: "input" top: "identity" top: "sublayer_in"
}}
layer {{
  name: "sublayer" type: "InnerProduct" bottom: "sublayer_in" top: "sublayer_out"
  inner_product_param {{ num_output: {D} bias_term: true }}
}}
layer {{
  name: "residual_add" type: "Eltwise" bottom: "identity" bottom: "sublayer_out" top: "output"
  eltwise_param {{ operation: SUM }}
}}
"""
        with ptrace("Net(residual connection)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(77)
        # Sublayer: near-identity transform (W ≈ I, small b)
        W = np.eye(D, dtype=np.float32) + rng.randn(D, D).astype(np.float32) * 0.01
        b = rng.randn(D).astype(np.float32) * 0.01
        net.layer_by_name("sublayer").blobs[0].from_numpy(W)
        net.layer_by_name("sublayer").blobs[1].from_numpy(b)

        x = rng.randn(M, D).astype(np.float32)

        with ptrace("residual forward") as t:
            out = net.forward({"input": x})
            t['shape'] = f"{x.shape}"

        expected = (x + linear_projection_np(x, W, b)).astype(np.float32)
        np.testing.assert_allclose(out["output"], expected, rtol=1e-4, atol=1e-4)

    def test_attention_output_via_innerproduct(self, ptrace):
        """Attention-weighted sum (weights @ V) via InnerProduct with V as weights.

        InnerProduct computes y = x @ W^T. If x is attention weights (M, S)
        and W = V^T (D, S), then y = weights @ V of shape (M, D).
        """
        M, S, D = 6, 4, 8

        prototxt = f"""name: "attn_output"
layer {{
  name: "weights" type: "Input" top: "weights"
  input_param {{ shape {{ dim: {M} dim: {S} }} }}
}}
layer {{
  name: "attn_v_proj" type: "InnerProduct" bottom: "weights" top: "output"
  inner_product_param {{ num_output: {D} bias_term: false }}
}}
"""
        with ptrace("Net(attn output via InnerProduct)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(33)
        V = rng.randn(S, D).astype(np.float32) * 0.5
        W = V.T.copy()
        net.layer_by_name("attn_v_proj").blobs[0].from_numpy(W)

        raw_weights = rng.randn(M, S).astype(np.float32)
        attn_weights = softmax_np(raw_weights, axis=1)

        with ptrace("attn weighted sum forward") as t:
            out = net.forward({"weights": attn_weights})
            t['shape'] = f"w={attn_weights.shape} V={V.shape}"

        expected = (attn_weights @ V).astype(np.float32)
        np.testing.assert_allclose(out["output"], expected, rtol=1e-5, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Scaled Dot-Product Attention (End-to-End)
# ═══════════════════════════════════════════════════════════════════════

class TestScaledDotProductAttention:
    """End-to-end Scaled Dot-Product Attention tests.

    Pipeline: scores -> Scale(1/sqrt(d_k)) -> Softmax(axis=1) -> InnerProduct(V)
    Q*K^T is pre-computed in numpy (no native MatMul layer), but the rest
    of the attention pipeline exercises caffe-ffi layers end-to-end.
    """

    def test_sdp_attention_pipeline(self, ptrace):
        """Full SDP Attention pipeline: Scale -> Softmax -> WeightedSum."""
        M, S, d_k, d_v = 12, 6, 16, 8
        scale_factor = 1.0 / np.sqrt(d_k)

        prototxt = f"""name: "sdp_attn"
layer {{
  name: "scores" type: "Input" top: "scores"
  input_param {{ shape {{ dim: {M} dim: {S} }} }}
}}
layer {{
  name: "scale" type: "Scale" bottom: "scores" top: "scaled"
  scale_param {{ bias_term: false }}
}}
layer {{
  name: "softmax" type: "Softmax" bottom: "scaled" top: "attn_weights"
  softmax_param {{ axis: 1 }}
}}
layer {{
  name: "attn_out" type: "InnerProduct" bottom: "attn_weights" top: "output"
  inner_product_param {{ num_output: {d_v} bias_term: false }}
}}
"""
        with ptrace("Net(SDP Attention)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(101)
        scale_w = np.full(S, scale_factor, dtype=np.float32)
        net.layer_by_name("scale").blobs[0].from_numpy(scale_w)

        V = rng.randn(S, d_v).astype(np.float32) * 0.5
        net.layer_by_name("attn_out").blobs[0].from_numpy(V.T.copy())

        raw_scores = rng.randn(M, S).astype(np.float32) * d_k**0.5

        with ptrace("SDP attention forward") as t:
            out = net.forward({"scores": raw_scores})
            t['shape'] = f"scores={raw_scores.shape} V={V.shape}"
            t['d_k'] = d_k

        attn_weights = softmax_np(raw_scores * scale_factor, axis=1)
        expected = (attn_weights @ V).astype(np.float32)
        np.testing.assert_allclose(out["output"], expected, rtol=1e-4, atol=1e-4)
        assert np.all(np.isfinite(out["output"])), "Non-finite outputs detected"

    def test_sdp_attention_identity_case(self, ptrace):
        """Attention with identity V returns the attention weights themselves."""
        M, S = 4, 3
        d_k = 4
        d_v = S
        scale_factor = 1.0 / np.sqrt(d_k)

        prototxt = f"""name: "sdp_identity"
layer {{
  name: "scores" type: "Input" top: "scores"
  input_param {{ shape {{ dim: {M} dim: {S} }} }}
}}
layer {{
  name: "scale" type: "Scale" bottom: "scores" top: "scaled"
  scale_param {{ bias_term: false }}
}}
layer {{
  name: "softmax" type: "Softmax" bottom: "scaled" top: "attn_weights"
  softmax_param {{ axis: 1 }}
}}
layer {{
  name: "attn_out" type: "InnerProduct" bottom: "attn_weights" top: "output"
  inner_product_param {{ num_output: {d_v} bias_term: false }}
}}
"""
        with ptrace("Net(SDP identity V)"):
            net = _make_net(prototxt)

        scale_w = np.full(S, scale_factor, dtype=np.float32)
        net.layer_by_name("scale").blobs[0].from_numpy(scale_w)

        V = np.eye(S, dtype=np.float32)
        net.layer_by_name("attn_out").blobs[0].from_numpy(V.T.copy())

        scores = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)

        with ptrace("SDP identity forward"):
            out = net.forward({"scores": scores})

        expected_weights = softmax_np(scores * scale_factor, axis=1)
        expected_output = (expected_weights @ V).astype(np.float32)

        np.testing.assert_allclose(out["output"], expected_output, rtol=1e-5, atol=1e-5)
        # First row: one-hot at position 0 should have dominant weight
        assert out["output"][0, 0] > out["output"][0, 1]
        assert out["output"][0, 0] > out["output"][0, 2]


# ═══════════════════════════════════════════════════════════════════════
# Multi-Head Projection Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMultiHeadProjection:
    """Tests for multi-head attention via multiple InnerProduct + Concat."""

    def test_multi_head_concat(self, ptrace):
        """Multiple per-head InnerProduct projections concatenated along axis=1.

        Uses explicit Split to fan out input to each head projection.
        """
        M, d_model = 6, 16
        num_heads = 4
        d_k = d_model // num_heads  # 4 per head

        # Generate split tops for num_heads consumers
        split_tops = " ".join(f'top: "head{i+1}_in"' for i in range(num_heads))
        head_layers = []
        for i in range(num_heads):
            head_layers.append(f"""layer {{
  name: "head{i+1}" type: "InnerProduct" bottom: "head{i+1}_in" top: "head{i+1}"
  inner_product_param {{ num_output: {d_k} bias_term: false }}
}}""")
        heads_str = "\n".join(head_layers)
        concat_bottoms = "\n".join(f'  bottom: "head{i+1}"' for i in range(num_heads))

        prototxt = f"""name: "multihead"
layer {{
  name: "input" type: "Input" top: "input"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "split" type: "Split" bottom: "input" {split_tops}
}}
{heads_str}
layer {{
  name: "concat" type: "Concat"
{concat_bottoms}
  top: "output"
  concat_param {{ axis: 1 }}
}}
"""
        with ptrace("Net(multi-head concat)"):
            net = _make_net(prototxt)

        rng = np.random.RandomState(202)
        x = rng.randn(M, d_model).astype(np.float32)

        head_weights = []
        for i in range(num_heads):
            W = rng.randn(d_k, d_model).astype(np.float32) * 0.1
            net.layer_by_name(f"head{i+1}").blobs[0].from_numpy(W)
            head_weights.append(W)

        with ptrace("multi-head forward") as t:
            out = net.forward({"input": x})
            t['num_heads'] = num_heads
            t['output_shape'] = f"{out['output'].shape}"

        assert out["output"].shape == (M, d_model), \
            f"Concat shape: {out['output'].shape} != ({M}, {d_model})"

        for i in range(num_heads):
            expected_head = linear_projection_np(x, head_weights[i])
            actual_head = out["output"][:, i*d_k:(i+1)*d_k]
            np.testing.assert_allclose(actual_head, expected_head, rtol=1e-5, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Transformer Encoder Block Integration Test
# ═══════════════════════════════════════════════════════════════════════

class TestTransformerEncoderBlock:
    """Integration test for a simplified Transformer encoder block (2D flattened).

    Pipeline: emb + PE -> implicit split -> (scores -> Scale -> Softmax -> IP)
                                          -> Eltwise SUM (residual) -> output
    """

    def test_encoder_block_forward(self, ptrace):
        """Simplified encoder block: PE + Attention + residual produces correct output."""
        batch, seq_len, d_model = 2, 4, 8
        M = batch * seq_len
        scale_factor = 1.0 / np.sqrt(d_model)

        # Pre-compute PE and flatten
        pe = sinusoidal_pe_broadcast(seq_len, d_model, batch).reshape(M, d_model)

        prototxt = f"""name: "encoder_block"
layer {{
  name: "emb" type: "Input" top: "emb"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "pe" type: "Input" top: "pe"
  input_param {{ shape {{ dim: {M} dim: {d_model} }} }}
}}
layer {{
  name: "add_pe" type: "Eltwise" bottom: "emb" bottom: "pe" top: "x"
  eltwise_param {{ operation: SUM }}
}}
layer {{
  name: "scores" type: "Input" top: "scores"
  input_param {{ shape {{ dim: {M} dim: {seq_len} }} }}
}}
layer {{
  name: "scale" type: "Scale" bottom: "scores" top: "scaled"
  scale_param {{ bias_term: false }}
}}
layer {{
  name: "softmax" type: "Softmax" bottom: "scaled" top: "attn_w"
  softmax_param {{ axis: 1 }}
}}
layer {{
  name: "attn_proj" type: "InnerProduct" bottom: "attn_w" top: "attn_out"
  inner_product_param {{ num_output: {d_model} bias_term: false }}
}}
layer {{
  name: "res_add" type: "Eltwise" bottom: "x" bottom: "attn_out" top: "output"
  eltwise_param {{ operation: SUM }}
}}
"""
        rng = np.random.RandomState(42)

        with ptrace("Net(encoder block)"):
            net = _make_net(prototxt)

        net.layer_by_name("scale").blobs[0].from_numpy(
            np.full(seq_len, scale_factor, dtype=np.float32))

        attn_W = rng.randn(d_model, seq_len).astype(np.float32) * 0.1
        net.layer_by_name("attn_proj").blobs[0].from_numpy(attn_W)

        emb = rng.randn(M, d_model).astype(np.float32) * 0.1
        scores = rng.randn(M, seq_len).astype(np.float32)

        with ptrace("encoder block forward") as t:
            out = net.forward({"emb": emb, "pe": pe, "scores": scores})
            t['shape'] = f"{out['output'].shape}"
            t['layers'] = len(net.layers_array())

        assert out["output"].shape == (M, d_model), \
            f"Output shape: {out['output'].shape} != ({M}, {d_model})"
        assert np.all(np.isfinite(out["output"])), "Non-finite output in encoder block"

        x = (emb + pe).astype(np.float32)
        attn_w = softmax_np(scores * scale_factor, axis=1)
        attn_out = (attn_w @ attn_W.T).astype(np.float32)
        expected = (x + attn_out).astype(np.float32)
        np.testing.assert_allclose(out["output"], expected, rtol=1e-4, atol=1e-4)
