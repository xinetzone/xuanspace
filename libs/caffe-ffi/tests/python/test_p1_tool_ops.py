"""P1 unit tests for the caffe-ffi "tool / post-processing" layers.

Covers ArgMax, BatchReindex, Filter, Parameter, Silence, SPP and Embed.
Each operator's forward pass is validated against a numpy reference
implementation (rtol=1e-3 / atol=1e-4), with output-shape assertions and
learnable-blob-count checks where applicable.

Reference semantics were derived from the C++ implementations in
``src/caffe_ffi/layers/*.cpp``.
"""

from __future__ import annotations

import textwrap

import numpy as np

from .conftest import require_cpp_extension
from .caffe_test_helpers import make_net


# ──────────────────────────────────────────────────────────────────────
# Shared numpy reference helpers
# ──────────────────────────────────────────────────────────────────────

def topk_indices_desc(vals_1d: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest entries of a 1-D array.

    Matches ``std::partial_sort(..., std::greater<pair<float,int>>)`` in
    ArgMaxLayer: values sorted descending, ties broken by the LARGER index
    first (because the pair's index is compared as the secondary key).
    """
    order = np.lexsort((-np.arange(len(vals_1d)), -vals_1d))
    return order[:k]


def argmax_axis_reference(x: np.ndarray, k: int, out_max_val: bool) -> np.ndarray:
    """ArgMax with ``axis=1`` over a 3-D ``[N, C, H]`` input.

    Overlaps the channel axis (axis=1) at each spatial position ``(n, h)``,
    producing an output of shape ``[N, k, H]``.
    """
    n, c, h = x.shape
    out = np.zeros((n, k, h), dtype=np.float32)
    for i in range(n):
        for j in range(h):
            vals = x[i, :, j]
            idx = topk_indices_desc(vals, k)
            if out_max_val:
                out[i, :, j] = vals[idx]
            else:
                out[i, :, j] = idx
    return out


def argmax_noaxis_reference(x: np.ndarray, k: int, out_max_val: bool) -> np.ndarray:
    """ArgMax without ``axis``: flattened per-instance max over count(1)."""
    n = x.shape[0]
    if out_max_val:
        out = np.zeros((n, 2, k), dtype=np.float32)
        for i in range(n):
            idx = topk_indices_desc(x[i], k)
            out[i, 0, :] = idx
            out[i, 1, :] = x[i][idx]
    else:
        out = np.zeros((n, 1, k), dtype=np.float32)
        for i in range(n):
            out[i, 0, :] = topk_indices_desc(x[i], k)
    return out


def spp_reference(x: np.ndarray, pyramid_height: int, pool: str) -> np.ndarray:
    """SPP forward reference (matches SPPLayer::Forward_cpu).

    ``pool`` is "MAX" or "AVE". AVE divides the window sum by the FULL kernel
    area ``kh*kw`` (caffe semantics, including padded zeros).
    """
    n, c, h, w = x.shape
    outputs = []
    for p in range(pyramid_height):
        nb = 1 << p
        kh = int(np.ceil(h / nb))
        ph = (kh * nb - h + 1) // 2
        kw = int(np.ceil(w / nb))
        pw = (kw * nb - w + 1) // 2
        pooled = np.zeros((n, c, nb, nb), dtype=np.float32)
        for ib in range(nb):
            h_start = ib * kh - ph
            for jb in range(nb):
                w_start = jb * kw - pw
                # Slice with python auto-clamping (valid region only).
                win = x[:, :, max(h_start, 0):h_start + kh,
                       max(w_start, 0):w_start + kw]
                if pool == "MAX":
                    pooled[:, :, ib, jb] = win.max(axis=(2, 3))
                else:
                    pooled[:, :, ib, jb] = win.sum(axis=(2, 3)) / (kh * kw)
        outputs.append(pooled.reshape(n, c * nb * nb))
    return np.concatenate(outputs, axis=1).reshape(n, -1, 1, 1)


# ──────────────────────────────────────────────────────────────────────
# ArgMax
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestArgMax:
    def _net(self, top_k, out_max_val, axis=None, input_shape=(3, 5, 4)):
        axis_txt = ""
        if axis is not None:
            axis_txt = f"axis: {axis}"
        dims_txt = " ".join(f"dim: {d}" for d in input_shape)
        return make_net(textwrap.dedent(f"""\
            name: "argmax_test"
            layer {{
              name: "data"
              type: "Input"
              top: "data"
              input_param {{ shape {{ {dims_txt} }} }}
            }}
            layer {{
              name: "argmax"
              type: "ArgMax"
              bottom: "data"
              top: "argmax_out"
              argmax_param {{
                top_k: {top_k}
                out_max_val: {str(out_max_val).lower()}
                {axis_txt}
              }}
            }}
        """))

    def test_axis_mode_indices(self):
        x = np.array([[[1, 5, 3, 2],
                       [4, 9, 1, 4],
                       [8, 7, 0, 6],
                       [2, 5, 3, 1],
                       [0, 2, 4, 6]],
                      [[9, 1, 4, 8],
                       [7, 0, 6, 2],
                       [5, 3, 1, 0],
                       [2, 4, 6, 8],
                       [1, 3, 5, 7]],
                      [[0, 6, 2, 5],
                       [3, 1, 0, 2],
                       [4, 6, 8, 1],
                       [3, 5, 7, 9],
                       [2, 4, 6, 8]]], dtype=np.float32)
        net = self._net(top_k=2, out_max_val=False, axis=1)
        out = net.forward({"data": x})
        expected = argmax_axis_reference(x, 2, out_max_val=False)
        assert out["argmax_out"].shape == (3, 2, 4)
        np.testing.assert_array_equal(out["argmax_out"], expected)

    def test_axis_mode_values(self):
        x = np.array([[[1, 5, 3, 2],
                       [4, 9, 1, 4],
                       [8, 7, 0, 6],
                       [2, 5, 3, 1],
                       [0, 2, 4, 6]],
                      [[9, 1, 4, 8],
                       [7, 0, 6, 2],
                       [5, 3, 1, 0],
                       [2, 4, 6, 8],
                       [1, 3, 5, 7]],
                      [[0, 6, 2, 5],
                       [3, 1, 0, 2],
                       [4, 6, 8, 1],
                       [3, 5, 7, 9],
                       [2, 4, 6, 8]]], dtype=np.float32)
        net = self._net(top_k=2, out_max_val=True, axis=1)
        out = net.forward({"data": x})
        expected = argmax_axis_reference(x, 2, out_max_val=True)
        assert out["argmax_out"].shape == (3, 2, 4)
        np.testing.assert_allclose(out["argmax_out"], expected, rtol=1e-3, atol=1e-4)

    def test_noaxis_mode_values_and_indices(self):
        # Without axis, out_max_val=true interleaves indices then values: [N, 2, top_k].
        x = np.array([[1, 5, 3, 2, 4],
                      [9, 1, 4, 8, 7],
                      [0, 6, 2, 5, 3]], dtype=np.float32)
        net = self._net(top_k=2, out_max_val=True, input_shape=(3, 5))
        out = net.forward({"data": x})
        expected = argmax_noaxis_reference(x, 2, out_max_val=True)
        assert out["argmax_out"].shape == (3, 2, 2)
        np.testing.assert_array_equal(out["argmax_out"], expected)

    def test_noaxis_mode_indices_only(self):
        x = np.array([[1, 5, 3, 2, 4],
                      [9, 1, 4, 8, 7],
                      [0, 6, 2, 5, 3]], dtype=np.float32)
        net = self._net(top_k=2, out_max_val=False, input_shape=(3, 5))
        out = net.forward({"data": x})
        expected = argmax_noaxis_reference(x, 2, out_max_val=False)
        assert out["argmax_out"].shape == (3, 1, 2)
        np.testing.assert_array_equal(out["argmax_out"], expected)

    def test_tie_breaking_higher_index_first(self):
        # Equal values: std::greater<pair<value,index>> picks the larger index first.
        x = np.array([[[5],
                       [5],
                       [3],
                       [1],
                       [5]]], dtype=np.float32)  # [1, 5, 1]
        net = self._net(top_k=2, out_max_val=False, axis=1)
        out = net.forward({"data": x})
        # top two values are 5 (indices 0,1,4) -> larger indices 4 and 1 win.
        np.testing.assert_array_equal(out["argmax_out"],
                                      np.array([[[4], [1]]], dtype=np.float32))

    def test_layer_registered(self):
        net = self._net(top_k=1, out_max_val=False, axis=1)
        assert "argmax" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# BatchReindex
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestBatchReindex:
    def _net(self, data_batch, index_len):
        return make_net(textwrap.dedent(f"""\
            name: "batch_reindex_test"
            layer {{
              name: "data"
              type: "Input"
              top: "data"
              input_param {{ shape {{ dim: {data_batch} dim: 3 }} }}
            }}
            layer {{
              name: "index"
              type: "Input"
              top: "index"
              input_param {{ shape {{ dim: {index_len} }} }}
            }}
            layer {{
              name: "reindex"
              type: "BatchReindex"
              bottom: "data"
              bottom: "index"
              top: "reindex_out"
            }}
        """))

    def test_forward_reorders_batch(self):
        data = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]],
                        dtype=np.float32)
        perm = np.array([3, 0, 2], dtype=np.float32)
        net = self._net(data_batch=4, index_len=3)
        out = net.forward({"data": data, "index": perm})
        expected = data[perm.astype(int)]
        assert out["reindex_out"].shape == (3, 3)
        np.testing.assert_allclose(out["reindex_out"], expected, rtol=1e-4, atol=1e-4)

    def test_output_shape_follows_index_count(self):
        data = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        perm = np.array([1, 1, 0, 3, 2, 1], dtype=np.float32)  # 6 indices -> batch 6
        net = self._net(data_batch=4, index_len=6)
        out = net.forward({"data": data, "index": perm})
        assert out["reindex_out"].shape == (6, 3)
        np.testing.assert_allclose(out["reindex_out"], data[perm.astype(int)],
                                   rtol=1e-4, atol=1e-4)

    def test_layer_registered(self):
        net = self._net(data_batch=3, index_len=2)
        assert "reindex" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# Filter
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestFilter:
    def _net(self, batch, feat, extra_data=False):
        extra = ""
        if extra_data:
            extra = (
                'layer { name: "data2" type: "Input" top: "data2" '
                f'input_param {{ shape {{ dim: {batch} dim: {feat} }} }} }}\n'
                'layer { name: "filter" type: "Filter" bottom: "data" bottom: "data2" '
                'bottom: "selector" top: "filter_out" top: "filter_out2" }\n'
            )
        else:
            extra = (
                'layer { name: "filter" type: "Filter" bottom: "data" '
                'bottom: "selector" top: "filter_out" }\n'
            )
        return make_net(textwrap.dedent(f"""\
            name: "filter_test"
            layer {{
              name: "data"
              type: "Input"
              top: "data"
              input_param {{ shape {{ dim: {batch} dim: {feat} }} }}
            }}
            layer {{
              name: "selector"
              type: "Input"
              top: "selector"
              input_param {{ shape {{ dim: {batch} }} }}
            }}
            {extra}
        """))

    def test_forward_gathers_kept_items(self):
        data = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
        selector = np.array([1, 0, 1, 0, 1], dtype=np.float32)
        net = self._net(batch=5, feat=3)
        out = net.forward({"data": data, "selector": selector})
        expected = data[[0, 2, 4]]
        assert out["filter_out"].shape == (3, 3)
        np.testing.assert_allclose(out["filter_out"], expected, rtol=1e-4, atol=1e-4)

    def test_forward_all_kept(self):
        data = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
        selector = np.ones((4,), dtype=np.float32)
        net = self._net(batch=4, feat=2)
        out = net.forward({"data": data, "selector": selector})
        assert out["filter_out"].shape == (4, 2)
        np.testing.assert_allclose(out["filter_out"], data, rtol=1e-4, atol=1e-4)

    def test_forward_multiple_data_bottoms(self):
        data = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
        data2 = data + 100.0
        selector = np.array([1, 0, 1, 0, 1], dtype=np.float32)
        net = self._net(batch=5, feat=2, extra_data=True)
        out = net.forward({"data": data, "data2": data2, "selector": selector})
        assert out["filter_out"].shape == (3, 2)
        assert out["filter_out2"].shape == (3, 2)
        np.testing.assert_allclose(out["filter_out"], data[[0, 2, 4]], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out["filter_out2"], data2[[0, 2, 4]], rtol=1e-4, atol=1e-4)

    def test_layer_registered(self):
        net = self._net(batch=3, feat=2)
        assert "filter" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# Parameter
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestParameter:
    def _net(self, shape_dims):
        dims_txt = " ".join(f"dim: {d}" for d in shape_dims)
        return make_net(textwrap.dedent(f"""\
            name: "param_test"
            layer {{
              name: "param"
              type: "Parameter"
              top: "param_out"
              parameter_param {{ shape {{ {dims_txt} }} }}
            }}
        """))

    def test_forward_exposes_param_blob(self):
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        net = self._net([2, 3])
        layer = net.layer_by_name("param")
        assert len(layer.blobs) == 1
        layer.blobs[0].from_numpy(weights)
        out = net.forward({})
        assert out["param_out"].shape == (2, 3)
        np.testing.assert_allclose(out["param_out"], weights, rtol=1e-4, atol=1e-4)

    def test_blob_shape_and_count(self):
        net = self._net([4, 5])
        layer = net.layer_by_name("param")
        assert len(layer.blobs) == 1
        assert tuple(layer.blobs[0].shape) == (4, 5)
        # Top blob shape matches the parameter blob shape.
        assert tuple(net.blob_by_name("param_out").shape) == (4, 5)

    def test_forward_shared_data(self):
        # Forward shares the parameter blob's data; updating the blob updates output.
        net = self._net([1, 3])
        layer = net.layer_by_name("param")
        layer.blobs[0].from_numpy(np.array([7.0, 8.0, 9.0], dtype=np.float32))
        out1 = net.forward({})
        layer.blobs[0].from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        out2 = net.forward({})
        np.testing.assert_allclose(out1["param_out"], [7.0, 8.0, 9.0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out2["param_out"], [1.0, 2.0, 3.0], rtol=1e-4, atol=1e-4)

    def test_layer_registered(self):
        net = self._net([2, 2])
        assert "param" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# Silence
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestSilence:
    def test_forward_consumes_bottom_no_output(self):
        prototxt = """name: "silence_only"
        layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
        layer { name: "silence" type: "Silence" bottom: "data" }
        """
        net = make_net(textwrap.dedent(prototxt))
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = net.forward({"data": x})  # must not raise
        assert "data" not in out  # Silence suppresses the blob (no top produced)

    def test_forward_does_not_disturb_sibling_layer(self):
        prototxt = """name: "silence_with_relu"
        layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
        layer { name: "relu" type: "ReLU" bottom: "data" top: "relu_out" }
        layer { name: "silence" type: "Silence" bottom: "data" }
        """
        net = make_net(textwrap.dedent(prototxt))
        x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        out = net.forward({"data": x})
        expected = np.maximum(x, 0)
        np.testing.assert_array_equal(out["relu_out"], expected)
        assert "data" not in out

    def test_layer_registered(self):
        prototxt = """name: "silence_reg"
        layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 } } }
        layer { name: "silence" type: "Silence" bottom: "data" }
        """
        net = make_net(textwrap.dedent(prototxt))
        assert "silence" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# SPP
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestSPP:
    def _net(self, pyramid_height, pool, channels=1, n=1, h=5, w=5):
        return make_net(textwrap.dedent(f"""\
            name: "spp_test"
            layer {{
              name: "data"
              type: "Input"
              top: "data"
              input_param {{ shape {{ dim: {n} dim: {channels} dim: {h} dim: {w} }} }}
            }}
            layer {{
              name: "spp"
              type: "SPP"
              bottom: "data"
              top: "spp_out"
              spp_param {{ pyramid_height: {pyramid_height} pool: {pool} }}
            }}
        """))

    def test_ave_values_match_reference(self):
        x = (np.arange(1 * 1 * 5 * 5, dtype=np.float32).reshape(1, 1, 5, 5) + 1) / 10.0
        net = self._net(pyramid_height=2, pool="AVE")
        out = net.forward({"data": x})
        expected = spp_reference(x, 2, "AVE")
        assert out["spp_out"].shape == (1, 5, 1, 1)
        np.testing.assert_allclose(out["spp_out"], expected, rtol=1e-3, atol=1e-4)

    def test_max_values_match_reference(self):
        rng = np.random.RandomState(0)
        x = rng.randn(1, 1, 5, 5).astype(np.float32)
        net = self._net(pyramid_height=2, pool="MAX")
        out = net.forward({"data": x})
        expected = spp_reference(x, 2, "MAX")
        assert out["spp_out"].shape == (1, 5, 1, 1)
        np.testing.assert_allclose(out["spp_out"], expected, rtol=1e-3, atol=1e-4)

    def test_output_shape_sum_of_bins(self):
        # pyramid_height=2 -> levels 1 + 4 = 5 bins; channels=2 -> 10 output channels.
        x = np.zeros((1, 2, 4, 4), dtype=np.float32)
        net = self._net(pyramid_height=2, pool="MAX", channels=2, n=1, h=4, w=4)
        out = net.forward({"data": x})
        assert out["spp_out"].shape == (1, 10, 1, 1)

    def test_pyramid_height3_shape(self):
        # pyramid_height=3 -> 1 + 4 + 16 = 21 bins per channel.
        x = np.zeros((1, 1, 4, 4), dtype=np.float32)
        net = self._net(pyramid_height=3, pool="MAX", channels=1, n=1, h=4, w=4)
        out = net.forward({"data": x})
        assert out["spp_out"].shape == (1, 21, 1, 1)

    def test_layer_registered(self):
        net = self._net(pyramid_height=1, pool="MAX")
        assert "spp" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# Embed
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestEmbed:
    def _net(self, num_output, input_dim, bias_term, input_shape):
        dims_txt = " ".join(f"dim: {d}" for d in input_shape)
        return make_net(textwrap.dedent(f"""\
            name: "embed_test"
            layer {{
              name: "data"
              type: "Input"
              top: "data"
              input_param {{ shape {{ {dims_txt} }} }}
            }}
            layer {{
              name: "embed"
              type: "Embed"
              bottom: "data"
              top: "embed_out"
              embed_param {{
                num_output: {num_output}
                input_dim: {input_dim}
                bias_term: {str(bias_term).lower()}
              }}
            }}
        """))

    def test_forward_with_bias(self):
        K, N = 4, 3
        weight = np.arange(K * N, dtype=np.float32).reshape(K, N)
        bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        idx = np.array([2, 0, 1, 3], dtype=np.float32)
        net = self._net(num_output=N, input_dim=K, bias_term=True, input_shape=[4])
        layer = net.layer_by_name("embed")
        assert len(layer.blobs) == 2  # weight + bias
        layer.blobs[0].from_numpy(weight)
        layer.blobs[1].from_numpy(bias)
        out = net.forward({"data": idx})
        expected = weight[idx.astype(int)] + bias
        assert out["embed_out"].shape == (4, N)
        np.testing.assert_allclose(out["embed_out"], expected, rtol=1e-3, atol=1e-4)

    def test_no_bias_single_blob(self):
        K, N = 3, 2
        weight = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        idx = np.array([2, 0, 1], dtype=np.float32)
        net = self._net(num_output=N, input_dim=K, bias_term=False, input_shape=[3])
        layer = net.layer_by_name("embed")
        assert len(layer.blobs) == 1  # weight only
        layer.blobs[0].from_numpy(weight)
        out = net.forward({"data": idx})
        expected = weight[idx.astype(int)]
        assert out["embed_out"].shape == (3, N)
        np.testing.assert_allclose(out["embed_out"], expected, rtol=1e-3, atol=1e-4)

    def test_output_shape_appends_embed_dim(self):
        # 2-D index input [2, 3] -> output [2, 3, N].
        K, N = 5, 4
        weight = np.arange(K * N, dtype=np.float32).reshape(K, N)
        idx = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.float32)
        net = self._net(num_output=N, input_dim=K, bias_term=False, input_shape=[2, 3])
        layer = net.layer_by_name("embed")
        layer.blobs[0].from_numpy(weight)
        out = net.forward({"data": idx})
        assert out["embed_out"].shape == (2, 3, N)
        expected = weight[idx.astype(int)]
        np.testing.assert_allclose(out["embed_out"], expected, rtol=1e-3, atol=1e-4)

    def test_layer_registered(self):
        net = self._net(num_output=2, input_dim=3, bias_term=True, input_shape=[2])
        assert "embed" in net.layer_names()