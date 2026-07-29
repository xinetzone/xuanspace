"""
End-to-end MLP example: create a simple MLP network, set weights manually,
run forward pass, and verify results against manual numpy computation.

Network architecture:
    Input (2 samples, 3 features)
    -> InnerProduct (3 -> 4) + ReLU
    -> InnerProduct (4 -> 2) + Softmax
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

import caffe_ffi
from caffe_ffi import caffe_pb2, net_param_from_string, net_from_param, _ffi_api


def create_mlp_prototxt() -> str:
    return """name: "mlp_demo"
input: "data"
input_shape {
  dim: 2
  dim: 3
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "ip1"
  inner_product_param {
    num_output: 4
    bias_term: true
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  inner_product_param {
    num_output: 2
    bias_term: true
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip2"
  top: "prob"
}
"""


def manual_mlp_forward(x, W1, b1, W2, b2):
    """Manual MLP forward pass using numpy for verification."""
    z1 = x @ W1.T + b1
    h1 = np.maximum(0, z1)
    z2 = h1 @ W2.T + b2
    z2_shifted = z2 - np.max(z2, axis=1, keepdims=True)
    exp_z2 = np.exp(z2_shifted)
    prob = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
    return z1, h1, z2, prob


def main():
    print("=" * 60)
    print("Caffe-FFI MLP Demo")
    print("=" * 60)
    print(f"FFI available: {_ffi_api.is_available()}")
    print()

    W1 = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ], dtype=np.float32)
    b1 = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    W2 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ], dtype=np.float32)
    b2 = np.array([0.001, 0.002], dtype=np.float32)

    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)

    print("Weights W1 (4x3):")
    print(W1)
    print(f"Biases b1: {b1}")
    print()
    print("Weights W2 (2x4):")
    print(W2)
    print(f"Biases b2: {b2}")
    print()
    print(f"Input x (2x3):\n{x}")
    print()

    print("-" * 60)
    print("Manual numpy computation:")
    print("-" * 60)
    z1, h1, z2, prob_manual = manual_mlp_forward(x, W1, b1, W2, b2)
    print(f"z1 = x @ W1.T + b1:\n{z1}")
    print(f"h1 = ReLU(z1):\n{h1}")
    print(f"z2 = h1 @ W2.T + b2:\n{z2}")
    print(f"prob = softmax(z2):\n{prob_manual}")
    print(f"prob sums: {prob_manual.sum(axis=1)}")
    print()

    print("-" * 60)
    print("Building network from prototxt...")
    print("-" * 60)
    prototxt = create_mlp_prototxt()
    param = net_param_from_string(prototxt)
    net = net_from_param(param)
    print(f"Network created: {net}")
    print(f"Blobs: {list(net.blobs_dict.keys())}")
    print(f"Layers: {list(net.layers_dict.keys())}")
    print()

    if _ffi_api.is_available():
        print("-" * 60)
        print("Setting weights via C++ extension...")
        print("-" * 60)
        layers = net.layers_array()
        ip1_layer = layers[0]
        ip2_layer = layers[2]
        
        if len(ip1_layer.blobs) >= 2:
            ip1_layer.blobs[0].from_numpy(W1)
            ip1_layer.blobs[1].from_numpy(b1.reshape(-1))
            print("Set W1 and b1 for ip1 layer")
        if len(ip2_layer.blobs) >= 2:
            ip2_layer.blobs[0].from_numpy(W2)
            ip2_layer.blobs[1].from_numpy(b2.reshape(-1))
            print("Set W2 and b2 for ip2 layer")
        print()

        print("-" * 60)
        print("Running forward pass via C++ extension...")
        print("-" * 60)
        out = net.forward({"data": x})
        for name, arr in out.items():
            print(f"Output '{name}':\n{arr}")
            print(f"Shape: {arr.shape}")
        print()

        if "prob" in out:
            print("-" * 60)
            print("Verification: comparing C++ output vs manual numpy")
            print("-" * 60)
            try:
                np.testing.assert_allclose(out["prob"], prob_manual, rtol=1e-5)
                print("✓ Output matches manual computation!")
            except AssertionError as e:
                print(f"✗ Mismatch: {e}")
                print(f"Max diff: {np.max(np.abs(out['prob'] - prob_manual))}")
    else:
        print("-" * 60)
        print("Note: C++ extension not available.")
        print("The Python API is loaded and working, but actual layer")
        print("computation requires the compiled C++ extension.")
        print()
        print("To fully test this example:")
        print("  1. Build caffe-ffi:")
        print("     pip install -e .")
        print("  2. Run this example again")
        print("-" * 60)

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
