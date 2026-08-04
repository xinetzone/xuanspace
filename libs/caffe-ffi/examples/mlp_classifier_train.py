"""MLP classifier training example — demonstrates the P4 training-engineering API.

Shows the full training workflow built on ``caffe_ffi.solver`` and
``caffe_ffi.serialization`` (Task 33):

1. Build a small MLP + SoftmaxWithLoss from a prototxt string.
2. Train it with :class:`Solver` + :class:`SGD` (momentum) on synthetic data.
3. Snapshot the weights to a ``.caffemodel`` with :func:`save_net`.
4. Load a fresh network from the saved model with :func:`load_net`.
5. Evaluate the reloaded model on the test set.

Usage:
    conda run -n py314 python examples/mlp_classifier_train.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

import caffe_ffi  # noqa: E402
from caffe_ffi import net_from_param, net_param_from_string  # noqa: E402
from caffe_ffi.solver import SGD, StepLR, Solver  # noqa: E402
from caffe_ffi.serialization import save_net, load_net  # noqa: E402

# ---------------------------------------------------------------------------
# Network + synthetic data
# ---------------------------------------------------------------------------

MLP_PROTO = '''
name: "mlp_classifier"
layer { name: "data" type: "Input" top: "data"
  input_param { shape { dim: 32 dim: 16 } } }
layer { name: "label" type: "Input" top: "label"
  input_param { shape { dim: 32 dim: 1 } } }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1"
  inner_product_param { num_output: 32 weight_filler { type: "msra" } } }
layer { name: "relu1" type: "ReLU" bottom: "fc1" top: "fc1" }
layer { name: "fc2" type: "InnerProduct" bottom: "fc1" top: "fc2"
  inner_product_param { num_output: 4 weight_filler { type: "msra" } } }
layer { name: "loss" type: "SoftmaxWithLoss" bottom: "fc2" bottom: "label" top: "loss" }
layer { name: "accuracy" type: "Accuracy" bottom: "fc2" bottom: "label" top: "accuracy" }
'''

N_CLASSES = 4
BATCH = 32


def make_net():
    return net_from_param(net_param_from_string(MLP_PROTO))


def make_batches(rng, n_batches, seed_data):
    """Yield ``n_batches`` mini-batches of synthetic class patterns."""
    for _ in range(n_batches):
        x = rng.randn(BATCH, 16).astype(np.float32)
        y = rng.randint(0, N_CLASSES, size=(BATCH, 1)).astype(np.float32)
        # Correlate input with the label so the network can learn.
        x += y.astype(np.float32) * 0.5
        yield {"data": x, "label": y}


def main():
    print("=" * 60)
    print("MLP classifier training — caffe-ffi Solver API")
    print("=" * 60)
    print(f"FFI available: {caffe_ffi.is_available()}")
    if not caffe_ffi.is_available():
        print("ERROR: C++ extension required. Install with `pip install -e .`.")
        return

    # 1. Build network
    net = make_net()
    print(f"Network: {net}")

    # 2. Train with Solver + SGD(momentum) + StepLR scheduler
    optimizer = SGD(lr=0.05, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    solver = Solver(net, optimizer, loss_blob="loss", metric_blob="accuracy", scheduler=scheduler)

    rng = np.random.RandomState(0)
    history = solver.fit(
        train_batches=lambda: make_batches(rng, 20, None),
        epochs=4,
        do_validate=True,
        val_batches=list(make_batches(np.random.RandomState(1), 5, None)),
        log_interval=10,
    )
    print(f"\nFinal loss: {history['loss'][-1]:.4f}  (initial {history['loss'][0]:.4f})")

    # 3. Save the trained model
    model_path = Path(tempfile.gettempdir()) / "caffe_ffi_mlp_classifier.caffemodel"
    save_net(net, model_path)
    print(f"Saved model -> {model_path}")

    # 4. Load into a fresh network
    fresh = make_net()
    load_net(fresh, model_path)
    np.testing.assert_allclose(
        fresh.layer_by_name("fc1").blobs[0].data,
        net.layer_by_name("fc1").blobs[0].data,
        rtol=1e-6,
    )
    print("Reloaded model matches original weights.")

    # 5. Evaluate the reloaded model
    rng = np.random.RandomState(2)
    accs = []
    for batch in make_batches(rng, 5, None):
        out = fresh.forward(batch)
        accs.append(float(out["accuracy"].flat[0]))
    print(f"Reloaded model accuracy: {np.mean(accs):.2%}")

    print("\nDone.")


if __name__ == "__main__":
    main()