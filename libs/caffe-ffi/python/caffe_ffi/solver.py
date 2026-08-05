"""Optimizers, learning-rate schedulers and the training-loop Solver for caffe-ffi.

This module provides the "training engineering" layer of caffe-ffi:

* ``Optimizer`` / ``SGD`` / ``Adam`` — parameter-update rules that read each
  learnable blob's ``diff`` (gradient) and update its ``data`` (weights) in place.
  The update is COW-aware: weights are written through ``mutable_data_tensor()``.
* ``LRScheduler`` and its concrete schedules — adjust ``optimizer.lr`` over epochs.
* ``Solver`` — a small training loop that wraps ``net.forward``/``net.backward``
  and an optimizer, exposing ``step`` (one batch) and ``fit`` (many epochs).

The design mirrors the validated manual loop in ``examples/lenet_mnist_train.py``:
forward → backward (seeding the loss blob's gradient with ``[1.0]``) → weight update.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from ._core import Net
from ._dtype import _as_float32

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "LRScheduler",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "Solver",
]


def _learnable_params(net: Net):
    """Yield ``(layer, blob_index, blob)`` for every learnable parameter blob.

    A "learnable" parameter is any blob owned by a layer with at least one blob
    (weights/biases). Layers without parameter blobs (activations, pooling, ...)
    contribute nothing. The layer name + blob index form a stable key for any
    per-parameter optimizer state.
    """
    for layer in net.layers_array():
        blobs = layer.blobs
        for i, blob in enumerate(blobs):
            yield layer, i, blob


class Optimizer:
    """Base class for parameter-update optimizers.

    Parameters
    ----------
    lr : float
        Base learning rate.
    weight_decay : float
        L2 weight-decay coefficient applied to the gradient as
        ``grad = diff + weight_decay * data`` before the update.
    """

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def _grad(self, blob) -> np.ndarray:
        """Return the effective gradient (with optional L2 weight decay)."""
        if self.weight_decay:
            return blob.diff_tensor + self.weight_decay * blob.data_tensor
        return blob.diff_tensor

    def step(self, net: Net) -> None:
        """Update all learnable parameter blobs of ``net`` using accumulated gradients.

        Subclasses must implement :meth:`_update_param`.
        """
        raise NotImplementedError

    def _update_param(self, blob, key, grad: np.ndarray) -> None:
        """Apply one optimizer step to a single parameter blob."""
        raise NotImplementedError

    def zero_grad(self, net: Net) -> None:
        """Zero the diff of every learnable blob (clears accumulated gradients)."""
        for _layer, _i, blob in _learnable_params(net):
            blob.diff_tensor.fill(0)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self.lr}, weight_decay={self.weight_decay})"


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and Nesterov.

    Parameters
    ----------
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient (0 disables momentum).
    weight_decay : float
        L2 weight decay.
    nesterov : bool
        Use Nesterov-accelerated momentum when ``momentum > 0``.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(lr, weight_decay)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._velocity: Dict[tuple, np.ndarray] = {}

    def step(self, net: Net) -> None:
        for layer, i, blob in _learnable_params(net):
            self._update_param(blob, (layer.name, i), self._grad(blob))

    def _update_param(self, blob, key, grad: np.ndarray) -> None:
        data = blob.mutable_data_tensor()
        if not self.momentum:
            data[:] -= self.lr * grad
            return
        v = self._velocity.get(key)
        if v is None:
            v = np.zeros_like(data)
            self._velocity[key] = v
        if self.nesterov:
            v_prev = v.copy()
            v[:] = self.momentum * v - self.lr * grad
            # Nesterov look-ahead: data -= mu*v_prev + (1+mu)*v
            data[:] += -self.momentum * v_prev + (1 + self.momentum) * v
        else:
            v[:] = self.momentum * v - self.lr * grad
            data[:] += v

    def state_dict(self) -> dict:
        """Return optimizer state (velocity buffers) for checkpointing."""
        return {f"{k[0]}:{k[1]}": v.copy() for k, v in self._velocity.items()}

    def load_state_dict(self, state: dict) -> None:
        """Restore optimizer state from a previous :meth:`state_dict`."""
        self._velocity = {
            (k.rsplit(":", 1)[0], int(k.rsplit(":", 1)[1])): _as_float32(v, field="optimizer velocity")
            for k, v in state.items()
        }


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2015) with bias correction.

    Parameters
    ----------
    lr : float
        Learning rate.
    beta1, beta2 : float
        Exponential decay rates for the first/second moment estimates.
    eps : float
        Small constant added to the denominator for numerical stability.
    weight_decay : float
        L2 weight decay.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(lr, weight_decay)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m: Dict[tuple, np.ndarray] = {}
        self._v: Dict[tuple, np.ndarray] = {}
        self._t: int = 0

    def step(self, net: Net) -> None:
        self._t += 1
        t = self._t
        b1 = self.beta1
        b2 = self.beta2
        b1t = 1.0 - b1 ** t
        b2t = 1.0 - b2 ** t
        for layer, i, blob in _learnable_params(net):
            key = (layer.name, i)
            grad = self._grad(blob)
            data = blob.mutable_data_tensor()
            m = self._m.get(key)
            if m is None:
                m = np.zeros_like(data)
                self._m[key] = m
            v = self._v.get(key)
            if v is None:
                v = np.zeros_like(data)
                self._v[key] = v
            m[:] = b1 * m + (1.0 - b1) * grad
            v[:] = b2 * v + (1.0 - b2) * (grad * grad)
            mhat = m / b1t
            vhat = v / b2t
            data[:] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def state_dict(self) -> dict:
        """Return optimizer state (m/v buffers + step count) for checkpointing."""
        return {
            "t": self._t,
            "m": {f"{k[0]}:{k[1]}": v.copy() for k, v in self._m.items()},
            "v": {f"{k[0]}:{k[1]}": v.copy() for k, v in self._v.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore optimizer state from a previous :meth:`state_dict`."""
        self._t = int(state.get("t", 0))
        parse = lambda key: (key.rsplit(":", 1)[0], int(key.rsplit(":", 1)[1]))  # noqa: E731
        self._m = {parse(k): _as_float32(v, field="optimizer m") for k, v in state.get("m", {}).items()}
        self._v = {parse(k): _as_float32(v, field="optimizer v") for k, v in state.get("v", {}).items()}


class LRScheduler:
    """Base class for learning-rate schedulers.

    Subclasses implement :meth:`get_lr` which returns the new learning rate for
    the current epoch. Calling :meth:`step` advances the epoch counter and writes
    the new rate back to ``optimizer.lr``.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> float:
        """Advance the scheduler and update ``optimizer.lr``; returns the new lr."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self.optimizer.lr = self.get_lr()
        return self.optimizer.lr

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_lr={self.base_lr})"


class StepLR(LRScheduler):
    """Decay the learning rate by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def get_lr(self) -> float:
        epoch = max(0, self.last_epoch)
        return self.base_lr * (self.gamma ** (epoch // self.step_size))


class MultiStepLR(LRScheduler):
    """Decay the learning rate by ``gamma`` at each milestone epoch."""

    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)

    def get_lr(self) -> float:
        epoch = max(0, self.last_epoch)
        decays = sum(1 for m in self.milestones if epoch >= m)
        return self.base_lr * (self.gamma ** decays)


class ExponentialLR(LRScheduler):
    """Decay the learning rate by ``gamma`` every epoch."""

    def __init__(self, optimizer: Optimizer, gamma: float = 0.9, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.gamma = float(gamma)

    def get_lr(self) -> float:
        epoch = max(0, self.last_epoch)
        return self.base_lr * (self.gamma ** epoch)


class CosineAnnealingLR(LRScheduler):
    """Cosine-annealed learning rate over ``T_max`` epochs.

    ``lr(epoch) = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2``
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)

    def get_lr(self) -> float:
        epoch = max(0, self.last_epoch)
        if self.T_max == 0:
            return self.eta_min
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2


class Solver:
    """Training loop that drives a ``Net`` through forward/backward/update.

    The default loss contract matches caffe-ffi's native loss layers (e.g.
    ``SoftmaxWithLoss``, ``Hinge``, ``MarginRanking``): during ``forward`` the
    network produces a scalar loss output blob, and ``backward`` seeds that
    blob's gradient with ``[1.0]``.

    Parameters
    ----------
    net : Net
        The network to train.
    optimizer : Optimizer
        Optimizer used to update learnable parameters.
    loss_blob : str
        Name of the loss output blob produced by the network (default ``"loss"``).
    metric_blob : str or None
        Optional output blob name used as a secondary metric (e.g. ``"accuracy"``).
    scheduler : LRScheduler or None
        Optional learning-rate scheduler stepped once per epoch.
    """

    def __init__(
        self,
        net: Net,
        optimizer: Optimizer,
        loss_blob: str = "loss",
        metric_blob: Optional[str] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        self.net = net
        self.optimizer = optimizer
        self.loss_blob = loss_blob
        self.metric_blob = metric_blob
        self.scheduler = scheduler
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "metric": [],
            "lr": [],
        }

    def train(self, mode: bool = True) -> None:
        """Switch every layer to training (``True``) or test/inference (``False``) mode."""
        for layer in self.net.layers_array():
            layer.set_train_mode(bool(mode))

    def step(self, inputs: Dict[str, np.ndarray]) -> float:
        """Run one training step on a batch of inputs.

        Parameters
        ----------
        inputs : dict of str to ndarray
            Mapping from input blob names to numpy arrays (labels included for
            native loss layers that consume a ``label`` blob).

        Returns
        -------
        float
            The scalar loss value for this batch.
        """
        out = self.net.forward(inputs)
        loss = float(out[self.loss_blob].flat[0]) if self.loss_blob in out else None
        self.net.backward({self.loss_blob: np.array([1.0], dtype=np.float32)})
        self.optimizer.step(self.net)
        if loss is not None:
            self.history["loss"].append(loss)
        if self.metric_blob and self.metric_blob in out:
            self.history["metric"].append(float(out[self.metric_blob].flat[0]))
        return loss

    def fit(
        self,
        train_batches: Union[Iterable[Dict[str, np.ndarray]], Callable[[], Iterable[Dict[str, np.ndarray]]]],
        epochs: int = 1,
        do_validate: bool = False,
        val_batches: Optional[Iterable[Dict[str, np.ndarray]]] = None,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the network for several epochs.

        Parameters
        ----------
        train_batches : iterable (or callable returning one) of input dicts
            Each element is a dict mapping input blob names to numpy arrays.
            If a callable, it is re-invoked at the start of every epoch.
        epochs : int
            Number of epochs (each epoch consumes the full ``train_batches``).
        do_validate : bool
            If ``True``, run a validation pass after each epoch using
            ``val_batches`` (each batch must contain the ``"__label__"`` key
            holding the ground-truth labels, or the metric is computed from the
            network's output directly).
        val_batches : iterable of dicts, optional
            Validation batches. Only used when ``do_validate`` is ``True``.
        log_interval : int
            Print a progress line every N steps.

        Returns
        -------
        dict of str to list of float
            Training history (``loss``, ``lr``, and optionally ``metric``).
        """
        self.train(True)
        if self.scheduler is not None:
            self.scheduler.step()
        for epoch in range(1, epochs + 1):
            epoch_loss: List[float] = []
            source = train_batches() if callable(train_batches) else train_batches
            for step_idx, batch in enumerate(source, start=1):
                loss = self.step(batch)
                epoch_loss.append(loss)
                if log_interval and step_idx % log_interval == 0:
                    avg = float(np.mean(epoch_loss[-log_interval:]))
                    print(
                        f"[epoch {epoch}/{epochs}] step {step_idx} | loss={avg:.4f}"
                        + (f" | lr={self.optimizer.lr:.6f}" if self.scheduler else "")
                    )
            if self.scheduler is not None:
                self.scheduler.step()
            if self.history:
                epoch_avg = float(np.mean(epoch_loss))
                self.history["lr"].append(self.optimizer.lr)
                print(f"[epoch {epoch}/{epochs}] avg_loss={epoch_avg:.4f}")
                if do_validate and val_batches is not None:
                    self.validate(val_batches)
        return self.history

    def validate(self, val_batches: Iterable[Dict[str, np.ndarray]]) -> float:
        """Run a validation pass; returns the mean metric (or loss) over batches."""
        self.train(False)
        scores: List[float] = []
        for batch in val_batches:
            out = self.net.forward(batch)
            if self.metric_blob and self.metric_blob in out:
                scores.append(float(out[self.metric_blob].flat[0]))
            elif self.loss_blob in out:
                scores.append(float(out[self.loss_blob].flat[0]))
        self.train(True)
        return float(np.mean(scores)) if scores else 0.0