---
id: caffe-ffi-api-reference
title: caffe-ffi 训练 API 参考
date: 2026-08-04
category: caffe-ffi
tags: [caffe-ffi, api, solver, serialization, reference]
source: "caffe-ffi-tvm-integration/tasks.md#T33"
---

# caffe-ffi 训练 API 参考

> 本文档是 `caffe_ffi.solver` 与 `caffe_ffi.serialization` 两个模块的完整 API 参考。使用示例见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

---

## 模块：`caffe_ffi.solver`

### 优化器

#### `Optimizer(lr=0.01, weight_decay=0.0)`
参数更新优化器基类。

- `lr: float` — 基础学习率。
- `weight_decay: float` — L2 权重衰减系数，作用于梯度 `grad = diff + weight_decay * data`。

方法：
- `step(net: Net) -> None` — 用 `net` 中所有可学习 blob 的累积梯度更新权重（子类实现）。
- `zero_grad(net: Net) -> None` — 清零所有可学习 blob 的 diff。
- `_grad(blob) -> np.ndarray` — 返回有效梯度（含可选 L2 weight decay）。
- `_update_param(blob, key, grad) -> None` — 对单个参数应用一步更新（子类实现）。

#### `SGD(lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)`
随机梯度下降，支持动量与 Nesterov 加速。

- `momentum: float` — 动量系数（0 禁用动量）。
- `nesterov: bool` — 动量>0 时使用 Nesterov 加速。

更新规则（无动量）：
```
data -= lr * grad
```
有动量：
```
v = momentum * v - lr * grad
data += v
```
Nesterov：
```
v = momentum * v - lr * grad
data += -momentum * v_prev + (1 + momentum) * v
```

方法：
- `state_dict() -> dict` — 返回速度缓冲区字典（key 为 `"layer_name:blob_index"`）。
- `load_state_dict(state: dict) -> None` — 从字典恢复速度缓冲区。

#### `Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)`
Kingma & Ba (2015) 优化器，带偏差校正。

更新规则：
```
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
mhat = m / (1 - beta1^t)
vhat = v / (1 - beta2^t)
data -= lr * mhat / (sqrt(vhat) + eps)
```

方法：
- `state_dict() -> dict` — 返回 `{"t", "m", "v"}` 状态。
- `load_state_dict(state: dict) -> None` — 恢复 m/v 与步数。

### 学习率调度器

#### `LRScheduler(optimizer, last_epoch=-1)`
调度器基类。

- `optimizer: Optimizer` — 目标优化器。
- `last_epoch: int` — 当前 epoch（初始 -1）。

方法：
- `get_lr() -> float` — 返回当前 epoch 的新学习率（子类实现）。
- `step(epoch=None) -> float` — 推进 epoch 并写回 `optimizer.lr`，返回新 lr。

#### `StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)`
每 `step_size` 个 epoch 将学习率乘以 `gamma`：
```
lr(epoch) = base_lr * gamma ** (epoch // step_size)
```

#### `MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)`
在每个 `milestones` 指定的 epoch 将学习率乘以 `gamma`。

#### `ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)`
每个 epoch 将学习率乘以 `gamma`：
```
lr(epoch) = base_lr * gamma ** epoch
```

#### `CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)`
余弦退火：
```
lr(epoch) = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
```

### 训练循环

#### `Solver(net, optimizer, loss_blob="loss", metric_blob=None, scheduler=None)`
驱动 `Net` 完成前向/反向/更新的训练循环。

- `net: Net` — 待训练网络。
- `optimizer: Optimizer` — 权重更新优化器。
- `loss_blob: str` — 网络产生的标量 loss 输出 blob 名（默认 `"loss"`）。
- `metric_blob: str | None` — 可选，作为验证指标的输出 blob 名（如 `"accuracy"`）。
- `scheduler: LRScheduler | None` — 可选，每个 epoch 结束后自动 `step()`。

属性：
- `history: dict` — 训练历史，含 `"loss"`/`"metric"`/`"lr"` 列表。

方法：
- `train(mode=True) -> None` — 切换所有层训练/推理模式。
- `step(inputs: dict) -> float` — 单个 batch 训练步，返回该 batch loss。
- `fit(train_batches, epochs=1, do_validate=False, val_batches=None, log_interval=10) -> dict` — 多 epoch 训练，返回 `history`。
- `validate(val_batches) -> float` — 推理模式验证，返回平均 metric（或 loss）。

`fit` 参数：
- `train_batches` — 可迭代的输入 dict 列表，或返回该迭代器的可调用对象（每 epoch 重新调用）。
- `epochs: int` — 训练 epoch 数。
- `do_validate: bool` — 每个 epoch 后是否跑验证。
- `val_batches` — 验证 batch 迭代器（`do_validate=True` 时使用）。
- `log_interval: int` — 每 N 步打印进度行。

---

## 模块：`caffe_ffi.serialization`

### `save_net(net, path) -> None`
将网络当前权重写入 `.caffemodel` 文件。

- `net: Net` — 待保存网络。
- `path: str | Path` — 目标文件路径。

### `load_net(net, path) -> None`
从 `.caffemodel` 文件加载权重到网络（按层名匹配）。

- `net: Net` — 目标网络（须与保存时结构兼容）。
- `path: str | Path` — 源文件路径。

### `net_parameter_to_file(net, path) -> None`
`save_net` 的底层实现，序列化权重为 `NetParameter` protobuf。

### `weights_to_dict(net) -> dict`
导出所有可学习权重为 `{"layer_name:blob_index": ndarray}` 字典（shallow copy）。

### `dict_to_weights(net, weights) -> None`
从字典恢复权重；未知 key 静默忽略。

---

## 顶层导出

`caffe_ffi` 顶层 `__init__.py` 已重新导出上述类与函数，可直接使用：

```python
import caffe_ffi
caffe_ffi.SGD, caffe_ffi.Adam, caffe_ffi.Solver,
caffe_ffi.StepLR, caffe_ffi.CosineAnnealingLR, ...
caffe_ffi.save_net, caffe_ffi.load_net, ...
```

---

## 相关文档

- [训练指南](TRAINING_GUIDE.md)
- [文档索引](../README.md)