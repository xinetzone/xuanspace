---
id: caffe-ffi-training-guide
title: caffe-ffi 训练指南
date: 2026-08-04
category: caffe-ffi
tags: [caffe-ffi, training, solver, optimizer, serialization, guide]
source: "caffe-ffi-tvm-integration/tasks.md#T33"
---

# caffe-ffi 训练指南

> 本文档介绍 caffe-ffi 的**训练工程化**能力（P4 阶段，Task 33）：基于 `caffe_ffi.solver` 与 `caffe_ffi.serialization` 模块，完成从网络构建 → 训练 → 模型保存/加载 → 评估的完整闭环。

---

## 一、概述

caffe-ffi 在完成 19 类层 Backward 反向传播与 LeNet/MNIST 端到端训练（97.95%）之后，P4 阶段将训练流程抽象为可复用组件：

| 模块 | 职责 | 关键类/函数 |
|------|------|-------------|
| `caffe_ffi.solver` | 优化器 + 学习率调度 + 训练循环 | `Optimizer`/`SGD`/`Adam`、`LRScheduler`/`StepLR`/`MultiStepLR`/`ExponentialLR`/`CosineAnnealingLR`、`Solver` |
| `caffe_ffi.serialization` | 模型权重保存/加载 | `save_net`/`load_net`/`net_parameter_to_file`/`weights_to_dict`/`dict_to_weights` |

训练流程与 `examples/lenet_mnist_train.py` 中验证过的手动循环完全一致：**前向 → 用 `[1.0]` 播种 loss blob 的梯度 → 反向 → 权重更新**。

---

## 二、快速开始

```python
import numpy as np
import caffe_ffi
from caffe_ffi import net_from_param, net_param_from_string
from caffe_ffi.solver import SGD, StepLR, Solver
from caffe_ffi.serialization import save_net, load_net

# 1. 构建网络（prototxt 或 NetParameter）
proto = '''
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
net = net_from_param(net_param_from_string(proto))

# 2. 组装优化器 + 调度器 + Solver
optimizer = SGD(lr=0.05, momentum=0.9)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
solver = Solver(net, optimizer, loss_blob="loss",
                metric_blob="accuracy", scheduler=scheduler)

# 3. 训练（train_batches 可传迭代器或返回迭代器的可调用对象）
rng = np.random.RandomState(0)
def make_batches(n):
    for _ in range(n):
        x = rng.randn(32, 16).astype(np.float32)
        y = rng.randint(0, 4, size=(32, 1)).astype(np.float32)
        x += y * 0.5
        yield {"data": x, "label": y}

history = solver.fit(make_batches(20), epochs=4,
                     do_validate=True, val_batches=list(make_batches(5)),
                     log_interval=10)

# 4. 保存模型
save_net(net, "model.caffemodel")

# 5. 加载到全新网络
fresh = net_from_param(net_param_from_string(proto))
load_net(fresh, "model.caffemodel")
```

完整可运行示例见 [examples/mlp_classifier_train.py](../../examples/mlp_classifier_train.py)。

---

## 三、优化器（Optimizer）

| 类 | 说明 |
|----|------|
| `Optimizer` | 基类。`lr`/`weight_decay`，`step(net)` 遍历所有可学习参数并原地更新，`zero_grad(net)` 清零梯度。 |
| `SGD` | 随机梯度下降，支持 `momentum`（动量）与 `nesterov`（Nesterov 加速）。 |
| `Adam` | Kingma & Ba (2015)，带偏差校正，`beta1`/`beta2`/`eps`。 |

**关键设计**：
- 权重更新通过 `blob.mutable_data_tensor()` **COW 感知**写入，自动触发写时克隆，保证零拷贝共享语义。
- 梯度来源为每个可学习 blob 的 `diff`（`diff_tensor`），可选叠加 L2 weight decay（`grad = diff + weight_decay * data`）。
- 每个参数以 `(layer_name, blob_index)` 为稳定 key，维护优化器状态（SGD 的 velocity、Adam 的 m/v）。

### 状态保存/恢复（checkpoint）

```python
state = optimizer.state_dict()   # 保存速度/矩状态
optimizer2.load_state_dict(state)  # 恢复，用于断点续训
```

---

## 四、学习率调度器（LRScheduler）

| 类 | 调度规则 |
|----|----------|
| `StepLR` | 每 `step_size` 个 epoch 乘以 `gamma`。 |
| `MultiStepLR` | 在每个 `milestones` 指定的 epoch 乘以 `gamma`。 |
| `ExponentialLR` | 每个 epoch 乘以 `gamma`。 |
| `CosineAnnealingLR` | 在 `T_max` 个 epoch 上余弦退火至 `eta_min`。 |

所有调度器在 `step()` 时推进 epoch 计数并**写回 `optimizer.lr`**。`Solver.fit` 会在每个 epoch 结束时自动调用 `scheduler.step()`。

---

## 五、训练循环（Solver）

`Solver` 封装前向/反向/更新，默认 loss 契约与 caffe-ffi 原生 loss 层一致：前向产生标量 loss 输出 blob，反向用 `[1.0]` 播种其梯度。

| 方法 | 说明 |
|------|------|
| `step(inputs)` | 单个 batch 的训练步，返回该 batch 的 loss。 |
| `fit(train_batches, epochs, ...)` | 多 epoch 训练循环，返回 `history`（`loss`/`metric`/`lr` 列表）。 |
| `validate(val_batches)` | 推理模式验证，返回平均 metric（或 loss）。 |
| `train(mode)` | 切换所有层训练/推理模式。 |

### 数据契约

- `train_batches`：可迭代的输入 dict 列表，或返回该迭代器的可调用对象（可调用对象会在每个 epoch 开头重新调用，适合数据增强/打乱）。
- 每个 batch 是 `{输入blob名: ndarray}` 映射，loss 层所需的 `label` 一并放入。
- `metric_blob` 用于验证时读取网络输出的标量指标（如 `"accuracy"`）。

---

## 六、模型序列化（Serialization）

caffemodel 格式是序列化的 `caffe.NetParameter` protobuf。caffe-ffi 按层 **名称** 匹配加载权重（见 `Net.CopyTrainedLayersFrom`），因此权重 caffemodel 只需每层的 `name` 与其 `blobs`。

| 函数 | 说明 |
|------|------|
| `save_net(net, path)` | 将网络当前权重写入 `.caffemodel` 文件。 |
| `load_net(net, path)` | 从 `.caffemodel` 加载权重到网络（按层名匹配）。 |
| `net_parameter_to_file(net, path)` | `save_net` 的底层实现。 |
| `weights_to_dict(net)` | 导出权重为 `{"layer_name:blob_index": ndarray}` 字典（便于 pickle/快照/检查）。 |
| `dict_to_weights(net, weights)` | 从字典恢复权重；未知 key 静默忽略。 |

> 序列化与 `caffe_ffi.io`（`read_net`/`read_net_from_binary` 等）兼容，产出的 protobuf 可通过 `read_net_from_binary` 重新解析。

---

## 七、训练/推理模式与 Dropout

- `Solver.train(True)` 将所有层切到训练模式；`Solver.validate` 内部自动切到推理模式。
- Dropout 层在训练模式启用 inverted dropout + mask 缓存；推理模式为恒等映射（并启用 COW 零拷贝共享优化）。

---

## 八、示例与测试

- **示例**：[examples/mlp_classifier_train.py](../../examples/mlp_classifier_train.py)（端到端训练 + 保存/加载 + 评估）
- **测试**：[tests/python/test_solver.py](../../tests/python/test_solver.py)（优化器/调度器/求解器）、[tests/python/test_serialization.py](../../tests/python/test_serialization.py)（权重字典与 caffemodel round-trip）
- **API 参考**：[API_REFERENCE.md](API_REFERENCE.md)

---

## 相关文档

- [文档索引](../README.md)
- [P4 阶段路线图](../../../.trae/specs/caffe-ffi-tvm-integration/p4-roadmap.md)
- [P3 阶段总复盘](../../../.agents/docs/retrospective/reports/code-optimization/retrospective-caffe-ffi-p3b-test-milestone-20260731/sections/19-p3-phase-retrospective.md)