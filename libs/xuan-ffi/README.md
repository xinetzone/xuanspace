# xuan-ffi

FFI 能力键与子进程隔离代理（P2-3 试点 C）。

## 能力键

```text
xuan:ffi/<name>#api
```

内置名称：`caffe`（`caffe_ffi`）、`demo`（`demo_ffi`）、`npu`（`npu_ffi`）。
消费者只注入能力键，不 import 具体 FFI 发行包；换实现或换传输模式都不改调用方代码。

## 两种传输

| 模式 | 语义 | 适用 |
|---|---|---|
| `in-process` | 同进程 import 目标模块并调用 | 默认；零往返开销 |
| `subprocess` | 独立解释器 + JSON Lines over stdio | 需反复装卸 A/B、崩溃可恢复 |

两种模式共用同一能力面：

```python
from xuan_ffi import create_ffi

api = create_ffi("demo", module="demo_ffi", mode="subprocess")
print(api.describe())
print(api.invoke("demo.math.gcd", 12, 18))
api.close()  # 结束子进程；可再次 start() 重新装载
```

## 边界

- `invoke` 的参数与返回值须可 JSON 序列化；`Blob`/`Net` 等对象图密集的 API
  仍走进程内直连。
- `close()` 只做逻辑退役：撤销能力句柄、结束子进程，**不承诺**卸载已映射的
  `.so`（ADR-03）。重载原生模块等价于重启进程，这正是子进程模式的意义。
- 清单声明 `lifecycle = "static"`，`xs plugin apply` 只登记能力键，不预启动
  子进程；worker 在首次调用时拉起。
