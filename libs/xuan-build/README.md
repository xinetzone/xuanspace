# xuan-build

构建后端能力层：把「按项目类型选择构建方式」收敛为一个能力键
`xuan:build/backend#dispatch`，消费者不再在业务代码中按类型分支。

## 结构

| 模块 | 职责 |
|---|---|
| `xuan_build.backends` | `BuildBackend` 协议与 python/native/static/other 四种实现 |
| `xuan_build.dispatch` | `BuildDispatcher`：项目类型 → 后端；未知类型落 `other` |
| `xuan_build` | 插件清单 `MANIFEST`、运行时 `PLUGIN`、默认 `dispatch()` |

## 使用

```python
from xuan_build import dispatch

backend = dispatch("python")
if backend.skip_message("demo") is None:
    ok = backend.build(project_path, reporter)
```

`reporter` 只需提供 `print(text: str) -> None`（Rich `Console` 直接满足），
库本身不依赖终端渲染，便于在测试中替换为记录器。
