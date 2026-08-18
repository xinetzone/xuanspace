# 插件开发

`okf` 的所有能力均由插件提供。插件通过 `provide` 声明「供给的服务」、通过 `inject` 声明「依赖的服务」，Harness 据此自动完成装配顺序与依赖注入。

## 插件模型

`Plugin` 是插件的声明式描述（`src/okf/plugin.py`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 插件名 |
| `apply` | `Callable[[Context, dict], None]` | 装配逻辑（在依赖满足后执行） |
| `inject` | `list[InjectSpec]` | 依赖的服务名列表 |
| `provide` | `list[str]` | 提供的服务名列表 |

`InjectSpec` 含 `name`（依赖服务名）与可选的 `config`。

## 默认插件清单

| 插件 | 注入（inject） | 提供（provide） |
|------|----------------|-----------------|
| `bundle_loader` | — | `bundle_accessor` |
| `conformance_checker` | `bundle_accessor` | `conformance_report` |
| `index_synthesizer` | `bundle_accessor` | `index_generator` |
| `log_synthesizer` | `bundle_accessor` | `log_generator` |
| `trust_deriver` | `bundle_accessor` | `trust_analyzer` |
| `link_resolver` | `bundle_accessor` | `link_analyzer` |
| `cli_adapter` | 上述全部服务 | `cli_services` |

`bundle_loader` 无依赖，最先生效；其余插件依赖 `bundle_accessor`；`cli_adapter` 依赖全部服务，最后生效。

## 装配流程

Harness 按 `[tool.okf.plugins]` 配置装配插件：

1. 读取 `pyproject.toml` 的 `[tool.okf.plugins]`（不存在则用默认清单）。
2. 逐个 `import` 插件类并实例化。
3. 依据 `inject` 依赖关系执行**拓扑排序**（Kahn 算法）。
4. 按序将插件注册到 `Context`；每个插件生效后 `provide` 的服务自动 `notify` 依赖方。

## 替换默认插件

第三方只需在 `pyproject.toml` 中声明同名的插件即可替换默认实现：

```toml
[tool.okf.plugins]
conformance_checker = "my_org.okf:CustomChecker"
```

`my_org.okf.CustomChecker` 既可以是返回 `Plugin` 对象的类（见下），也可以是可调用对象（会被自动包装为 `Plugin(name=..., apply=instance)`）。

## 自定义插件示例

以下插件提供一个新的服务 `my_service`，并依赖 `bundle_accessor`：

```python
from okf.plugin import InjectSpec, Plugin


class CustomChecker:
    def __call__(self, ctx, config):
        bundle = ctx.get("bundle_accessor")  # 依赖注入
        # ... 自定义校验逻辑 ...
        ctx.provide("my_service", lambda: len(bundle.concepts))

    def __new__(cls):
        instance = object.__new__(cls)
        return Plugin(
            name="my_checker",
            apply=instance,
            inject=[InjectSpec("bundle_accessor")],
            provide=["my_service"],
        )
```

关键点：

- `__call__` 是装配逻辑，仅当依赖满足时执行。
- `ctx.get(name)` 获取依赖服务，`ctx.provide(name, impl)` 供应服务。
- `__new__` 返回 `Plugin` 描述，声明 `inject` / `provide`。

## 加载失败的处理

`Harness._load_plugins_from_map` 对加载失败的插件打印警告与堆栈（`Warning: Failed to load plugin ...`）而**不中断**整个装配过程；后续插件仍按拓扑顺序继续加载。