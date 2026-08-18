# okf-toolchain

Open Knowledge Format (OKF) v0.2 工具链：一个**零运行时依赖**、**可插拔**的 Python 工具，用于在生产/消费 OKF Bundle 时完成校验、脚手架生成、索引/日志合成、信任等级推导等标准化操作。

## 特性

- **零运行时依赖**：仅使用 Python 3.14.6 标准库（`dataclasses`/`pathlib`/`argparse`/`re`/`asyncio`/`tomllib`），不依赖 PyYAML/typer/rich 等第三方包。
- **Dataclass 数据模型**：所有核心类型使用 `@dataclass(frozen=True)` 定义，不可变、类型安全。
- **时空可组合性架构**（Cordis 启发）：可逆效应（Revertible Effects）+ 响应式协同效应（Reactive Coeffects）。
- **Harness 架构**（DeepSeek Harness 启发）：一切皆插件、无特权内核、Capability Seam 三角色、事件系统五种分发模式。
- **scikit-build-core 构建**：纯 Python 起步，预留 C/C++ 原生模块扩展点。

## 安装

```powershell
pip install .
```

## 命令

| 命令 | 说明 |
|------|------|
| `okf validate <path> [--strict]` | 一致性校验（§11） |
| `okf init <path>` | 创建 Bundle 骨架 |
| `okf index <path>` | 生成/更新 `index.md` |
| `okf inspect <path> [concept_id]` | 查看概念详情 |
| `okf trust <path> [concept_id]` | 信任等级与保鲜状态 |
| `okf list <path> [--type X] [--tag Y]` | 列出概念 |

## 插件装配

插件通过 `pyproject.toml` 的 `[tool.okf.plugins]` 配置节声明式装配，第三方实现可替换任意默认插件：

```toml
[tool.okf.plugins]
conformance_checker = "my_org.okf:CustomChecker"
```

## 参考

- OKF 规范：`vendor/knowledge-catalog/okf/SPEC.md`