# xuan-plugin

Xuanspace 插件内核：可逆装配、能力注册与生命周期管理。

## 定位

本库是 Cordis 组件范式的最小移植，提供五项能力：

- **可撤销效果**：每个副作用在注册时返回逆操作，退役即回滚；
- **反应式协效应**：组件声明依赖集合，依赖实现变化时仅受影响组件退役重插；
- **能力注册**：服务按能力键注册与查找，同装配内提供集合两两不交；
- **生命周期状态机**：`PENDING` / `INACTIVE` / `LOADING` / `ACTIVE` / `UNLOADING` / `FAILED` / `DISPOSED` 七态；
- **装配求解**：清单解析（K2）、确定性拓扑定序、隔离域、声明/实际对账与失败回滚。

## 边界

- 零运行时依赖，仅使用标准库；
- 不在本库内实现锁文件与治理门禁（CI 断言）——那些属于治理层；
- 原生扩展不承诺进程内热卸载（CPython 不执行 `dlclose`），须以静态生命周期使用。

## 安装

```bash
python -m pip install -e libs/xuan-plugin
```

## 使用

```python
from xuan_plugin import Context, Plugin

ctx = Context()
ctx.plugin(Plugin(name="demo", apply=lambda c, cfg: None))
```

## 设计文档

设计说明按章节分卷存放于 [doc/README.md](doc/README.md)：事实基线、洞察、七模式、XPC 方案、对抗审查、落地计划、P1-2 复盘与 ADR；逐页行区间与源文指纹见 [doc/MANIFEST.md](doc/MANIFEST.md)。
