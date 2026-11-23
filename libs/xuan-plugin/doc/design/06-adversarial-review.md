> 原子化来源：[xuanspace-pluginization-design.md](file:///media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md) · 原文 §6（L397–L413）
> 文档索引：[README.md](../README.md)

## 6. V — 对抗审查（4 视角）

| # | 视角 | 异议 | 处置 |
|---|---|---|---|
| V1 | 魔鬼（技术极限） | "Python 没有模块卸载语义；`tvm-ffi`/`caffe-ffi` 的 `.so` 一旦 dlopen 就无法回滚，你的'可撤销'是假的。" | **承认边界并前移**：K3 只承诺"效果可撤销 + 引用可释放"，不承诺"库可卸载"；对原生后端提供 `subprocess` 隔离模式（能力句柄指向子进程），把"不可撤销"约束在子进程边界内。文档与 `reversible` 字段明示该限制。 |
| V2 | 魔鬼 | "entry points 动态发现会让不同机器的装配结果不同，与 H4 冲突。" | 双通道合并后**以 lock 为准**：发现阶段只产出候选，求解结果写入 lock，H4 校验"重算 == lock"；未在 lock 中的新插件需显式 `xs plugin add` 才生效。 |
| V3 | 新人（可用性） | "五张契约 + 十步算法，新人记不住，会绕过。" | 提供 `xs plugin new` 模板（生成合规清单 + 空 apply + Disposable 骨架）与单页 checklist；`xs plugin verify` 给出可执行的修复建议；文档只保留"三条公理"作为心智模型。 |
| V4 | 新人 | "错误信息说 INACTIVE，但新人不知道为什么自己没生效。" | INACTIVE 必须携带 `reason` 与"差哪个键、谁可能提供"；`xs plugin why <id>` 直接输出因果链。 |
| V5 | 老板（投入产出） | "这要改 CLI、compose、构建后端、FFI，收益怎么算？" | 只先做 P0–P1（kernel 下沉 + CLI 插件化），两周内可用两个硬指标对比：新增命令的改动文件数（目标 1–2 个）、装配失败定位时间（目标 < 1 分钟）。P2 起按试点收益逐个放行。 |
| V6 | 老板 | "插件化会不会让发布更不可控（谁都能扩展）？" | 默认关闭动态发现；发行包内的插件需在 lock 中显式登记；作用域（K5）限制可见能力，相当于"能力即权限"（F14）。 |
| V7 | 未来（演进） | "`id(impl)` 改成摘要后，每次代码微调都会触发全下游重载，反而更慢。" | 摘要按**能力契约**计算（方法名 + 签名 + 显式 `api_version`），不含实现字节；实现内部优化不改变身份。等价于论文 §6.6 的"结构兼容"而非"字节相等"。 |
| V8 | 未来 | "如果将来要跨语言（JS/Rust 插件），Python 的 Context 模型会不会成为枷锁？" | L0 只定义**协议**（键语法、清单字段、状态机、对账语义），Python 实现是参考实现；跨语言通过 `xuan.plugins.lock` 与 JSON-RPC 能力代理对接，不共享内存对象。 |

> V 门结论：方案在"可撤销"与"可复现"两处需显式降级承诺（V1、V7 已处置），其余异议不改变总体结构。

---

相关页面：[文档索引](../README.md) · 上一篇：[方案：Xuan Plugin Contract（XPC）](05-solution.md) · 下一篇：[C — 落地计划（原子任务）](07-roadmap.md)
