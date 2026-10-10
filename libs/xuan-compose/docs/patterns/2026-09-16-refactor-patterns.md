# 单体重构可复用模式六则（xuan-compose T1-T12 萃取）

> 来源：2026-09-16 交付总结报告 §6「E 萃取：可复用模式（G3，供后续单体重构迁移）」，
> 2026-10-10 原子化重写独立成篇，内容零增删改（标题与导航性文字随原子化调整）。
> **本篇是可更新的模式沉淀，不是冻结快照**：六个模式本体可随后续重构实践增补/修订，
> 但 2026-09-16 案例事实（提交号、数字、D13 等）保持原样。
> 配套案例：[交付报告](../delivery/2026-09-16-delivery-summary.md)、
> [对等审计](../parity/README.md)、[T11 审查](../review/2026-09-16-t11-adversarial-review.md)。

## 1. 快照对拍迁移（vendor-independent snapshot）

- 触发：把"装饰器/单例构造的 CLI 解析"从第三方单体迁出、又要证明参数表对等时。
- 核心步骤：写 vendor 无关的树序列化器（含 meta 溯源）→ 一次性脚本加载上游构造
  parser 树生成 JSON 快照入库 → 新库纯函数构造器读快照逐节点对拍 → 快照仅升级基线时重生成。
- 反模式：在每个测试里 import vendor 上游（CI 强依赖检出、版本漂移即红）；
  或把关键维度（default/nargs/required）排除在快照外"让测试变绿"。

## 2. 分层切片流水线（自底向上，一片一门禁）

- 触发：单体拆包。顺序：基础→规范→执行→翻译→引擎→命令→CLI→门禁→审计→V。
- 核心步骤：每片冻结上游行号 → 翻译并登记差异 → 移植该层上游单测 →
  补真实分支测试 → 该层门禁全绿才开下一片；原子提交按片/按职责拆批。
- 反模式：多层并进后统一测试（缺陷定位跨层）；顺手重写算法（无法证明对等）。

## 3. 跨 Task 符号去重检查

- 触发：多个 Task 迁移同一单体的相邻区域（上游函数本就散落）。
- 核心步骤：每片用 AST 符号盘点核对"上游符号→目标模块"，发现同名跨模块即停手去重。
- 反模式（本项目真实踩中）：T4 已迁入 dependencies.py 的 check_dep_conditions，
  T7 按 handler 区域又逐字节重译一份，直到 T11 符号盘点才暴露（[D13](../parity/differences.md#d13)）。
  教训：去重检查不能只放在最后审计，应每片执行。

## 4. 上游怪癖"测试固化"而非"顺手修正"

- 触发：新写测试时实际行为与"合理预期"冲突。
- 核心步骤：先假设测试错、回到上游代码核对；确认是上游行为则按真实输出断言并登记怪癖。
  本项目 10+ 次此类纠正（`1m:30s` 非法、sub_dir 不剥 `./`、`--service-ports` 删端口后重建、
  bytes 被 is_list 判真、OverrideTag scalar 无类型解析等）。
- 反模式：为让断言符合直觉而改生产代码——那是在制造与上游的不对等。
- 怪癖清单：[parity/upstream-quirks.md](../parity/upstream-quirks.md)。

## 5. 双平台门禁分工 + 只读子模块纪律

POSIX 语义测试以 WSL py3.14 为权威（与上游 CI 一致），lint/类型以 Windows py314 为权威，
平台路径差异登记而非强改；vendor 子模块只读（曾发生整树 481 文件 CRLF 翻转，
以 `--ignore-cr-at-eol` 证实零语义差异后 restore，TR-11.1）。

## 6. Git 操作约定（Windows/WSL 混合环境）

- 中文 commit message 用 UTF-8 文件 + `git commit -F`（规避 PowerShell→wsl heredoc 引号丢失）；
- 子模块先提交、主仓后 docs(spec) 再 chore(submodule) 抬 gitlink；
- 子模块/主仓推送前 `fetch` 确认纯 ahead 再 fast-forward。
