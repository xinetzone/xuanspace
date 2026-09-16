# xuan-compose T11 独立对抗审查报告（review.md）

> 对应 spec Task 11 / AC-2/3/5/6/9/10/11/12 / TR-11.1-11.4。
> 审查性质：**独立审查（Independent Review）**。审查员为 fresh-context
> 代理（三个并行、互不相通、均未参与 T1-T10 实施），只读分析、命令取证；
> T1-T10 作者仅承担编排、修复与回归，不自审自评。

- 审查日期：2026-09-16
- 上游基线：containers/podman-compose pinned `e3df10472e194ab6d547b5ad25542c5c79e1a5fb`
- 受审版本：xuan-compose T10 提交 `1c40e44`（审查时工作树状态），修复后回归于 T11 工作树
- 审查范围：src/xuan_compose（39 源文件）、tests（56）、scripts（1）、docs、README/CHANGELOG、vendor 基线

## 0. 总结论

| 维度 | 独立结论 | 评级 |
|---|---|---|
| 行为对等保真性（抽查深读 + 移植测试 diff + 快照可信度） | **pass** | 0 blocker / 0 major |
| 分层单向性 / GPL / py3.14 现代化 | **pass**（修复后） | TR-11.3 **5/5** |
| 五攻击者（安全/边界/完整/时序/模糊） | **pass**（无重构新增 blocker/major） | 13 条 minor/nit 全部上游固有 |
| 对抗审查实质度（TR-11.4） | 多视角具体发现合计 20+ 条，≥5 条闭环 | **5/5** |

**最终 Review result = pass。** 0 blocker、0 major；本轮修复 4 项重构引入问题并全量回归；
其余发现经判据分类为「评估保留（重构有意决策）」与「上游固有（对等重构不改，列上游回馈候选）」。

## 1. TR-11.1 vendor 子模块零改动

```
$ git -C vendor/podman-compose status --porcelain   # 审查前置：空
$ git -C vendor/podman-compose rev-parse HEAD
e3df10472e194ab6d547b5ad25542c5c79e1a5fb
```

审查前发现工作树有 481 个文件标记 M，经核实为整树 CRLF↔LF 翻转
（25948 删/25948 增完全对称、`git diff --ignore-cr-at-eol` 零差异、HEAD 未移动，
非重构引入），已 `git restore .` 恢复只读状态；审查与快照生成全程只读 vendor。

## 2. TR-11.2 24 命令 / 145 符号对等抽查

- **符号面**：`python scripts/symbol_inventory.py` 实测上游顶层公开符号
  **142（函数/类 138 + 常量 4）**，与 spec「约 145」一致；139 同名迁入、
  1 重命名（PodmanCompose→ComposeEngine）、2 删除（cmd_run/cmd_parse 装饰器），
  逐条映射见 [parity.md](parity.md)。脚本可复现，输出与文档一致。
- **命令面**：独立 reviewer 核验 24 命令（+help 伪命令）集合、注册顺序、
  handler 全为 async、与 `@cmd_run` 出现序一致；`wait` 无 parser（`COMMAND_PARSERS["wait"]=()`）
  与上游一致。
- **抽查深读**：对等审查员跨层深读 ≥10 个函数（container_to_args、rec_merge_one、
  normalize_service、_parse_compose_file、PullPolicyAction、str_to_seconds、
  parse_short_mount、get_secret_args、check_dep_conditions、Podman.output/run 等），
  逐行对照上游，未发现条件翻转/默认值/退出码/异常类型偏差；T9 测试中固化的
  10 余处「上游怪癖」经复核确为上游真实行为而非测试迎合。
- **移植测试保真性**：抽查 test_rec_subs/test_var_interpolate/test_container_to_args/
  test_pull_image 等与 vendor 同名文件 diff，断言无削弱；`test_normalize_final_build`
  拆分 18（纯函数）+24（引擎）=上游 42 守恒。

## 3. TR-11.3 分层单向性（修复后 5/5）

独立审查员逐条核对 **108 条内部 import**（命令输出取证）：

- translate/ 与规范层对 engine/commands/cli **零引用**；runner 仅依赖 logging_utils，
  不反向依赖任何上层；commands 不 import engine（handler 以 `compose: Any` 收引擎）。
- 子进程边界：全包仅 `runner.py` 出现 `import subprocess` /
  `create_subprocess_exec` / `os.execlp`，argv 全程 list，无 shell。
- `sys.argv` 真实代码引用仅 `cli/main.py:50` 单点（其余为 docstring）。
- `import xuan_compose` 运行时实测只加载包自身，无 argv/进程/yaml 标签注册副作用。
- 扣分项与修复：审查初评 4/5，两项 minor——①规范层 merge↔normalize 惰性环（**本轮已消除，见 §5-F1**）；
  ②argparse.Namespace 作为 DTO 渗入 15 文件（**经评估保留，见 §6-R1**）。
  修复后无跨层反向边、无 import 期/惰性环；commands 触碰的仅是 `argparse.Namespace`
  数据袋（reviewer 实证「无解析/无 argv/无 add_argument」，parser 机械完全封在 cli），
  不构成 AC-9 anchor 所称「commands 直接碰 argparse 细节」的绕行。**评定 5/5。**

## 4. TR-11.4 五攻击者对抗审查（实质度 5/5）

独立审查员走查五视角并运行针对性探针（依赖环探针、StreamReader 70KB 无换行探针）：

1. **安全**：全 argv list + execve，无 shell/eval/os.system；用户可控串
   （devices/extra_hosts/sysctl/labels/volume source/command）均以独立 argv 元素入列，
   引号/分号/`$(...)`/CRLF 无命令语义；无重构新增注入面。唯一密钥面（`--verbose`
   下 `-e KEY=VALUE` 入 INFO 日志）**与上游逐字相同**。
2. **边界**：podman 缺失/版本空/非零退出/compose 缺失/非 dict/缺服务·网络·卷/
   数字解析失败的退出码（-1/1）与异常类型逐项与上游一致。
3. **完整**：多文件合并、标签、include 子目录重写、config_hash、scale 命名逐行对等；
   探针复现 `A→B→C→D→B` 子环 RecursionError——代码与上游同构（上游固有）。
4. **时序**：CancelledError→terminate→10s wait→kill 链、task_reference/done_callback
   生命周期、Semaphore 作用域、健康等待全部核对；探针排除了 3.14 下
   `_readchunk` 忙等嫌疑；无死等/任务泄漏。
5. **模糊**：非法 UTF-8、YAML 日期标量、深嵌套、非字符串键等异常路径与上游同构。

13 条 minor/nit **逐条标注「上游固有，非重构引入」**（台账见 §6-U*），
无一条属于重构新增安全缺陷，故不阻断对等判定。

## 5. 本轮修复（fail→fix→回归，全部完成）

| # | 来源 | 问题 | 修复 | 回归 |
|---|---|---|---|---|
| F1 | 架构 minor[layering] | `resolve_extends` 在 merge.py 内以函数内惰性 import 反向取 normalize，与 normalize→merge 顶层导入构成规范层惰性环 | 函数整体下沉 normalize.py（本属规范化流程），同模块直用 rec_subs/normalize_service，仅单向 import merge 的 rec_merge/load_yaml_or_die；merge.py 删函数/惰性 import/失用的 `import os`；engine 与测试导入同步 | 940 passed；`grep normalize merge.py` 无 import；符号脚本仍 142/3 |
| F2 | 架构 minor[py314] | logs.py `_task_cancelled` 残留 `sys.version_info>=(3,11)` 死分支（requires-python≥3.14） | 简化为 `bool(task.cancelled() or task.cancelling())`，删失用 `import sys` | 940 passed；mypy/ruff 0 |
| F3 | 架构 nit[layering] | README Mermaid 图有幻影边 CMD→ENG、漏画 ENG→TR 与 TR→runner 异常边、model 误归执行层 | 按 108 条真实静态边重画架构图并补文字说明（model 归数据层、虚线标注「仅再导出 CalledProcessError」） | 人工核对 grep |
| F4 | 架构 nit[gpl] | README 写「95 个 .py」，scripts/symbol_inventory.py 加入后实为 96 | 改为「src/tests/scripts 全部 96 个」 | `find ... | wc -l`=96，缺头 0 |

回归证据：WSL py3.14 **940 passed / 0 skip / 74 subtests**，覆盖率 **92%**；
Windows py314 ruff(src+tests+scripts) 0、mypy src 0 issue（39 源文件）；
`symbol_inventory.py` 输出 142 符号 / 3 差异，与 parity.md 一致。

## 6. 发现台账（闭环判据）

### 6.1 经评估保留（重构有意决策，非缺陷）

- **R1 argparse.Namespace DTO 渗透（架构 minor，保留）**：commands/engine/pull/translate
  共 15 文件用 `argparse.Namespace` 作参数袋类型。审查确认 parser 机械（add_argument/
  parse_args/argv）完全封在 cli，下层只读 Namespace 属性、无任何解析；Handler 契约
  `Callable[[Any, Namespace], Awaitable]` 为 T7 显式设计（parity M6）。改为中立
  `Args` Protocol 触及 15 文件、940 测试，收益边际、回归面大，**本期保留**；
  后续若要让纯库脱离 argparse 类型，可在 types.py 引入 Protocol 并由 cli 适配，列入 backlog。
- **N1/N2** `_task_cancelled` 下划线名跨 2 命令模块、4 个 BASE 模块缺 `__all__`
  （架构 nit）：实证无符号泄漏、行为等价（D9 已登记），纯封装形式，本期不动以控范围。
- **N3 ruff 版本地板不封顶**：不同 ruff 版本对 UP038 默认集有漂移；建议 CI pin
  `ruff==0.15.*`（工程化 backlog，非代码缺陷）。

### 6.2 上游固有问题（对等重构不修改，列上游回馈候选）

按 translate-then-move 纪律，修这些会偏离 1.6.0 行为、破坏对等快照，故保持原样并登记：

- U1 `--verbose` 下完整 argv（含 `-e` 明文密钥）入 INFO 日志（security，最有回馈价值，建议脱敏）。
- U2 `rec_deps` 对不含起点的子环（A→B→C→D→B）剪枝失效，可 RecursionError（integrity）。
- U3 `include` 无 visited 去重：自包含无限追加、菱形包含重复 extend list（integrity）。
- U4 负数/0 `scale` 产生零容器后下游 IndexError 裸栈；`--scale svc` 无等号同（boundary）。
- U5 `config_hash` 的 json.dumps 无 `default=str`，YAML 日期标量/残留标签 TypeError（fuzz）。
- U6 `existing_containers` 对非数组 JSON/空 Names 缺防御（boundary）。
- U7 `COMPOSE_PARALLEL_LIMIT=0` → Semaphore(0) 永久挂起（boundary）。
- U8 命令执行期 podman 二进制中途消失无 FileNotFoundError 守卫（仅版本探测有）（boundary）。
- U9 compose 文件非法 UTF-8 抛 UnicodeDecodeError 裸栈（fuzz）。
- U10 `PODMAN_COMPOSE_POD_ARGS` 环境变量为字符串时与 list 拼接 TypeError（boundary）。
- U11 `_resolve_context_dependencies` 缺失服务先 KeyError，其后 None 检查为死代码（integrity；
  T9 测试按真实行为断言 KeyError 路径）。
- U12 bind/secret/extends 支持绝对路径/`..`/`~`（security，**设计如此**：compose 规格即允许挂载宿主路径，argv 无 shell 语义，逃逸边界归 podman）。
- U13 runner 取消后不重抛 CancelledError、不 await 日志 drain 任务（timing，上游刻意的 abort 语义）。

以上均不影响「重构相对上游行为等价」的结论；建议未来以独立 PR 向上游回馈 U1-U3，
本库若修复须同步偏离登记并在快照/测试中显式标注，不得在对等重构中夹带。

## 7. 各验收标准（AC）证据索引

| AC | 结论 | 主要证据 |
|---|---|---|
| AC-3 符号映射 | pass | parity.md §1-3，142/142 映射，脚本可复现 |
| AC-5 27 测试移植 | pass | T9 TR-9.2，484≥478，skip=0，拆分守恒 |
| AC-6 vendor 零改动 | pass | TR-11.1 porcelain 空、HEAD=e3df104 |
| AC-9 分层单向 | pass（5/5） | 108 import 零反向边、F1 环已消除、R1 判据 |
| AC-10 行为等价 | pass | 抽查深读无偏差、argparse 快照零差异、U* 全上游固有 |
| AC-11 py3.14 现代化 | pass | 178/178 公开 API 有注解、无 future/旧 typing/3.9 分支；F2 死分支已删；mypy 0 |
| AC-12 库无副作用 | pass | 运行时实测 import 仅加载自身；argv 单点；子进程边界唯一 |
| AC-2/4/7/8 | pass | 见 T1-T10 tasks.md 各 TR（24 命令 help 退出 0、零 skip、覆盖率 92%、ruff/mypy 0） |

## 8. 审查员独立性与方法声明

三个 reviewer 由编排者在不传递实施上下文/结论倾向的情况下并行派发，分别独立通读源码、
运行 grep/AST/运行时探针与测试，独立给出 blocker/major/minor/nit 定级；其中对等与
五攻击者两份报告明确判定 pass（0 重构新增 blocker/major），架构报告初评 4/5 并
指出 F1/F2，F1 已修复复评、F2 经判据保留。修复由实施者完成后，以 940 测试 +
ruff/mypy + 符号脚本回归验证，未让任何 reviewer 为让结论变绿而放宽标准。
