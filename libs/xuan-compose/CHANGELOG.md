# Changelog

本项目版本号独立于上游；所有重构切片（T1-T12）均保持对
containers/podman-compose v1.6.0（commit
`e3df10472e194ab6d547b5ad25542c5c79e1a5fb`）的行为对等，差异见
[docs/parity.md](docs/parity.md)。

## 0.1.0 — 初始分层重构（T1-T10）

由 containers/podman-compose v1.6.0（GPL-2.0-only）翻译式分层重构为
Python ≥ 3.14 的 src 布局纯 Python 库。

### 切片交付

- **T1 脚手架与 GPL 基线**：scikit-build-core（纯 Python，无 cmake 段）、
  PEP 621 元数据、LICENSE 与署名、`__version__="1.6.0+xuan.1"`、SPDX 头。
- **T2 基础层**：compat / errors / types / logging_utils / interpolation / envfile。
- **T3 规范层**：normalize / merge（`!override`/`!reset` 标签全局注册等价保留）/ discovery。
- **T4 执行层**：runner（全包唯一子进程边界）/ model（StrEnum）/ dependencies / pull 前置。
- **T5 翻译层**：translate 子包 8 模块（mounts/networks/ports/resources/secrets/
  build/container_args/run_args），上游 11 个测试文件 256 例。
- **T6 引擎层**：ComposeEngine（重命名自 PodmanCompose）compose 文件解析与状态装配。
- **T7 命令层**：commands 子包 24 个 handler 与 COMMAND_HANDLERS 显式注册表
  （install_handlers），替代 @cmd_run 装饰器副作用。
- **T8 CLI 层**：cli/parser.py（COMMAND_PARSERS 24 键 + PullPolicyAction +
  23 parser 函数）、cli/main.py 显式装配链、`python -m xuan_compose` 与
  `[project.scripts] xuan-compose`；TR-8.3 argv 读取单点。
- **T9 全量对等门禁**：argparse 参数表快照对上游零差异（11 维，prog 除外）；
  上游 26 个单元测试文件全量移植（484 ≥ 478，skip=0）；覆盖率整体 91%、
  核心模块 interpolation/normalize/merge/engine/translate 全部 ≥90%；
  WSL py3.14 940 passed。
- **T10 对等审计与文档**：142 个上游公开符号 100% 映射（docs/parity.md）、
  py3.14 差异逐条登记、README 完稿（库 API/CLI/架构图/测试/集成验证）。
- **T11 独立对抗审查**：3 个 fresh-context reviewer（对等/架构/五攻击者）结论
  全 pass、0 blocker/major；修复 4 项重构引入问题（resolve_extends 下沉消除
  规范层惰性环、删 3.11 版本守卫死分支、架构图校正、SPDX 计数），审查报告见
  docs/review.md；13 条上游固有健壮性问题登记为上游回馈候选（不在对等重构中夹带修改）。

### 主要架构差异（行为等价）

- 消除模块级全局单例与导入期 argv 读取：显式构造 `ComposeEngine()`、
  CLI 装配期单点注入 `engine.executable`。
- 装饰器副作用注册 → 显式注册表（COMMAND_HANDLERS / COMMAND_PARSERS）。
- 分层单向依赖：runner ← 规范/翻译层 ← engine ← commands ← cli。
- 全面 Python 3.14 惯用法（StrEnum、PEP 649 惰性注解、完整类型注解）。
- 容器标签版本标识为 `1.6.0+xuan.1`。

### 保留的上游怪癖（未顺手"修正"）

退出码 -1、`str_to_seconds` 正则怪癖、sub_dir 路径不剥 `./`、
PullPolicyAction 静默分支、up/down 编排中的非标准输出与分支细节等，
逐条在 docs/parity.md §5.3 登记并由测试固化。
