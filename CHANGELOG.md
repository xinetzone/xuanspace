# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**：`xs docs` 子命令改名为 `xs doc`，按裁定不保留 `docs` 别名
- **BREAKING**：`xs init` 脚手架现创建 `doc/` 目录（原为 `docs/`），模板文本同步更新为 `doc/`
- 内部重构：`commands/docs_cmd.py` 重命名为 `commands/doc_cmd.py`，函数统一为 `doc_*` 前缀（不改变用户可见接口）

## [0.1.0] - 2026-07-23

### Added
- 初始版本发布
- xs CLI 工具（list/doctor/init/build/new/update/py-compat/deps/version/docs/meta/toolchain）
- Monorepo 目录结构（apps/libs/vendor/tools/docs/scripts/attic/.agents/.meta）
- Sphinx + MyST Markdown 文档系统
- CMakePresets.json 跨平台构建预设
- AGENTS.md AI 协作规范
- Python 3.14.6+ 严格兼容
- 多包管理器支持（PDM/uv/pip）
- GitHub Actions CI 配置
