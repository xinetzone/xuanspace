# vendor/ - 第三方依赖目录

## 概述

`vendor/` 目录存放 Xuanspace（玄境）项目中纳入源码管理的外部第三方代码。这些代码通常需要进行本地修改（patch）以适配项目需求，或作为 git submodule 纳入管理。

## 与其他依赖管理方式的区别

| 方式 | 适用场景 | 目录位置 |
|------|----------|----------|
| **vendor/** | 需要本地修改/patch 的第三方库 | vendor/（作为 git submodule） |
| **projects/** | 第一方子项目（自己开发维护） | projects/ |
| **pip/PDM** | 纯使用不需修改的第三方依赖 | 通过 pyproject.toml 管理，不入库 |

## 准入标准

一个第三方代码库应放入 `vendor/` 当且仅当满足以下条件：

1. **需要本地修改/patch**：无法通过正常途径（如上游 PR、monkey patch）解决问题
2. **作为 git submodule 管理**：便于追踪上游更新和同步本地补丁
3. 纯使用不需修改的依赖一律用 pip/pdm 安装，不放入 vendor

## 常见使用场景

- 需要应用本地补丁（fork）的开源项目
- 上游有 bug 需要临时修复，等待上游合并
- 需要添加项目特定功能且无法通过扩展实现

## 管理方式

vendor 目录下的第三方库**必须**作为 **git submodule** 进行管理：

```bash
# 添加第三方库作为 submodule
git submodule add <upstream-repo-url> vendor/<library-name>

# 初始化已有的 submodules
git submodule update --init --recursive

# 更新 submodule 到最新版本
cd vendor/<library-name>
git fetch
git checkout <new-version-tag>
cd ../..
git add vendor/<library-name>
```

## 修改与补丁管理

每个 vendor 库的本地修改应通过以下方式管理：

1. 在 submodule 内创建本地分支（如 `xuan-patches`）
2. 所有本地补丁提交到该分支
3. 在库目录下创建 `PATCHES.md` 记录每个补丁的说明：

```markdown
# 本地补丁记录

## Patch 1: 修复 Windows 路径问题
- 问题描述：上游代码使用 Unix 风格路径，Windows 下出错
- 修改文件：src/path_utils.py
- 上游状态：已提交 PR #123，等待合并
- 责任人：张三
- 创建日期：2026-01-15
```

## 命名规范

- 子目录直接使用**原项目名称**命名（保留原项目的命名风格）
- 不添加额外前缀或后缀

**正确示例**：
- `weasyprint/` - 保留原项目名 WeasyPrint
- `some-internal-lib/` - 内部库原名

## 注意事项

1. **许可证合规**：确保 vendor 代码的许可证与项目许可证兼容，并保留原版权声明
2. **最小化原则**：只 vendor 必要的库，不要复制整个仓库的无关文件
3. **定期同步**：定期检查上游更新，及时合并安全修复
4. **优先上游化**：本地修复应尽量向上游提交 PR，减少维护负担
5. **文档记录**：所有本地修改必须有清晰记录

## 现有子项目

（待实际添加后更新此表）

| 目录名 | 来源 | 版本 | 本地修改 | 说明 |
|--------|------|------|----------|------|
| `tvm-ffi/` | [apache/tvm-ffi](https://github.com/apache/tvm-ffi) | v0.1.12+ | 无 | Apache TVM 的 C++ Foreign Function Interface 库，提供跨语言 FFI 绑定 |
| `caffe/` | [daoflows/caffe](https://github.com/daoflows/caffe) | main | 有 | BVLC/caffe 的 fork，最小化 caffe protobuf 库，集成 tvm-ffi Python 绑定，提供 caffe-cpp-slim 无依赖核心 |
