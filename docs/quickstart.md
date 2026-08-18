# 快速开始

本指南帮助你在 5 分钟内完成 Xuanspace 的安装与首次使用。

## 前置条件

| 工具 | 版本要求 | 必需 | 说明 |
|---|---|---|---|
| Python | 3.14.6+ | ✅ 必需 | 不满足版本会给出明确错误提示 |
| Git | 2.x+ | ✅ 必需 | 用于克隆仓库和子模块管理 |
| PDM | 2.x+ | 推荐 | 提供 workspace 自动链接 |
| uv | 0.4+ | 可选 | 快速安装替代方案 |
| pip | 24+ | 备选 | Python 自带，标准方式 |
| CMake | 3.26+ | 按需 | 仅构建 C++ 原生扩展时需要 |
| Ninja | 1.11+ | 按需 | 仅构建 C++ 原生扩展时需要 |

## 安装

### 1. 克隆仓库

```bash
git clone --recurse-submodules https://github.com/xinetzone/xuanspace.git
cd xuanspace
```

### 2. 安装依赖

选择一种包管理器安装：

**PDM（推荐）** — 自动 workspace 链接：

```bash
pdm install
```

**uv（快速）** — 安装速度最快：

```bash
uv pip install -e ".[dev]"
```

**pip（标准）** — Python 自带，无需额外安装：

```bash
pip install -e ".[dev]"
```

### 3. 验证安装

```bash
xs --help
xs --version
```

## 初次使用

### 查看所有子项目

```bash
xs list
```

输出示例：

```
  apps/
    (无)
  libs/
    xuan-core           python    0.1.0
    xuan-ext-demo       native    0.1.0
  tools/
    xs-cli              python    0.1.0
```

### 检查开发环境

```bash
xs doctor
```

该命令会报告 Python 版本、包管理器可用性、CMake/Ninja 版本、Sphinx 版本等。

### 创建你的第一个项目

```bash
# 创建纯 Python 库
xs new --type python my-first-lib

# 查看创建结果
xs list
```

### 构建文档

```bash
# 构建 HTML 文档
xs docs build

# 启动本地预览（默认端口 8000）
xs docs serve
```

## 下一步

- 查看 [架构设计](architecture) 了解项目整体设计
- 查看 [构建系统](build-system) 了解构建工具链
- 查看 [CLI 参考](cli/index) 了解所有可用命令
- 查看 [贡献指南](contributing) 参与项目开发