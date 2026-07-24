# 用户指南

本章节包含 Xuanspace 的完整使用指南。

## 子项目管理

### 创建新项目

```bash
# 创建纯 Python 库
xs new --type python my-lib

# 创建 C++ 原生扩展
xs new --type native my-ext

# 创建静态项目（HTML/JS 等）
xs new --type static my-site
```

### 查看项目列表

```bash
# 表格形式
xs list

# JSON 格式
xs list --json

# 按类型筛选
xs list --type python

# 按目录筛选
xs list --dir libs
```

### 检测受影响项目

当修改某个库后，查看哪些应用受影响：

```bash
xs affected
```

## 依赖管理

### 查看依赖状态

```bash
# 检查过时依赖
xs deps check

# 查看依赖树
xs deps tree

# 列出过时包
xs deps outdated
```

### 更新依赖

```bash
# 预览更新（不实际修改）
xs deps update --dry-run

# 更新所有依赖
xs deps update

# 只更新指定项目
xs deps update --project xuan-core

# 限制更新类型
xs deps update --type minor
```

### Python 兼容性检查

```bash
# 检查所有依赖的 Python 3.13 兼容性
xs py-compat

# JSON 输出
xs py-compat --json

# 检查特定 Python 版本
xs py-compat --py 3.13
```

## 版本管理

### 查看版本

```bash
# 查看所有子项目版本
xs version show

# 查看指定项目
xs version show xuan-core
```

### 版本 bump

```bash
# 主版本号 bump（破坏性变更）
xs version bump xuan-core --type major

# 次版本号 bump（新功能）
xs version bump xuan-core --type minor

# 修订版本号 bump（Bug 修复）
xs version bump xuan-core --type patch
```

## 构建

### 构建项目

```bash
# 构建所有项目
xs build

# 构建指定项目
xs build --project xuan-ext-demo

# 按类型构建
xs build --type native
```

## 工具链管理

### 检查工具链

```bash
xs toolchain check
```

### 安装工具

```bash
# 通过 pip 安装 CMake
xs toolchain install cmake

# 安装 Ninja
xs toolchain install ninja
```

### 列出工具

```bash
xs toolchain list
```

## 文档管理

### 构建文档

```bash
xs docs build
```

### 预览文档

```bash
# 默认端口 8000
xs docs serve

# 指定端口
xs docs serve --port 9000
```

### 清理构建产物

```bash
xs docs clean
```

### 检查链接

```bash
xs docs linkcheck
```

## 元数据管理

### 初始化元数据目录

```bash
xs meta init
```

### 扫描元数据状态

```bash
xs meta scan
```

### 验证 frontmatter 合规性

```bash
xs meta validate

# 自动修复
xs meta validate --fix
```

### 同步元数据

```bash
xs meta sync
```

## 环境诊断

```bash
xs doctor
```

输出示例：

```
Python:       3.13.2  ✓
PDM:          2.22.0  ✓
uv:           未安装  ⚠
pip:          24.0    ✓
CMake:        3.30.0  ✓
Ninja:        1.12.0  ✓
Sphinx:       8.0.0   ✓
```

## 子模块与依赖更新

```bash
# 更新所有 git 子模块
xs update --submodules

# 更新所有 pip 依赖
xs update --deps

# 同时更新子模块和依赖
xs update --all
```