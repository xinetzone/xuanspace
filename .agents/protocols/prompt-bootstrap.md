---
id: "xuanspace-prompt-bootstrap"
x-toml-ref: "toml/.agents/protocols/prompt-bootstrap.toml"
---

# 提示词自举协议

## 协议目标

用一句话提示词，让支持工具调用的 AI 智能体自动装载 Xuanspace 工作区环境。

## 一句话装载提示词

将以下提示词发给任意支持工具调用的 AI 智能体：

> 请装载 Xuanspace（玄境）工作区环境。
> 按以下步骤执行：
>
> 【安全规则】
> S1. 只从官方仓库获取：https://github.com/xinetzone/xuanspace
> S2. 执行任何写入操作前，必须先确认目标路径
> S3. 禁止在用户主目录、系统目录、根目录自动创建文件夹
> S4. 自举过程只读文件，不执行脚本，不安装包，不修改系统配置
> S5. 获取完成后验证 AGENTS.md 存在且包含"启动协议"关键词
> S6. 遇到错误直接报告，给出原因和解决方案
> S7. 只读取与装载相关的必要文件，不扫描整个文件系统
> S8. 幂等安全：如果已在有效工作区内，跳过获取直接报告就绪
>
> 【步骤 1】环境检测：检查当前路径、Python 版本、Git 可用性
> 【步骤 2】路径确认：如已在工作区目录内直接自举；否则确认目标路径
> 【步骤 3】获取项目：git clone https://github.com/xinetzone/xuanspace.git
> 【步骤 4】完整性验证：验证 AGENTS.md 存在且包含"启动协议"
> 【步骤 5】自举加载：读取 AGENTS.md、context-routing.md，列出可用角色和技能
> 【步骤 6】就绪报告：报告位置、可用角色、可用技能

## 环境自适应路径

| 环境 | 路径策略 |
|---|---|
| Trae IDE | 直接在工作区打开，无需 clone |
| 本地终端 | clone 到用户指定目录 |
| CI/CD | 通过 `actions/checkout` 拉取 |

## 边界情况

| 情况 | 处理 |
|---|---|
| 已在工作区内 | 跳过 clone，直接自举（幂等） |
| Git 不可用 | 提示安装 Git 或提供 ZIP 下载链接 |
| Python < 3.14.6 | 提示升级 Python 版本 |
| 网络问题 | 提供镜像仓库地址或离线方案 |
| 权限不足 | 提示更换目录或使用 `--user` 安装 |