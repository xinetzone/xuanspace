# {{name}}

{{name}} 是 Xuanspace（玄境）Python monorepo 中的静态/非Python子项目。

## 项目说明

静态项目类型用于放置不需要 Python 运行环境的纯前端、工具页面、演示页面等资源。

## 放置规则

- 可放置纯前端项目（HTML/CSS/JavaScript/TypeScript）
- 可放置静态文档、工具页面、演示页面
- 可放置静态资源集合（图片、字体、配置文件等）
- **无需**配置 pyproject.toml
- **无需** Python 环境即可运行（直接在浏览器打开 index.html 即可）

## 目录结构

```
{{name}}/
├── index.html          # 入口页面（本模板提供）
├── README.md           # 项目说明文档
├── .gitignore          # Git 忽略配置
├── css/                # 样式文件（可选，自行创建）
├── js/                 # 脚本文件（可选，自行创建）
├── assets/             # 静态资源（可选，自行创建）
└── ...                 # 其他项目文件
```

## 快速使用

1. 直接在浏览器中打开 `index.html` 即可预览
2. 或使用任意静态文件服务器（如 `python -m http.server`、`npx serve` 等）

```bash
# 使用 Python 内置服务器（如已安装 Python）
python -m http.server 8000

# 使用 Node.js serve
npx serve .
```

## 维护状态

- **状态**: 开发中 (Alpha)
- **维护者**: Xuanspace Team
- **运行环境**: 任意现代浏览器
