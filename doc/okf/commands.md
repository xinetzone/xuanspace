# CLI 命令参考

`okf` 是一个基于 `argparse` 的命令行工具，提供全局选项与 6 个子命令。

## 全局选项

| 选项 | 说明 |
|------|------|
| `--version` / `-V` | 显示版本号 |
| `--help` | 显示帮助信息 |

## 命令总览

| 命令 | 说明 |
|------|------|
| `okf validate <path> [--strict]` | 一致性校验（§11） |
| `okf init <path>` | 创建 Bundle 骨架 |
| `okf index <path>` | 生成 / 更新 `index.md` |
| `okf inspect <path> [concept_id]` | 查看概念详情 |
| `okf trust <path> [concept_id]` | 信任等级与保鲜状态 |
| `okf list <path> [--type X] [--tag Y]` | 列出概念 |

所有命令都以 Bundle 路径 `<path>` 作为第一个位置参数。

## okf validate

对 Bundle 执行一致性校验，输出错误与警告报告；存在错误时退出码为 1。

```bash
okf validate <path> [--strict]
```

- 校验含严格项（错误）与宽松项（警告）两个层级，详见[一致性校验](conformance)。
- **`--strict` 说明**：该标记当前被解析但未参与校验逻辑——`validate` 始终执行完整的严格项 + 宽松项检查；`--strict` 作为预留选项存在。

## okf init

创建 Bundle 骨架目录结构：

```bash
okf init <path>
```

创建以下内容（已存在则跳过）：

- `index.md`（内容为 `# <目录名>`）
- `log.md`（内容为 `# Change Log`）
- 目录 `concepts/`、`playbooks/`、`references/`

## okf index

根据 Bundle 内的概念文件生成 / 更新 `index.md`：

```bash
okf index <path>
```

生成结果遵循 §8 结构（详见[索引与日志合成](synthesis)），写入 `<path>/index.md` 并同时打印到标准输出。

## okf inspect

查看 Bundle 概览或单个概念的详情：

```bash
okf inspect <path> [concept_id]
```

- **不带 `concept_id`**：打印 Bundle 根路径、概念数量，以及每个概念的 `<id> [<type>] <title>` 列表。
- **带 `concept_id`**：打印该概念的类型、标题、描述、标签、完整 frontmatter，以及正文前 5 行。概念不存在时输出错误并以退出码 1 结束。

## okf trust

输出概念的信任等级与保鲜状态：

```bash
okf trust <path> [concept_id]
```

- **不带 `concept_id`**：输出全部概念的 `信任等级` 与 `保鲜状态`。
- **带 `concept_id`**：只输出指定概念；概念不存在时输出错误并以退出码 1 结束。

输出格式：`信任等级=<tier>, 保鲜状态=<已过期|保鲜中>`。推导规则详见[信任与保鲜](trust)。

## okf list

按条件过滤并列出概念：

```bash
okf list <path> [--type X] [--tag Y]
```

| 选项 | 说明 |
|------|------|
| `--type` | 按 `type` 字段精确过滤 |
| `--tag` | 按 `tags` 是否包含指定标签过滤 |

输出格式为制表符分隔的 `<concept_id>\t<type>\t<title>`。