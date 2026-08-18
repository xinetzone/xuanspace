# 信任与保鲜

OKF v0.2 为每个概念维护**信任等级**（Trust Tier）与**保鲜状态**（staleness），用于回答「这条知识有多可信」与「这条知识是否已过期」。对应规范 §5。

## 信任等级

`TrustTier` 是三级枚举：

| 等级 | 取值 | 含义 |
|------|------|------|
| `UNVERIFIED` | `unverified` | 未验证 |
| `MACHINE_CONFIRMED` | `machine_confirmed` | 仅机器确认 |
| `HUMAN_REVIEWED` | `human_reviewed` | 经人工审阅 |

### 推导规则

`derive_trust_tier` 依据 frontmatter 的 `verified` 字段推导信任等级：

1. **无 `verified` 字段** → `unverified`
2. **`verified` 非空，但所有条目的 `by` 均不以 `human:` 开头** → `machine_confirmed`
3. **至少一个条目的 `by` 以 `human:<id>` 开头** → `human_reviewed`

`by` 字段的 `human:` 前缀用于标记「此验证由人类完成」，例如 `human:alice`；不含 `human:` 前缀的 `by`（如 `metric-collector`）被视为机器验证。

### 判定优先级

- `verified` 为空列表，或所有条目 `by` 均为空 → `unverified`
- 只要存在一个 `human:` 开头的 `by` → `human_reviewed`（短路判定）
- 否则 → `machine_confirmed`

## 保鲜状态

`is_stale` 依据 frontmatter 的 `stale_after` 字段判定保鲜状态：

| 条件 | 结果 |
|------|------|
| 无 `stale_after` 字段 | 永不过期（`False`） |
| `stale_after` 无法解析为日期 | 不过期（`False`） |
| `今天 >= stale_after` | 已过期（`True`） |
| `今天 < stale_after` | 保鲜中（`False`） |

## 状态字段

`status` 字段取值为 `draft` / `stable` / `deprecated` 三者之一，非法或缺省时默认为 `draft`。

## 来源与使用窗口

`parse_sources` 解析 frontmatter 的 `sources` 数组，每个来源可含：

| 字段 | 说明 |
|------|------|
| `resource` | 来源资源标识 |
| `id` | 来源 ID |
| `title` | 来源标题 |
| `author` | 作者 |
| `usage_count` | 使用次数 |
| `last_modified` | 最后修改日期 |

`usage_window` 提供使用窗口（`from` / `to` 两个日期）。

## 委派到插件

在 CLI 中，信任推导由 `TrustDeriver` 插件承载，它消费 `bundle_accessor` 服务、提供 `trust_analyzer` 服务。`okf trust` 命令即调用 `trust_analyzer`，对单概念返回 `{"trust_tier": ..., "is_stale": ...}`，对全 Bundle 返回以概念 ID 为键的同构字典。