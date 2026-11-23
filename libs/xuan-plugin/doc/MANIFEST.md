# 原子化清单（MANIFEST）

- 源文件（归档冻结）：`/media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md`
- 源文件 SHA-256：`21c021fdda1ef2d5c1758495cbe643b75f83acf49f8887138e673e1c9a8a0419`
- 源文件体量：60,008 字节 / 759 行
- 原子化日期：2026-11-23
- 切分规则：按 `## N.` 章节边界切页，覆盖无空隙、无重叠；页尾 `---` 分隔线不入页
- 原位置：`/media/pc/data/ai/xuanspace-pluginization-design.md` 已于 2026-11-23 删除（正文由本目录承载）
- 冻结：原件逐字节存入 `.agents/archive/xuanspace-pluginization-design-20261123/raw/`，`MANIFEST.sha256` 已校验；本目录不复制原件

## 主题分组与重编号（2026-11-23）

- 分组依据：按主题归入 `design/`、`retro/`、`appendix/` 三组，见 [README.md](README.md)「主题分组总览」。
- 重编号：文件名编号改为分组后的全局阅读序；正文小节号仍为原文 § 号，未改动。
- 正文守恒：本次只改页首横幅与页尾导航行（链接目标；跨组时为对齐新阅读序一并调整导航标签），正文行零改动；以行级 diff 断言「变更行均含 `](`」复验。
- 指纹口径：下表 SHA-256 为**分组改链后的现行指纹**；原子化时刻的旧指纹已随链接改指失效，不再保留。

| 旧路径 | 新路径 |
|---|---|
| `00-tldr.md` | `design/00-tldr.md` |
| `01-facts.md` | `design/01-facts.md` |
| `02-insights.md` | `design/02-insights.md` |
| `03-patterns.md` | `design/03-patterns.md` |
| `04-first-principles.md` | `design/04-first-principles.md` |
| `05-solution.md` | `design/05-solution.md` |
| `06-adversarial-review.md` | `design/06-adversarial-review.md` |
| `07-roadmap.md` | `design/07-roadmap.md` |
| `08-retrospective-p1-2.md` | `retro/08-retrospective-p1-2.md` |
| `09-metrics.md` | `retro/09-metrics.md` |
| `10-appendix.md` | `appendix/16-appendix.md` |
| `11-retro-facts.md` | `retro/10-retro-facts.md` |
| `12-retro-insights.md` | `retro/11-retro-insights.md` |
| `13-retro-patterns.md` | `retro/12-retro-patterns.md` |
| `14-retro-adversarial.md` | `retro/13-retro-adversarial.md` |
| `15-retro-delivery.md` | `retro/14-retro-delivery.md` |
| `16-retro-changelog.md` | `retro/15-retro-changelog.md` |
| `17-retro-appendix-a.md` | `appendix/17-retro-appendix-a.md` |

## 页清单（设计源，2026-11-23 原子化）

| 文件 | 原文节 | 行区间 | 非空正文行 | SHA-256 |
|---|---|---|---|---|
| `design/00-tldr.md` | §0 | L13–L38 | 17 | `2ce2d50660d0e42c793ee048da9b60f88758173d0520c3544a99ba9e31eacc5b` |
| `design/01-facts.md` | §1 | L39–L104 | 55 | `d678caf050299448dfbd5b588a8c3a9552f1ae13339fe0c6fdae0e1a768e0c7b` |
| `design/02-insights.md` | §2 | L105–L143 | 26 | `c9b88e9ad3ca8dbc1c71830dd82b15a1d6549af2b1fdd9c027b3af53c485d479` |
| `design/03-patterns.md` | §3 | L144–L208 | 51 | `357c5a71dbe46ecb10929eabeefd3a242302de116789f2239d2cc7040282301a` |
| `design/04-first-principles.md` | §4 | L209–L234 | 16 | `b9c9025868f181307bf3f995a7d6927dbe17a0d301b3821a233f2043edc2405d` |
| `design/05-solution.md` | §5 | L235–L396 | 134 | `9addd9401f7afa99bb1ff432c7d6ed97ae805d574948e2935aefb78b454c50f7` |
| `design/06-adversarial-review.md` | §6 | L397–L413 | 12 | `aea879b11b3027eb08ceb0ae2a541ec3a716ae424937d2529f89add5d9937a57` |
| `design/07-roadmap.md` | §7 | L414–L451 | 31 | `33a3dbad055e65be9f0dc593b7d5aecd2ed09d938ff858ac7e11aac764d1d5c2` |
| `retro/08-retrospective-p1-2.md` | §8 | L452–L631 | 138 | `0fa696cfea4f84ba35128c6098e8a38166ed39485ecbd9d5a6c2a9a4731c3b10` |
| `retro/09-metrics.md` | §9 | L632–L648 | 12 | `43a91749642b9f1cbb4745c5c7e1a241d1be344130d4f237a85e4a20dd532dce` |
| `appendix/16-appendix.md` | §10 | L649–L759 | 93 | `446091b26b65fd1549ef00d061797e896f7955bb4bb26b86668f910a634921be` |

## 守恒断言

- 源正文非空行（去 H1 与 `---`）：592
- 构成：导言 7 行 + 分卷 585 行
- 判定方式：「源正文 + 导言」与「全部页面正文」做 `Counter` 双向差集，missing / extra 均须为 0
- 实测：源 592 行，输出 592 行，missing 0，extra 0
- 分组后复验：正文行未改动，仅链接目标变更，上述守恒结论维持

## 第二部：复盘报告原子化（2026-11-23）

- 源文件：`.trae/specs/plugin-kernel-extraction/retrospective-20261121.md`（原子化后该独立文件已删除，正文由本目录承载）
- 源文件 SHA-256：`64060c3c0840d370237935f887e76862f14d728c848ce01410a588d756ab087f`
- 源文件体量：23,651 字节 / 320 行
- 切分规则：按 `## 一`～`## 七` 章节边界切页，覆盖无空隙、无重叠；页首 `>` 来源横幅与页尾 `---` 及导航行不入正文
- 链接改指：2 处指向 `tasks.md` 的同级相对链接，改写为仓库内 `.trae/specs/plugin-kernel-extraction/tasks.md`（相对路径失效修正，等价改写）
- 时点：2026-11-21（截至 `f42c37d`）；P2-10/P2-11 之后的现行口径见 [README.md](README.md) 第二部导言

| 文件 | 原文节 | 行区间 | 非空正文行 | SHA-256 |
|---|---|---|---|---|
| `retro/10-retro-facts.md` | 一 | L11–L57 | 37 | `5d216da6c85bc2bedc74e996d9e83597c2ae1b497f4f477cc5244df5e833ea56` |
| `retro/11-retro-insights.md` | 二 | L61–L93 | 23 | `90608b6c71477ec22d814ef62ab2faa7ecd98aef00cfb3ef6c67b19b2ac12c8b` |
| `retro/12-retro-patterns.md` | 三 | L97–L124 | 19 | `356f0ecd5345bf4cb3325ede4d88387f72f5e2c33f076b8e08cec1cdfad85b80` |
| `retro/13-retro-adversarial.md` | 四 | L128–L139 | 10 | `deb6d0a58db67d0b40e85c520910ef7c7151523e6d25f1231d7368adb99679cc` |
| `retro/14-retro-delivery.md` | 五 | L143–L169 | 19 | `a62de148ade55abefeb93778552683e555efef764c25cbea1aaa03de3f009467` |
| `retro/15-retro-changelog.md` | 六 | L173–L191 | 15 | `28d7a6a56d456bb04b76ad677208328a893457366096df85f9b47f5799ba713b` |
| `appendix/17-retro-appendix-a.md` | 七 | L195–L320 | 92 | `1d4b200f99602c8007562abb8f388edf34b8e4c7bce0265686d7d930fcabc274` |

## 守恒断言（第二部）

- 源正文非空行（去 H1 与 `---`）：228
- 构成：导言 6 行 + 分卷 215 行 + 页间 `---` 7 行（不入页）
- 判定方式：同第一部，`Counter` 双向差集（先按上述规则归一化 2 处链接改指）
- 实测：源分卷 215 行 ↔ 输出 215 行，missing 0，extra 0
- 分组后复验：正文行未改动，仅链接目标变更，上述守恒结论维持
