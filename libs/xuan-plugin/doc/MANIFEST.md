# 原子化清单（MANIFEST）

- 源文件（归档冻结）：`/media/pc/data/ai/.agents/archive/xuanspace-pluginization-design-20261123/raw/xuanspace-pluginization-design.md`
- 源文件 SHA-256：`21c021fdda1ef2d5c1758495cbe643b75f83acf49f8887138e673e1c9a8a0419`
- 源文件体量：60,008 字节 / 759 行
- 原子化日期：2026-11-23
- 切分规则：按 `## N.` 章节边界切页，覆盖无空隙、无重叠；页尾 `---` 分隔线不入页
- 原位置：`/media/pc/data/ai/xuanspace-pluginization-design.md` 已于 2026-11-23 删除（正文由本目录承载）
- 冻结：原件逐字节存入 `.agents/archive/xuanspace-pluginization-design-20261123/raw/`，`MANIFEST.sha256` 已校验；本目录不复制原件

## 页清单

| 文件 | 原文节 | 行区间 | 非空正文行 | SHA-256 |
|---|---|---|---|---|
| `00-tldr.md` | §0 | L13–L38 | 17 | `466efeddd693d14143cebee10cf69e226a1f2a402828f4d244385b2427c5bb61` |
| `01-facts.md` | §1 | L39–L104 | 55 | `3d00b301ffe3f64dde6128a0fa480a2ba901c25b91a271377e28a509e0804648` |
| `02-insights.md` | §2 | L105–L143 | 26 | `6f6d0dcca628c53d9e84bbf9d988873d3aeed084e3378a11f9e1736d87475221` |
| `03-patterns.md` | §3 | L144–L208 | 51 | `1ab0ea8015eb83fd28b28e036cec86c476f35a98474ac5fb0c060464e58914dd` |
| `04-first-principles.md` | §4 | L209–L234 | 16 | `91ee1da408e19fa8a42ae595a387039113bd1441690fc4bbaf6df9127c899bd9` |
| `05-solution.md` | §5 | L235–L396 | 134 | `88a307ef1289b9ebc8e74875f05cbabc0319c5bbff718464227561cc1cf8fd21` |
| `06-adversarial-review.md` | §6 | L397–L413 | 12 | `cdecc794b41102bfe5d277994bc55c1396c2a23adf2b1d4b8f0e97de7ecc3727` |
| `07-roadmap.md` | §7 | L414–L451 | 31 | `6a259c49031887fc411382f305fb8113f2a627c734663eed3dbc992b4fa19d76` |
| `08-retrospective-p1-2.md` | §8 | L452–L631 | 138 | `99a71099c0da19e4d4aa577ba6a8b8d969c6d4a5568868cb653a58f91fbb718a` |
| `09-metrics.md` | §9 | L632–L648 | 12 | `788b6d4024efcca044f617688415c2925794882720cab94b6f47427fb8251b51` |
| `10-appendix.md` | §10 | L649–L759 | 93 | `cfd10fef8747e64897e3ce8d57094490e41e58c557ccb508e2c143a09aa0c950` |

## 守恒断言

- 源正文非空行（去 H1 与 `---`）：592
- 构成：导言 7 行 + 分卷 585 行
- 判定方式：「源正文 + 导言」与「全部页面正文」做 `Counter` 双向差集，missing / extra 均须为 0
- 实测：源 592 行，输出 592 行，missing 0，extra 0

## 第二部：复盘报告原子化（2026-11-23）

- 源文件：`.trae/specs/plugin-kernel-extraction/retrospective-20261121.md`（原子化后该独立文件已删除，正文由本目录承载）
- 源文件 SHA-256：`64060c3c0840d370237935f887e76862f14d728c848ce01410a588d756ab087f`
- 源文件体量：23,651 字节 / 320 行
- 切分规则：按 `## 一`～`## 七` 章节边界切页，覆盖无空隙、无重叠；页首 `>` 来源横幅与页尾 `---` 及导航行不入正文
- 链接改指：2 处指向 `tasks.md` 的同级相对链接，改写为仓库内 `.trae/specs/plugin-kernel-extraction/tasks.md`（相对路径失效修正，等价改写）
- 时点：2026-11-21（截至 `f42c37d`）；P2-10/P2-11 之后的现行口径见 [README.md](README.md) 第二部导言

| 文件 | 原文节 | 行区间 | 非空正文行 | SHA-256 |
|---|---|---|---|---|
| `11-retro-facts.md` | 一 | L11–L57 | 37 | `4dd6defa306a4bc73dc3f3086a9cf4ab9b2543a4970afb84d92de92c57fcf225` |
| `12-retro-insights.md` | 二 | L61–L93 | 23 | `54b8da7f1293930dac6e6d9a1a5cc40f228dd15646cbd856857a9d5a9a453c79` |
| `13-retro-patterns.md` | 三 | L97–L124 | 19 | `635d7c67e66cdfc8cac04616eac15c7cd331a42bee18b310699ba3a2a2622360` |
| `14-retro-adversarial.md` | 四 | L128–L139 | 10 | `ac513c1ee05c4d69ab9163ef1bb500f9ee543f5d6eaa8ae00505beaff7173c05` |
| `15-retro-delivery.md` | 五 | L143–L169 | 19 | `382d1d1bf8696263252086d7dfbe1515dcba5e73edcec10ce634c362a484fa8f` |
| `16-retro-changelog.md` | 六 | L173–L191 | 15 | `0c806df8cb76945f5508dff9a92d693412c3222aaf0fd3de6368ae9cfd0cb5c1` |
| `17-retro-appendix-a.md` | 七 | L195–L320 | 92 | `b91058194a7896804904291238e219d64e1daf38c4ff57da3fac5f512a7ef466` |

## 守恒断言（第二部）

- 源正文非空行（去 H1 与 `---`）：228
- 构成：导言 6 行 + 分卷 215 行 + 页间 `---` 7 行（不入页）
- 判定方式：同第一部，`Counter` 双向差集（先按上述规则归一化 2 处链接改指）
- 实测：源分卷 215 行 ↔ 输出 215 行，missing 0，extra 0
