> 分卷来源：`tasks.md` 原文 L1775–L1816（2026-11-23 切分，逐字节保真；本行与下两行横幅不属原文）
> 上册：[../tasks.md](../tasks.md) · 上一卷：07-p2-c-invariants.md · 下一卷：（无）
> 主题：P2-D：设计文档按主题分组
## Task P2-19-1: 文档按主题分组（三目录 + 重编号 + 改链）

- **Status**: `completed`
- **目标**：把 `libs/xuan-plugin/doc` 的 20 篇平铺文档按主题归入 `design/`、
  `retro/`、`appendix/` 三组并重编号，降低定位成本且不丢失原文溯源。
- **改动**：
  - 新增目录：`design/`（00–07 设计论证，8 篇）、`retro/`（08–15 复盘与
    度量，8 篇）、`appendix/`（16–17 附录与基线，2 篇）；
  - 重编号：11→10、12→11、13→12、14→13、15→14、16→15、10→16；00–09 与 17 不变；
  - 改链：18 篇页首 README 回链与页尾上一篇/下一篇、5 处 `../../../` 上跳路径、
    1 处外部入链（`.trae/specs/plugin-kernel-extraction/tasks/07-p2-c-invariants.md`）；
  - 索引重建：`doc/README.md` 改为「主题分组总览 + 篇目索引 + 编号规则」，
    `doc/MANIFEST.md` 增改名映射表并重算现行指纹。
- **验收**：
  - Markdown 链接可达 109 条、断链 0；
  - 行级 diff 变更 78 行，全部命中页首横幅或页尾导航（含 `](`），正文违规 0；
  - 页尾导航链与新全局序 00→17 一致，违规 0；
  - `doc/MANIFEST.md` 18 行路径与 SHA-256 全一致；
  - `check_pyproject_style.py`、`check_declaration_consistency.py`、
    `check_dependency_direction.py`、`check_spec_integrity.py` 均 rc=0；
  - 全部文档 LF，CRLF 0。
- **Notes**:
  - **V1 被否方案一：仅 README 加主题视图、零移动**——物理结构不变，定位成本
    下降有限。
  - **V2 被否方案二：按原「第一部/第二部」物理切分**——那是来源维度，09 度量与
    10 附录的主题归属仍混杂。
  - **V3 被否方案三：按七概念机械切 6 目录**——多数目录仅 1–2 篇，目录开销大于
    收益，且 05 方案、16 更新记录不属单一概念。
  - **V4 被否方案四：按类型切 index/design/retro/appendix**——`design` 内仍 8 篇
    平铺，定位成本未实质下降。
  - **V5 代价与对冲**：重编号使 `doc/MANIFEST.md` 每页 SHA-256 失效；已改为
    「现行指纹」口径，保留原文行区间与非空行数作溯源，正文守恒以行级 diff 断言复验。
  - **V6 页码与正文小节号脱钩**：文件名编号为阅读序，正文小节号仍为原文 § 号
    （如 `## 10. 附录` 现为 16 号页），已在 `doc/README.md`「编号规则」写明。
  - **V7 边界**：正文一字未改；未动 `doc/` 以外文档；外部入链仅改 1 行；本轮
    台账追加 P2-19 并重建索引，未纳入他人改动。

---

> 复盘报告（P2-4～P2-11 全链，R → I → E → C，G1–G4 通过）：
> 已原子化并入 [libs/xuan-plugin/doc/README.md（第二部）](../../../../libs/xuan-plugin/doc/README.md)，原独立文件已删除。

