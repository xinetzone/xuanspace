> 原子化来源：`retrospective-20261121.md` 原文 §六（L173–L192）；该独立文件已于 2026-11-23 并入本目录并删除，唯一权威为本目录
> 文档索引：[README.md](../README.md)

## 六、更新记录

| 版本 | 日期 | 触发 | 变更 | 事实复核 |
|:--:|:--:|------|------|------|
| v1.0 | 2026-11-21 | 「全面复盘 + 洞察 + 更新报告」首次执行 | 建立 R/I/E/V/C 五节，登记 28 条事实、4 条洞察、1 个模式 + 4 条反模式、6 条异议 | 事实全部来自当轮实测 |
| v1.1 | 2026-11-21 | 同一指令第二次执行（增量复核） | 补版本元信息与本表；在 [tasks.md](../../../../.trae/specs/plugin-kernel-extraction/tasks.md) 末尾补指针；未改动 R/I/E 结论 | 复核 F21–F26 常量：`_INACTIVE = "__INACTIVE__"`、`_DIGEST_LEN = 16`、`_ADDRESS_RE = 0x[0-9a-fA-F]{8,}`、`_MAX_FLUSH_ROUNDS = 100` 与正文一致；**零修订** |
| v1.2 | 2026-11-21 | 「设计文档合并」指令（A 概念：原子化） | 新增「七、附录 A」：按源文章节切为 A1–A12 十二个原子块，逐块给出判据与实现对照；补合并锚点（547 行 + md5）与失效条件 | 复核设计文档：事实自称 30 条、实际编号 34 条；状态图 11 边、代码 15 边；ADR-05 未落地（`providers[0]` 仍在） |

### 复核方式（可重跑）

```bash
git log --format='%h %s' --shortstat f6fd5dd^..f42c37d      # F1–F7
grep -n "_INACTIVE =\|_DIGEST_LEN =\|_ADDRESS_RE =\|_MAX_FLUSH_ROUNDS =" \
  libs/xuan-plugin/src/xuan_plugin/{plugin,context}.py         # F23–F24
python -m pytest -q tests libs/xuan-plugin/tests libs/xuan-build/tests libs/xuan-ffi/tests   # F10
cd tools/okf && python -m pytest -q                            # F11
```

> 复核纪律：本报告只记录可重复命令的输出，不接受「与上次一致」的口头结论；若复跑数字与正文不符，以实测为准并递增版本。

---

相关页面：[文档索引](../README.md) · 上一篇：[C — 交付与未决项（G4）](14-retro-delivery.md) · 下一篇：[附录](../appendix/16-appendix.md)
