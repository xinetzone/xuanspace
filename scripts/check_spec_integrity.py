#!/usr/bin/env python3
"""Xuanspace spec 任务台账完整性门禁（防并发写入导致的章节重复与半截合并）。

背景：`.trae/specs/<spec>/tasks.md` 是多轮会话反复追加的台账文件。工作区位于
      SMB/UNC 挂载时，编辑器缓冲区、多个会话与同步客户端可能同时写入同一文件，
      出现「同一任务编号出现两次」「任务块被拼接成半截」等静默损坏；这类损坏
      在写入当下不可见，只在提交前的 diff 里才暴露，容易把错误内容带进历史。

检查：对 `.trae/specs/*/tasks.md` 与 `.trae/specs/*/tasks/*.md`（台账分卷）断言
      1. `## Task <id>` 标题的编号在全部扫描文件中唯一（重复即失败，列出文件与行号）；
      2. 每个任务块内 `- **Status**:` 至多出现一次（重复说明块被拼接）；
      3. 文件不含 NUL 字节（截断或拼接损坏的典型特征）。

用法：python scripts/check_spec_integrity.py
退出码：0 全部通过；1 存在损坏；2 未找到任何 tasks.md（仓库结构异常）。
"""

import re
import sys
from pathlib import Path

TASK_HEADING = re.compile(r"^## Task ([A-Za-z0-9][A-Za-z0-9.\-]*)", re.MULTILINE)
STATUS_LINE = re.compile(r"^\s*-\s*\*\*Status\*\*:", re.MULTILINE)


def collect(path: Path, root: Path) -> tuple[list[str], dict[str, list[tuple[str, int]]]]:
    """解析单个台账文件，返回（本文件问题, 编号 -> [(相对路径, 行号)]）。"""
    rel = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    issues: list[str] = []

    if b"\x00" in raw:
        issues.append(f"{rel}: 含 NUL 字节，疑似截断或拼接损坏")

    text = raw.decode("utf-8", errors="replace")
    headings = list(TASK_HEADING.finditer(text))

    found: dict[str, list[tuple[str, int]]] = {}
    for match in headings:
        line_no = text.count("\n", 0, match.start()) + 1
        found.setdefault(match.group(1), []).append((rel, line_no))

    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[match.start() : end]
        status_count = len(STATUS_LINE.findall(block))
        if status_count > 1:
            line_no = text.count("\n", 0, match.start()) + 1
            issues.append(
                f"{rel}: 任务 {match.group(1)} 的块内出现 {status_count} 条 Status（行 {line_no}），疑似块被拼接"
            )

    return issues, found


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    spec_root = root / ".trae" / "specs"
    files = sorted(spec_root.glob("*/tasks.md")) + sorted(spec_root.glob("*/tasks/*.md"))
    if not files:
        print(f"未找到任何台账文件（{spec_root}），请检查仓库结构。")
        return 2

    issues: list[str] = []
    seen: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for path in files:
        spec = path.relative_to(spec_root).parts[0]
        file_issues, found = collect(path, root)
        issues.extend(file_issues)
        for task_id, locations in found.items():
            seen.setdefault((spec, task_id), []).extend(locations)

    for (spec, task_id), locations in sorted(seen.items()):
        if len(locations) > 1:
            located = "、".join(f"{rel}:{line_no}" for rel, line_no in locations)
            issues.append(f"[{spec}] 任务编号 {task_id} 重复出现（{located}），疑似并发写入合并或分卷重复")

    print(f"扫描到 {len(files)} 个台账文件（tasks.md 与 tasks/ 分卷），共 {len(seen)} 个任务编号")
    if issues:
        print(f"\n发现 {len(issues)} 处台账损坏：")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("全部通过 spec 台账完整性校验：任务编号唯一、任务块单一、无 NUL 字节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
