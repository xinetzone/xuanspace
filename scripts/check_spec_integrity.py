#!/usr/bin/env python3
"""Xuanspace spec 任务台账完整性门禁（防并发写入导致的章节重复与半截合并）。

背景：`.trae/specs/<spec>/tasks.md` 是多轮会话反复追加的台账文件。工作区位于
      SMB/UNC 挂载时，编辑器缓冲区、多个会话与同步客户端可能同时写入同一文件，
      出现「同一任务编号出现两次」「任务块被拼接成半截」等静默损坏；这类损坏
      在写入当下不可见，只在提交前的 diff 里才暴露，容易把错误内容带进历史。

检查：对 `.trae/specs/*/tasks.md` 断言
      1. `## Task <id>` 标题的编号在该文件内唯一（重复即失败，列出编号与行号）；
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


def check_file(path: Path, root: Path) -> list[str]:
    """校验单个 tasks.md，返回问题列表（空则通过）。"""
    rel = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    issues: list[str] = []

    if b"\x00" in raw:
        issues.append(f"{rel}: 含 NUL 字节，疑似截断或拼接损坏")

    text = raw.decode("utf-8", errors="replace")
    headings = list(TASK_HEADING.finditer(text))

    seen: dict[str, list[int]] = {}
    for match in headings:
        line_no = text.count("\n", 0, match.start()) + 1
        seen.setdefault(match.group(1), []).append(line_no)

    for task_id, line_nos in sorted(seen.items()):
        if len(line_nos) > 1:
            located = ", ".join(str(n) for n in line_nos)
            issues.append(f"{rel}: 任务编号 {task_id} 重复出现（行 {located}），疑似并发写入合并")

    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[match.start() : end]
        status_count = len(STATUS_LINE.findall(block))
        if status_count > 1:
            line_no = text.count("\n", 0, match.start()) + 1
            issues.append(
                f"{rel}: 任务 {match.group(1)} 的块内出现 {status_count} 条 Status（行 {line_no}），疑似块被拼接"
            )

    return issues


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    spec_root = root / ".trae" / "specs"
    files = sorted(spec_root.glob("*/tasks.md"))
    if not files:
        print(f"未找到任何 tasks.md（{spec_root}），请检查仓库结构。")
        return 2

    issues: list[str] = []
    task_total = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        task_total += len(TASK_HEADING.findall(text))
        issues.extend(check_file(path, root))

    print(f"扫描到 {len(files)} 个 tasks.md，共 {task_total} 个任务编号")
    if issues:
        print(f"\n发现 {len(issues)} 处台账损坏：")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("全部通过 spec 台账完整性校验：任务编号唯一、任务块单一、无 NUL 字节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
