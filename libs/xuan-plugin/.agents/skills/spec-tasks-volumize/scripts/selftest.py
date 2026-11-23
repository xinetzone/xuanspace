#!/usr/bin/env python3
"""volumize.py 端到端自检：在系统临时目录内跑全链路，不触碰工作区。

覆盖：围栏内伪锚点排除、无尾换行源、逐字节重建、索引重渲染幂等、改指归一化、
      源被索引覆盖后的拒绝（防重跑）、锚点歧义的拒绝。
用法：python selftest.py   （全部通过退出 0）
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLI = HERE / "volumize.py"
TPL = HERE.parent / "templates" / "index.md.tmpl"

BANNER = [
    "> 分卷来源：`{source_name}` 原文 L{start}–L{end}（{date} 切分，逐字节保真；本行与下 {k} 行横幅不属原文）",
    "> 上一卷：{prev} · 下一卷：{next} · 索引：[{index_rel}]({index_rel})",
    "> 主题：{topic}",
]

BODY = (
    "# 样例台账\n\n"
    "导言段落，指向后续任务。\n\n"
    "```\n"
    "## Task 2: 围栏内的伪锚点（必须被忽略）\n"
    "```\n\n"
    "## Task 1: 第一项\n\n"
    "- **Status**: `completed`\n"
    "- 链接：](../libs/x.md)\n\n"
    "## Task 2: 第二项\n\n"
    "- **Status**: `pending`\n"
    "- 链接：](../libs/y.md)\n"
)

VOLUMES = [
    {"file": "00-head.md", "topic": "导言与前段", "anchor": None},
    {"file": "01-tail.md", "topic": "任务块", "anchor": "## Task 1:"},
]

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" :: {detail}" if not ok and detail else ""))


def run(plan: Path, cmd: str, *extra: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CLI), cmd, "--plan", str(plan), *extra],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def make_case(root: Path, name: str, text: str, volumes: list[dict]) -> tuple[Path, Path]:
    case = root / name
    case.mkdir()
    src = case / "tasks.md"
    src.write_bytes(text.encode("utf-8"))
    plan = {
        "source": str(src),
        "archive": str(case / "archive"),
        "out_dir": str(case / "tasks"),
        "date": "2026-11-23",
        "banner": BANNER,
        "volumes": volumes,
        "rewrites": [],
        "index": {"path": str(src), "template": str(TPL)},
        "guards": {"forbid_body_contains": ["## 分卷索引"]},
    }
    path = case / "plan.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, case


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="volumize-selftest-") as tmp:
        root = Path(tmp)

        # T1 无尾换行源：逐字节重建
        plan, case = make_case(root, "no-trailing-nl", BODY.rstrip("\n"), VOLUMES)
        code, out = run(plan, "split")
        record("T1 无尾换行 split 通过", code == 0, out)
        code, out = run(plan, "verify")
        record("T1 无尾换行 BYTE_EXACT", code == 0 and "BYTE_EXACT True" in out, out)

        # T2 有尾换行源：split + check 幂等
        plan, case = make_case(root, "trailing-nl", BODY, VOLUMES)
        code, out = run(plan, "split")
        record("T2 split 通过且刷出索引", code == 0 and "INDEX_RENDERED" in out, out)
        idx = (case / "tasks.md").read_text(encoding="utf-8")
        record("T2 索引含两卷与两条任务", "00-head.md" in idx and "01-tail.md" in idx and "| `2` |" in idx, idx[:400])
        code, out = run(plan, "check")
        record("T2 check 首次通过", code == 0 and "INDEX_OK" in out, out)
        code, out = run(plan, "check")
        record("T2 check 幂等", code == 0 and "INDEX_OK" in out, out)

        # T3 改指：归一化等价 + 指纹刷新 + 索引同步
        code, out = run(plan, "relink", "--old", "](../libs/", "--new", "](../../libs/")
        record("T3 relink 归一化等价", code == 0 and "NORMALIZED_EQUAL True" in out, out)
        record("T3 链接已全量改指", (case / "tasks" / "01-tail.md").read_bytes().count(b"](../../libs/") == 2, "")
        code, out = run(plan, "check")
        record("T3 relink 后 check 通过（指纹已刷新）", code == 0 and "INDEX_OK" in out, out)

        # T4 源已被索引覆盖：拒绝重跑
        code, out = run(plan, "split")
        record("T4 已覆盖源拒绝重切", code == 2 and "PLAN_ERROR" in out, out)

        # T5 锚点歧义：围栏外重复命中即失败
        dup = BODY + "\n## Task 1: 重复锚点\n"
        plan, _ = make_case(root, "ambiguous", dup, VOLUMES)
        code, out = run(plan, "split")
        record("T5 锚点歧义拒绝", code == 2 and "期望 1 处命中" in out, out)

        # T6 lock 缺失警告与 relock 重建（不动正文与索引）
        plan, case = make_case(root, "lock-missing", BODY, VOLUMES)
        code, out = run(plan, "split")
        record("T6 split 前置通过", code == 0, out)
        lock_file = case / "tasks" / ".volumize.lock.json"
        before = {p.name: p.read_bytes() for p in (case / "tasks").glob("*.md")}
        idx_before = (case / "tasks.md").read_bytes()
        lock_file.unlink()
        code, out = run(plan, "verify")
        record("T6 缺 lock 显式警告且退出码不变", code == 0 and "LOCK_MISSING" in out and "BYTE_EXACT True" in out, out)
        code, out = run(plan, "relock")
        record("T6 relock 重建并 LOCK_OK", code == 0 and "LOCK_REBUILT" in out and "LOCK_OK" in out, out)
        after = {p.name: p.read_bytes() for p in (case / "tasks").glob("*.md")}
        record("T6 relock 不改正文与索引", before == after and (case / "tasks.md").read_bytes() == idx_before, "")
        code, out = run(plan, "verify")
        record("T6 重建后警告消失", code == 0 and "LOCK_OK" in out and "LOCK_MISSING" not in out, out)

    bad = [name for name, ok, _ in results if not ok]
    print(f"\nSUMMARY {len(results) - len(bad)}/{len(results)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())