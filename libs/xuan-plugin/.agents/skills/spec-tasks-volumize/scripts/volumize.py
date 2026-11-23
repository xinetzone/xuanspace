#!/usr/bin/env python3
"""volumize.py — 长 Markdown 台账的无损分卷、派生索引与指纹对账。

设计基线（与一次性脚本的关键差异）：
1. 以「行数组 + 区间」建模，重建 = "\\n".join(所有区间拼接)，故逐字节等价由构造保证，
   不再依赖「末卷是否以换行结尾」的特例分支；
2. 分卷覆盖面显式断言（首段 s==0、单调递增、末段 e==len），锚点要求唯一命中并默认跳过代码块；
3. 指纹分域：body_sha256（去横幅正文，可比对冻结原件）与 file_sha256（含横幅，可校验文件落盘）；
4. 索引是派生视图，`check` 可在内存重渲染并与磁盘逐字节比对，杜绝「散文承诺同步」。

子命令：
  freeze   固化原件到 archive/raw 并写 sha256sum 兼容 MANIFEST
  split    按锚点切卷 → 写分卷 → 重建断言 → 输出 lock
  verify   从分卷重建，与冻结原件（按 plan.rewrites 归一化）比对
  relink   对全部卷执行声明式改指 → 复算全部指纹 → 重渲染索引 → 归一化验证
  check    verify + 索引重渲染比对（幂等门）

用法：python volumize.py <子命令> --plan plan.json [--old A --new B] [--strict-index]
退出码：0 通过 / 1 校验失败 / 2 计划或环境错误
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_HEADING_RE = r"^##\s+Task\s+(?P<id>[A-Za-z0-9][A-Za-z0-9._-]*):?\s*(?P<title>.*)$"
DEFAULT_STATUS_RE = r"^\s*-\s*\*\*Status\*\*:\s*`(?P<status>[^`]+)`"
DEFAULT_BLOCK_END_RE = r"^##\s"

VOLUME_TABLE_HEAD = (
    "| 卷 | 主题 | 原文区间 | 字节 | 任务数 | 正文 SHA-256 前 12 位 | 文件 SHA-256 前 12 位 |\n"
    "|---|---|---|---|---|---|---|"
)
TASK_TABLE_HEAD = "| 编号 | 标题 | 状态 | 所在分卷 |\n|---|---|---|---|"


class PlanError(Exception):
    """计划文件或前置条件不满足（退出码 2）。"""


# ---------- 基础工具 ----------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text_bytes(path: Path) -> str:
    """按字节读取后解码，避免 text 模式的换行归一化破坏逐字节保真。"""
    return path.read_bytes().decode("utf-8")


def write_text_bytes(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def load_plan(path: Path) -> dict:
    if not path.exists():
        raise PlanError(f"计划文件不存在：{path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    for key in ("source", "archive", "out_dir", "volumes", "banner"):
        if key not in plan:
            raise PlanError(f"计划缺少必需字段：{key}")
    if not plan["volumes"]:
        raise PlanError("volumes 为空")
    return plan


def raw_path(plan: dict) -> Path:
    return Path(plan["archive"]) / "raw" / Path(plan["source"]).name


def fence_mask(parts: list[str]) -> list[bool]:
    """标记围栏代码块内部行（围栏行本身亦标记），用于排除伪锚点。"""
    mask: list[bool] = []
    inside = False
    for line in parts:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            inside = not inside
            mask.append(True)
            continue
        mask.append(inside)
    return mask


def resolve_anchor(parts: list[str], anchor: str | None, occurrence: int = 1) -> int:
    """锚点必须在围栏外唯一命中；返回 0 基行下标。"""
    if anchor is None:
        return 0
    mask = fence_mask(parts)
    hits = [i for i, line in enumerate(parts) if line.startswith(anchor) and not mask[i]]
    if len(hits) != occurrence:
        raise PlanError(
            f"锚点 {anchor!r} 期望 {occurrence} 处命中，实际 {len(hits)} 处：行 {[h + 1 for h in hits][:10]}"
            "（围栏代码块内命中已排除；如需第 n 处请显式写 occurrence）"
        )
    return hits[occurrence - 1]


def strip_banner(file_text: str, banner_lines: int) -> str:
    """写入侧的严格逆运算：去掉至多一个尾换行，切行后丢弃前 banner_lines 行。"""
    body = file_text[:-1] if file_text.endswith("\n") else file_text
    return "\n".join(body.split("\n")[banner_lines:])


# ---------- freeze ----------


def cmd_freeze(plan: dict, force: bool = False) -> int:
    src = Path(plan["source"])
    raw = raw_path(plan)
    if not src.exists():
        raise PlanError(f"源文件不存在：{src}")
    src_bytes = src.read_bytes()
    digest = sha256_bytes(src_bytes)
    if raw.exists():
        if sha256_bytes(raw.read_bytes()) == digest:
            print(f"FREEZE_SKIP 已冻结且一致 {raw} sha={digest}")
            return 0
        if not force:
            raise PlanError(f"归档副本与源不一致且未指定 --force：{raw}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(src_bytes)
    if sha256_bytes(raw.read_bytes()) != digest:
        raise PlanError("冻结副本回读哈希不一致（写入通道异常）")
    manifest = raw.parent.parent / "MANIFEST.sha256"
    write_text_bytes(manifest, f"{digest}  raw/{raw.name}\n")
    print(f"FREEZE_OK {raw} bytes={len(src_bytes)} sha={digest}")
    return 0


# ---------- split ----------


def cmd_split(plan: dict) -> int:
    src = Path(plan["source"])
    raw = raw_path(plan)
    out = Path(plan["out_dir"])
    if not raw.exists():
        code = cmd_freeze(plan)
        if code != 0:
            return code
    raw_bytes = raw.read_bytes()
    raw_digest = sha256_bytes(raw_bytes)
    if src.exists():
        src_text = read_text_bytes(src)
        for guard in plan.get("guards", {}).get("forbid_body_contains", []):
            if guard in src_text:
                raise PlanError(
                    f"源 {src} 已含派生标记 {guard!r}：源已被索引覆盖，拒绝以它为分卷源；请删除/移走索引后再从归档 {raw} 取源"
                )

    text = raw_bytes.decode("utf-8")
    parts = text.split("\n")
    volumes = plan["volumes"]
    starts: list[int] = []
    for vol in volumes:
        starts.append(resolve_anchor(parts, vol.get("anchor"), vol.get("occurrence", 1)))
    ends = starts[1:] + [len(parts)]

    if starts[0] != 0:
        raise PlanError(f"首卷必须从 L1 起，实际 L{starts[0] + 1}")
    for prev, cur in zip(starts, starts[1:]):
        if cur <= prev:
            raise PlanError(f"卷起点非单调递增：L{prev + 1} → L{cur + 1}")
    if any(e <= s for s, e in zip(starts, ends)):
        raise PlanError("存在空卷区间")

    banner_templates = plan["banner"]
    if not banner_templates:
        raise PlanError("banner 不能为空（去掉全部横幅后无法区分正文边界，且重建断言失去意义）")
    banner_lines = len(banner_templates)
    index_rel = os.path.relpath(Path(plan["index"]["path"]).resolve(), out.resolve()).replace(os.sep, "/")
    date = plan.get("date", "")

    out.mkdir(parents=True, exist_ok=True)
    facts: list[dict] = []
    for i, vol in enumerate(volumes):
        s, e = starts[i], ends[i]
        body = "\n".join(parts[s:e])
        banner = [
            tpl.format(
                source_name=src.name,
                start=s + 1,
                end=e,
                date=date,
                k=banner_lines - 1,
                index_rel=index_rel,
                prev=volumes[i - 1]["file"] if i else "（无）",
                next=volumes[i + 1]["file"] if i + 1 < len(volumes) else "（无）",
                topic=vol.get("topic", ""),
            )
            for tpl in banner_templates
        ]
        file_text = "\n".join(banner) + "\n" + body + "\n"
        write_text_bytes(out / vol["file"], file_text)
        facts.append(
            {
                "file": vol["file"],
                "topic": vol.get("topic", ""),
                "start": s,
                "end": e,
                "banner_lines": banner_lines,
                "body_bytes": len(body.encode("utf-8")),
                "body_sha256": sha256_bytes(body.encode("utf-8")),
                "file_bytes": len(file_text.encode("utf-8")),
                "file_sha256": sha256_bytes(file_text.encode("utf-8")),
            }
        )

    rebuilt = "\n".join(
        "\n".join(strip_banner(read_text_bytes(out / f["file"]), banner_lines).split("\n")) for f in facts
    )
    if sha256_bytes(rebuilt.encode("utf-8")) != raw_digest:
        raise PlanError("逐字节重建断言失败：分卷去横幅拼接与冻结原件不一致")
    for f in facts:
        blocks = [ln for ln in parts[f["start"] : f["end"]] if re.match(DEFAULT_HEADING_RE, ln)]
        f["task_ids"] = [re.match(DEFAULT_HEADING_RE, ln).group("id") for ln in blocks]

    lock = {"source_sha256": raw_digest, "source_bytes": len(raw_bytes), "source_lines": len(parts), "volumes": facts}
    write_text_bytes(out / (plan.get("lock_name", ".volumize.lock.json")), json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(f"SPLIT_OK volumes={len(facts)} parts={len(parts)} source_sha={raw_digest}")
    print(f"BYTE_EXACT True banner_lines={banner_lines}")
    if "index" in plan:
        write_text_bytes(Path(plan["index"]["path"]), render_index(plan, text))
        print(f"INDEX_RENDERED {plan['index']['path']}")
    return 0


# ---------- verify ----------


def rebuild_from_volumes(plan: dict) -> str:
    out = Path(plan["out_dir"])
    banner_lines = len(plan["banner"])
    chunks: list[str] = []
    for vol in plan["volumes"]:
        text = read_text_bytes(out / vol["file"])
        chunks.extend(strip_banner(text, banner_lines).split("\n"))
    return "\n".join(chunks)


def apply_rewrites(text: str, plan: dict) -> str:
    for rw in plan.get("rewrites", []):
        text = text.replace(rw["old"], rw["new"])
    return text


def write_lock(plan: dict) -> None:
    """按磁盘现状重算全部分卷指纹并落 lock（改指后必须刷新，否则 verify 会报漂移）。"""
    rebuilt = rebuild_from_volumes(plan)
    parts = rebuilt.split("\n")
    banner_lines = len(plan["banner"])
    facts = []
    for vol in plan["volumes"]:
        s = resolve_anchor(parts, vol.get("anchor"), vol.get("occurrence", 1))
        e = _volume_end(parts, plan, vol)
        body = "\n".join(parts[s:e])
        data = (Path(plan["out_dir"]) / vol["file"]).read_bytes()
        facts.append(
            {
                "file": vol["file"],
                "topic": vol.get("topic", ""),
                "start": s,
                "end": e,
                "banner_lines": banner_lines,
                "body_bytes": len(body.encode("utf-8")),
                "body_sha256": sha256_bytes(body.encode("utf-8")),
                "file_bytes": len(data),
                "file_sha256": sha256_bytes(data),
            }
        )
    raw_bytes = raw_path(plan).read_bytes()
    lock = {
        "source_sha256": sha256_bytes(raw_bytes),
        "source_bytes": len(raw_bytes),
        "source_lines": len(parts),
        "volumes": facts,
    }
    write_text_bytes(
        Path(plan["out_dir"]) / plan.get("lock_name", ".volumize.lock.json"),
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
    )


def cmd_verify(plan: dict, quiet: bool = False) -> int:
    raw_bytes = raw_path(plan).read_bytes()
    raw_text = raw_bytes.decode("utf-8")
    rebuilt = rebuild_from_volumes(plan)
    exact = sha256_bytes(rebuilt.encode("utf-8")) == sha256_bytes(raw_bytes)
    rewrites = plan.get("rewrites", [])
    normalized = sha256_bytes(apply_rewrites(rebuilt, plan).encode("utf-8")) == sha256_bytes(
        apply_rewrites(raw_text, plan).encode("utf-8")
    )
    print(f"RAW_SHA {sha256_bytes(raw_bytes)}")
    print(f"REBUILT_SHA {sha256_bytes(rebuilt.encode('utf-8'))}")
    print(f"BYTE_EXACT {exact}")
    print(f"NORMALIZED_EQUAL {normalized} rewrites={len(rewrites)}")
    ok = exact if not rewrites else normalized
    if not ok:
        print("VERIFY_FAIL 分卷与冻结原件不等价", file=sys.stderr)
        return 1
    lock_path = Path(plan["out_dir"]) / plan.get("lock_name", ".volumize.lock.json")
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        drift = []
        for f in lock["volumes"]:
            data = (Path(plan["out_dir"]) / f["file"]).read_bytes()
            if len(data) != f["file_bytes"] or sha256_bytes(data) != f["file_sha256"]:
                drift.append(f["file"])
        if drift:
            print(f"LOCK_DRIFT {drift}（指纹已变，需重算并同步索引）", file=sys.stderr)
            return 1
        if not quiet:
            print(f"LOCK_OK volumes={len(lock['volumes'])}")
    else:
        print(
            f"LOCK_MISSING {lock_path.name} 不存在：LOCK_DRIFT 检测层已跳过，"
            "请跑 relock 重建 lock（本次 BYTE_EXACT/INDEX_OK 结论不受影响）",
            file=sys.stderr,
        )
    return 0


# ---------- 索引渲染 ----------


def extract_tasks(parts: list[str], plan: dict) -> list[dict]:
    cfg = plan.get("tasks", {})
    heading = re.compile(cfg.get("heading_regex", DEFAULT_HEADING_RE))
    status_re = re.compile(cfg.get("status_regex", DEFAULT_STATUS_RE))
    block_end = re.compile(cfg.get("block_end_regex", DEFAULT_BLOCK_END_RE))
    default_status = cfg.get("default_status", "未标注")
    rows: list[dict] = []
    for vol in plan["volumes"]:
        start = resolve_anchor(parts, vol.get("anchor"), vol.get("occurrence", 1))
        end = _volume_end(parts, plan, vol)
        for i in range(start, end):
            m = heading.match(parts[i])
            if not m:
                continue
            status = default_status
            for j in range(i + 1, end):
                if block_end.match(parts[j]):
                    break
                sm = status_re.match(parts[j])
                if sm:
                    status = sm.group("status")
                    break
            rows.append({"id": m.group("id"), "title": m.group("title").strip(), "status": status, "file": vol["file"]})
    return rows


def _volume_end(parts: list[str], plan: dict, vol: dict) -> int:
    volumes = plan["volumes"]
    idx = volumes.index(vol)
    if idx + 1 < len(volumes):
        return resolve_anchor(parts, volumes[idx + 1].get("anchor"), volumes[idx + 1].get("occurrence", 1))
    return len(parts)


def render_index(plan: dict, rebuilt_text: str) -> str:
    tpl_path = Path(plan["index"]["template"])
    tpl = tpl_path.read_text(encoding="utf-8")
    parts = rebuilt_text.split("\n")
    vols = plan["volumes"]
    all_tasks = extract_tasks(parts, plan)
    facts = []
    for vol in vols:
        s = resolve_anchor(parts, vol.get("anchor"), vol.get("occurrence", 1))
        e = _volume_end(parts, plan, vol)
        body = "\n".join(parts[s:e])
        data = (Path(plan["out_dir"]) / vol["file"]).read_bytes()
        facts.append(
            {
                "file": vol["file"],
                "topic": vol.get("topic", ""),
                "start": s,
                "end": e,
                "bytes": len(data),
                "body_sha": sha256_bytes(body.encode("utf-8")),
                "file_sha": sha256_bytes(data),
                "tasks": sum(1 for t in all_tasks if t["file"] == vol["file"]),
            }
        )
    vol_rows = [
        f"| {f['file']} | {f['topic']} | L{f['start'] + 1}–L{f['end']} | {f['bytes']} | {f['tasks']} | "
        f"`{f['body_sha'][:12]}…` | `{f['file_sha'][:12]}…` |"
        for f in facts
    ]
    tasks = all_tasks
    index_dir = Path(plan["index"]["path"]).parent
    task_rows = []
    for t in tasks:
        rel = os.path.relpath((Path(plan["out_dir"]) / t["file"]).resolve(), index_dir.resolve()).replace(os.sep, "/")
        task_rows.append(f"| `{t['id']}` | {t['title']} | `{t['status']}` | [{t['file']}]({rel}) |")
    values = {
        "source_name": Path(plan["source"]).name,
        "source_sha256": sha256_bytes(raw_path(plan).read_bytes()),
        "source_bytes": str(len(raw_path(plan).read_bytes())),
        "source_lines": str(len(parts)),
        "volume_count": str(len(facts)),
        "task_count": str(len(tasks)),
        "volume_table": VOLUME_TABLE_HEAD + "\n" + "\n".join(vol_rows),
        "task_table": TASK_TABLE_HEAD + "\n" + "\n".join(task_rows),
        "date": plan.get("date", ""),
        "archive_rel": os.path.relpath(raw_path(plan).resolve(), index_dir.resolve()).replace(os.sep, "/"),
        "rewrites_note": "；".join(f"`{r['old'].strip()}` → `{r['new'].strip()}`" for r in plan.get("rewrites", [])) or "无",
    }
    out = tpl
    for key, val in values.items():
        out = out.replace("{{" + key + "}}", val)
    leftover = re.findall(r"\{\{([a-z_]+)\}\}", out)
    if leftover:
        raise PlanError(f"模板存在未定义占位符：{sorted(set(leftover))}")
    return out


def cmd_check(plan: dict) -> int:
    code = cmd_verify(plan)
    if code != 0:
        return code
    if "index" not in plan:
        print("INDEX_SKIP 计划未声明 index")
        return 0
    rebuilt = rebuild_from_volumes(plan)
    expected = render_index(plan, rebuilt)
    idx_path = Path(plan["index"]["path"])
    actual = read_text_bytes(idx_path) if idx_path.exists() else ""
    if expected != actual:
        exp_lines, act_lines = expected.split("\n"), actual.split("\n")
        for i in range(max(len(exp_lines), len(act_lines))):
            a = exp_lines[i] if i < len(exp_lines) else "<EOF>"
            b = act_lines[i] if i < len(act_lines) else "<EOF>"
            if a != b:
                print(f"INDEX_DRIFT 首个差异 L{i + 1}\n  期望：{a}\n  实际：{b}", file=sys.stderr)
                break
        return 1
    print(f"INDEX_OK {idx_path}")
    return 0


# ---------- relink / relock ----------


def cmd_relock(plan: dict) -> int:
    """按磁盘现状重建 lock：不切分、不写分卷正文、不渲染索引。

    用于 lock 被误删或过期后的修复；前提是分卷与冻结原件仍自洽，
    否则随后的 verify 会以 exit 1 暴露问题。
    """
    write_lock(plan)
    lock_path = Path(plan["out_dir"]) / plan.get("lock_name", ".volumize.lock.json")
    print(f"LOCK_REBUILT {lock_path}")
    return cmd_verify(plan)


def cmd_relink(plan: dict, old: str, new: str) -> int:
    out = Path(plan["out_dir"])
    total = 0
    for vol in plan["volumes"]:
        path = out / vol["file"]
        text = read_text_bytes(path)
        count = text.count(old)
        if count:
            write_text_bytes(path, text.replace(old, new))
            total += count
            print(f"RELINK {vol['file']} rewrites={count}")
    if total == 0:
        raise PlanError(f"声明的改指串在全部卷中零命中：{old!r}（拒绝静默空操作）")
    plan.setdefault("rewrites", []).append({"old": old, "new": new})
    write_lock(plan)
    if "index" in plan:
        expected = render_index(plan, rebuild_from_volumes(plan))
        write_text_bytes(Path(plan["index"]["path"]), expected)
        print(f"INDEX_RENDERED {plan['index']['path']} total_rewrites={total}")
    return cmd_verify(plan)


# ---------- 入口 ----------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="长 Markdown 台账无损分卷与派生索引")
    ap.add_argument("command", choices=["freeze", "split", "verify", "relink", "relock", "check"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--force", action="store_true", help="freeze 允许覆盖不一致的归档副本")
    ap.add_argument("--old", help="relink 的原文串")
    ap.add_argument("--new", help="relink 的新文串")
    args = ap.parse_args(argv)
    try:
        plan = load_plan(Path(args.plan))
        if args.command == "freeze":
            return cmd_freeze(plan, args.force)
        if args.command == "split":
            return cmd_split(plan)
        if args.command == "verify":
            return cmd_verify(plan)
        if args.command == "check":
            return cmd_check(plan)
        if args.command == "relock":
            return cmd_relock(plan)
        if not args.old or args.new is None:
            raise PlanError("relink 需要 --old 与 --new")
        code = cmd_relink(plan, args.old, args.new)
        # 改指是计划的一部分：落盘后 check/verify 才能按同一归一化口径复验
        write_text_bytes(Path(args.plan), json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
        return code
    except PlanError as exc:
        print(f"PLAN_ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))