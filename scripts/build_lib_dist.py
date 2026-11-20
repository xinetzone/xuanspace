#!/usr/bin/env python3
"""
脚本名称：build_lib_dist.py
用途：构建 libs/ 下指定子项目的 wheel，并在输出目录生成 sha256 CHECKSUMS，供内网/离线分发
使用方法：
    python scripts/build_lib_dist.py --lib xuan-cmd
    python scripts/build_lib_dist.py --lib xuan-cmd --outdir /srv/wheels --no-isolation
输出布局（默认 <仓库>/dist/libs/<库名>/）：
    <库名>-<版本>-py3-none-any.whl
    CHECKSUMS            # sha256sum 兼容格式，按文件名排序
说明：
    产物目录位于 dist/ 下，不纳入 Git；重复执行会重建并覆盖同名 wheel，CHECKSUMS 每次重算。
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIBS = REPO / "libs"
CHECKSUMS_NAME = "CHECKSUMS"


def fail(message: str) -> None:
    print(f"错误：{message}", file=sys.stderr)
    raise SystemExit(1)


def read_version(lib_dir: Path) -> str:
    pyproject = lib_dir / "pyproject.toml"
    if not pyproject.is_file():
        fail(f"{pyproject} 不存在，libs/ 下只有独立发行版才可发布")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    if "version" not in project:
        fail(f"{pyproject} 未声明静态 version（dynamic 版本暂不支持此脚本）")
    return str(project["version"])


def build_wheel(lib_dir: Path, staging: Path, no_isolation: bool) -> Path:
    cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(staging), str(lib_dir)]
    if no_isolation:
        cmd.insert(3, "--no-isolation")
    print(f"构建：{' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        fail("wheel 构建失败")
    wheels = sorted(staging.glob("*.whl"))
    if len(wheels) != 1:
        fail(f"预期 1 个 wheel，实际 {len(wheels)} 个：{[w.name for w in wheels]}")
    return wheels[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(outdir: Path) -> Path:
    files = sorted(p for p in outdir.iterdir() if p.is_file() and p.name != CHECKSUMS_NAME)
    if not files:
        fail(f"{outdir} 中没有可登记的发布物")
    lines = [f"{sha256(p)}  {p.name}" for p in files]
    target = outdir / CHECKSUMS_NAME
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 libs/ 子项目 wheel 并生成 CHECKSUMS")
    parser.add_argument("--lib", required=True, help="libs/ 下的目录名，例如 xuan-cmd")
    parser.add_argument("--outdir", help="输出目录，默认 <仓库>/dist/libs/<库名>")
    parser.add_argument("--no-isolation", action="store_true", help="复用当前环境构建，不创建隔离环境")
    args = parser.parse_args()

    lib_dir = LIBS / args.lib
    if not lib_dir.is_dir():
        fail(f"未知子项目：{args.lib}（可选：{', '.join(sorted(p.name for p in LIBS.iterdir() if p.is_dir()))}）")

    version = read_version(lib_dir)
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else REPO / "dist" / "libs" / args.lib
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{args.lib}-build-") as staging:
        wheel = build_wheel(lib_dir, Path(staging), args.no_isolation)
        target = outdir / wheel.name
        shutil.move(str(wheel), target)
    print(f"发布物：{target}（{args.lib} {version}）")

    checksums = write_checksums(outdir)
    print(f"校验清单：{checksums}")
    print(checksums.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
