"""
xs meta 命令模块
管理文档元数据（YAML frontmatter + TOML 二分法）

YAML frontmatter 只允许：id, x-toml-ref, source, version
其他元数据存储在 .meta/toml/ 下对应路径的 TOML 文件中
"""


import tomllib
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()

ALLOWED_YAML_FIELDS = {"id", "x-toml-ref", "source", "version"}
META_DIRNAME = ".meta"
TOML_SUBDIR = "toml"

META_README = """# 文档元数据目录

本目录采用「内容-元数据二分法」管理文档元数据：

## 结构

- `toml/` — TOML 格式的元数据文件，与 docs/ 目录结构镜像对应
  - 例如 `docs/guide/index.md` 的元数据在 `toml/docs/guide/index.toml`

## 规则

1. **YAML frontmatter** 只允许 4 个核心字段：
   - `id` — 文档唯一标识
   - `x-toml-ref` — 指向 TOML 元数据文件的相对路径
   - `source` — 内容来源
   - `version` — 文档版本

2. **TOML 元数据** 存储所有其他字段：
   - `title` — 文档标题
   - `date` — 创建/更新日期
   - `tags` — 标签数组
   - `category` — 分类
   - `changelog` — 变更记录
   - 以及其他任意自定义字段

3. YAML 字段优先于 TOML 同名字段

## 命令

- `xs meta scan` — 扫描文档元数据状态
- `xs meta validate [--fix]` — 验证/修复不合规 frontmatter
- `xs meta init` — 初始化元数据目录
- `xs meta sync <path>` — 同步 frontmatter 与 TOML 元数据
"""


def _get_toml_path(workspace_root: Path, doc_path: Path) -> Path:
    rel = doc_path.relative_to(workspace_root)
    return workspace_root / META_DIRNAME / TOML_SUBDIR / rel.parent / (doc_path.stem + ".toml")


def _has_frontmatter(content: str) -> bool:
    return content.startswith("---\n")


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    if not _has_frontmatter(content):
        return {}, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, content
    fm_text = content[4:end]
    body = content[end + 5 :]
    meta: dict = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [i.strip().strip('"').strip("'") for i in val[1:-1].split(",")]
            meta[key] = [i for i in items if i]
        elif val.lower() in ("true", "false"):
            meta[key] = val.lower() == "true"
        else:
            meta[key] = val.strip('"').strip("'")
    return meta, body


def _build_frontmatter(fields: dict) -> str:
    lines = ["---"]
    for key in ("id", "x-toml-ref", "source", "version"):
        if key in fields:
            val = fields[key]
            if isinstance(val, list):
                lines.append(f"{key}: [{', '.join(str(v) for v in val)}]")
            elif isinstance(val, bool):
                lines.append(f"{key}: {str(val).lower()}")
            else:
                lines.append(f'{key}: "{val}"' if " " in str(val) or "-" in str(val) else f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _toml_key(key: str) -> str:
    return key.replace("-", "_")


def _from_toml_key(key: str) -> str:
    return key.replace("_", "-")


def _write_toml_meta(toml_path: Path, extra_fields: dict, existing_data: dict | None = None) -> None:
    try:
        import tomli_w
    except ImportError:
        console.print("[red]错误: tomli_w 未安装，请运行 pip install tomli-w[/red]")
        raise typer.Exit(1)

    if existing_data is None:
        existing_data = {}
        if toml_path.exists():
            try:
                with open(toml_path, "rb") as f:
                    existing_data = tomllib.load(f)
            except Exception:
                existing_data = {}

    data = dict(existing_data)
    for key, val in extra_fields.items():
        tk = _toml_key(key)
        if key == "tags" and isinstance(val, str):
            val = [t.strip() for t in val.strip("[]").split(",") if t.strip()]
        data[tk] = val

    if "date" not in data:
        data["date"] = datetime.now().strftime("%Y-%m-%d")

    toml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(toml_path, "wb") as f:
        tomli_w.dump(data, f)


def _read_toml_meta(toml_path: Path) -> dict:
    if not toml_path.exists():
        return {}
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return {_from_toml_key(k): v for k, v in data.items()}
    except Exception:
        return {}


def _collect_doc_files(path: Path) -> list[Path]:
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    if path.is_file():
        return [path.resolve()]
    files = []
    for ext in ("*.md", "*.rst"):
        for f in path.rglob(ext):
            try:
                f.relative_to(workspace_root / META_DIRNAME)
                continue
            except ValueError:
                pass
            if ".git" in f.parts or "__pycache__" in f.parts:
                continue
            files.append(f)
    return sorted(files)


def meta_init(
    directory: Path | None = typer.Argument(None, help="工作区目录，默认为当前目录"),
    force: bool = typer.Option(False, "--force", "-f", help="覆盖已存在的 README"),
) -> None:
    """初始化文档元数据目录结构"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    meta_root = workspace_root / META_DIRNAME
    toml_root = meta_root / TOML_SUBDIR

    meta_root.mkdir(parents=True, exist_ok=True)
    toml_root.mkdir(parents=True, exist_ok=True)

    gitkeep = toml_root / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    readme = meta_root / "README.md"
    if not readme.exists() or force:
        readme.write_text(META_README, encoding="utf-8")
        console.print(f"[green]✓ 创建 {readme.relative_to(workspace_root)}[/green]")
    else:
        console.print(f"[dim]{readme.relative_to(workspace_root)} 已存在，跳过[/dim]")

    console.print(f"[green]✓ 元数据目录已初始化: {meta_root.relative_to(workspace_root)}/[/green]")


def meta_validate(
    path: Path = typer.Argument(None, help="要检查的文档文件或目录，默认为工作区根目录"),
    fix: bool = typer.Option(False, "--fix", help="自动修复不合规的 frontmatter"),
) -> None:
    """验证文档 frontmatter 是否符合二分法规范"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    target = path or workspace_root
    files = _collect_doc_files(target)

    errors = 0
    fixed = 0

    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue

        fm, body = _parse_frontmatter(content)
        bad_fields = set(fm.keys()) - ALLOWED_YAML_FIELDS

        if bad_fields:
            errors += 1
            rel = f.relative_to(workspace_root)
            console.print(f"[red]✗ {rel}[/red]: 发现禁止字段: {', '.join(sorted(bad_fields))}")

            if fix:
                toml_path = _get_toml_path(workspace_root, f)
                extra = {k: fm[k] for k in bad_fields if k in fm}
                _write_toml_meta(toml_path, extra)

                clean_fm = {k: v for k, v in fm.items() if k in ALLOWED_YAML_FIELDS}
                rel_toml = toml_path.relative_to(workspace_root).as_posix()
                clean_fm["x-toml-ref"] = rel_toml
                new_content = _build_frontmatter(clean_fm) + body
                f.write_text(new_content, encoding="utf-8")
                fixed += 1
                console.print(f"  [green]→ 已修复，元数据移至 {rel_toml}[/green]")

    if errors == 0:
        console.print(f"[green]✓ 所有文档 frontmatter 合规（共 {len(files)} 个文档）[/green]")
    else:
        if fix:
            console.print(f"[yellow]修复了 {fixed}/{errors} 个文档[/yellow]")
        else:
            console.print(f"[yellow]发现 {errors} 个问题，使用 --fix 自动修复[/yellow]")


def meta_scan(
    directory: Path | None = typer.Argument(None, help="要扫描的目录，默认为工作区根目录"),
) -> None:
    """扫描工作区文档元数据状态"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    scan_dir = directory or workspace_root

    md_files = _collect_doc_files(scan_dir)
    total = len(md_files)
    with_fm = 0
    bad_fm = 0
    has_toml = 0

    table = Table(title="文档元数据状态", show_header=True, header_style="bold magenta")
    table.add_column("文档", style="cyan")
    table.add_column("Frontmatter", width=12)
    table.add_column("禁止字段", width=20)
    table.add_column("TOML元数据", width=12)

    for f in md_files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, _ = _parse_frontmatter(content)
        has_fm = bool(fm)
        bad = set(fm.keys()) - ALLOWED_YAML_FIELDS
        toml_path = _get_toml_path(workspace_root, f)
        toml_exists = toml_path.exists()

        if has_fm:
            with_fm += 1
        if bad:
            bad_fm += 1
        if toml_exists:
            has_toml += 1

        rel = f.relative_to(workspace_root)
        fm_status = "[green]✓[/green]" if has_fm else "[dim]-[/dim]"
        bad_status = f"[red]{', '.join(sorted(bad))}[/red]" if bad else "[dim]-[/dim]"
        toml_status = "[green]✓[/green]" if toml_exists else "[dim]-[/dim]"
        table.add_row(str(rel), fm_status, bad_status, toml_status)

    console.print(table)
    console.print()
    console.print(f"总计: {total} 个文档 | 有frontmatter: {with_fm} | 不合规: {bad_fm} | 有TOML元数据: {has_toml}")


def meta_sync(
    path: Path = typer.Argument(None, help="要同步的文件或目录，默认为工作区根目录"),
) -> None:
    """同步 YAML frontmatter 与 TOML 元数据，确保两者一致"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    target = path or workspace_root
    files = _collect_doc_files(target)

    synced = 0
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue

        fm, body = _parse_frontmatter(content)
        toml_path = _get_toml_path(workspace_root, f)

        bad_fields = set(fm.keys()) - ALLOWED_YAML_FIELDS
        need_write = False

        if bad_fields:
            extra = {k: fm[k] for k in bad_fields if k in fm}
            _write_toml_meta(toml_path, extra)
            clean_fm = {k: v for k, v in fm.items() if k in ALLOWED_YAML_FIELDS}
            clean_fm["x-toml-ref"] = toml_path.relative_to(workspace_root).as_posix()
            content = _build_frontmatter(clean_fm) + body
            need_write = True

        if "x-toml-ref" in fm and toml_path.exists():
            pass
        elif toml_path.exists() and not fm.get("x-toml-ref"):
            fm["x-toml-ref"] = toml_path.relative_to(workspace_root).as_posix()
            content = _build_frontmatter(fm) + body
            need_write = True

        if need_write:
            f.write_text(content, encoding="utf-8")
            synced += 1
            console.print(f"[green]✓ 同步 {f.relative_to(workspace_root)}[/green]")

    if synced == 0:
        console.print("[green]✓ 所有文档元数据已同步[/green]")
    else:
        console.print(f"[green]✓ 同步了 {synced} 个文档[/green]")


app = typer.Typer(help="文档元数据管理命令", add_completion=False)
app.command("init")(meta_init)
app.command("validate")(meta_validate)
app.command("scan")(meta_scan)
app.command("sync")(meta_sync)
