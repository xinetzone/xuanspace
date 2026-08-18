"""OKF v0.2 §11 一致性校验模块。

提供 Bundle 的严格项与宽松项校验，生成 ConformanceReport。
"""

from .frontmatter import parse_frontmatter  # noqa: F401
from .links import check_broken_links, parse_links_with_context
from .models import Bundle, ConformanceReport

__all__ = [
    "validate_strict",
    "validate_lenient",
    "check_bundle",
    "is_conformant",
    "format_report",
]


def validate_strict(bundle: Bundle) -> list[str]:
    """严格项检查（不通过则 Bundle 不合规）。

    检查项：
    - 每个非保留 ``.md`` 文件有可解析 YAML frontmatter（frontmatter 非空，body 非空）
    - 每个 frontmatter 含非空 ``type``
    - 保留文件名（``index.md`` / ``log.md``）遵守 §8/§9 结构

    返回错误描述列表，格式：``"ERROR: <problem_description>"``。
    空列表表示全部通过。
    """
    errors: list[str] = []

    for concept_id, concept in bundle.concepts.items():
        if not concept.frontmatter:
            errors.append(f"ERROR: Concept '{concept_id}' has no frontmatter")
        if concept.body.strip() == "":
            errors.append(f"ERROR: Concept '{concept_id}' has empty body")
        if not concept.type:
            errors.append(f"ERROR: Concept '{concept_id}' has empty type")

    if not bundle.indices:
        errors.append("ERROR: Missing index.md (bundle.indices is empty)")

    if not bundle.logs:
        errors.append("ERROR: Missing log.md (bundle.logs is empty)")

    return errors


def validate_lenient(bundle: Bundle) -> list[str]:
    """宽松项检查（输出警告，不拒绝 Bundle）。

    检查项：
    - 缺失可选字段（title、description）
    - 未知 ``type`` 值（OKF 不集中注册 type，输出提示但不拒绝）
    - 未知扩展键（``extra`` 字典非空时输出提示）
    - 断链检测（调用 ``check_broken_links``）
    - 缺失 ``index.md`` / ``log.md``

    返回警告描述列表，格式：``"WARNING: <problem_description>"``。
    空列表表示无警告。
    """
    warnings: list[str] = []

    for concept_id, concept in bundle.concepts.items():
        if not concept.title:
            warnings.append(f"WARNING: Concept '{concept_id}' is missing title")
        if not concept.description:
            warnings.append(f"WARNING: Concept '{concept_id}' is missing description")

        warnings.append(
            f"WARNING: Concept '{concept_id}' has type '{concept.type}' "
            f"(OKF does not centrally register types)"
        )

        if concept.extra:
            extra_keys = ", ".join(sorted(concept.extra.keys()))
            warnings.append(
                f"WARNING: Concept '{concept_id}' has unknown extension keys: {extra_keys}"
            )

        try:
            links = parse_links_with_context(concept.path, bundle.root)
        except FileNotFoundError:
            warnings.append(
                f"WARNING: Concept '{concept_id}' file not found at {concept.path}"
            )
            continue
        broken = check_broken_links(links, bundle)
        warnings.extend(broken)

    if not bundle.indices:
        warnings.append("WARNING: Missing index.md (bundle.indices is empty)")

    if not bundle.logs:
        warnings.append("WARNING: Missing log.md (bundle.logs is empty)")

    return warnings


def check_bundle(bundle: Bundle) -> ConformanceReport:
    """汇总严格项与宽松项。

    依次调用 :func:`validate_strict` 与 :func:`validate_lenient`，
    返回包含全部 errors 与 warnings 的 :class:`ConformanceReport`。
    """
    errors = validate_strict(bundle)
    warnings = validate_lenient(bundle)
    return ConformanceReport(errors=errors, warnings=warnings, bundle=bundle)


def is_conformant(report: ConformanceReport) -> bool:
    """判断 Bundle 是否合规：``len(report.errors) == 0``。"""
    return len(report.errors) == 0


def format_report(report: ConformanceReport) -> str:
    """格式化输出校验报告。

    格式::

        === OKF Conformance Report ===
        Bundle: <bundle_root>

        Errors (N):
          - ERROR: ...
          - ERROR: ...

        Warnings (M):
          - WARNING: ...
          - WARNING: ...

        Result: PASS / FAIL
    """
    lines: list[str] = []
    lines.append("=== OKF Conformance Report ===")
    lines.append(f"Bundle: {report.bundle.root if report.bundle else 'N/A'}")
    lines.append("")
    lines.append(f"Errors ({len(report.errors)}):")
    if report.errors:
        for err in report.errors:
            lines.append(f"  - {err}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Warnings ({len(report.warnings)}):")
    if report.warnings:
        for warn in report.warnings:
            lines.append(f"  - {warn}")
    else:
        lines.append("  (none)")
    lines.append("")
    result = "PASS" if len(report.errors) == 0 else "FAIL"
    lines.append(f"Result: {result}")
    return "\n".join(lines)
