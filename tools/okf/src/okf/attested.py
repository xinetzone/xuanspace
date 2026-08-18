"""Attested Computation 契约解析器（零第三方依赖）。

仅使用 Python 标准库 ``pathlib`` 与 ``re``。
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import AttestedComputation, Attester, ComputationParameter, Concept, Executor

__all__ = [
    "is_attested_computation",
    "parse_attested_computation",
    "extract_computation_from_body",
    "load_computation_file",
]

# 匹配 "# Computation" 标题后的第一个代码围栏
_COMPUTATION_FENCE_RE = re.compile(
    r"^#\s+Computation[ \t]*\n```(?:\w+)?[ \t]*\n(.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


def is_attested_computation(concept: Concept) -> bool:
    """判断概念是否为 Attested Computation 类型。"""
    return concept.type == "Attested Computation"


def parse_attested_computation(
    concept: Concept, bundle_root: Path | None = None
) -> AttestedComputation:
    """解析 Attested Computation 契约字段。

    Parameters
    ----------
    concept:
        待解析的 Concept 实例。
    bundle_root:
        Bundle 根目录，用于解析 ``computation`` 字段中指定的文件路径。
        仅当 ``computation`` 为非空文件路径时需要。

    Returns
    -------
    AttestedComputation
        解析后的 Attested Computation 模型。
    """
    fm = concept.frontmatter

    # ── Step 1: 解析 frontmatter 字段 ──────────────────────────────────

    runtime = fm.get("runtime")
    if not isinstance(runtime, str) or not runtime.strip():
        raise ValueError("'runtime' must be a non-empty string")

    parameters: list[ComputationParameter] = []
    raw_params = fm.get("parameters", [])
    if raw_params and isinstance(raw_params, list):
        for entry in raw_params:
            parameters.append(
                ComputationParameter(
                    name=str(entry.get("name", "")),
                    type=str(entry.get("type", "")),
                    required=bool(entry.get("required", False)),
                )
            )

    computation_raw = fm.get("computation")
    computation: str | None = None
    if isinstance(computation_raw, str) and computation_raw.strip():
        computation = computation_raw.strip()

    executor_raw = fm.get("executor", {})
    executor = Executor(
        resource=str(executor_raw.get("resource", "")),
        receipt=(
            [str(r) for r in executor_raw["receipt"]]
            if isinstance(executor_raw.get("receipt"), list)
            else []
        ),
    )

    attester_raw = fm.get("attester", {})
    attester = Attester(resource=str(attester_raw.get("resource", "")))

    # ── Step 2: 计算逻辑加载 ───────────────────────────────────────────

    computation_content: str | None = None

    if computation is None or computation == "":
        # 内联计算：从 body 中提取 # Computation 围栏代码块
        computation_content = extract_computation_from_body(concept.body)
    else:
        # 文件式计算：按路径加载文件内容
        if bundle_root is None:
            raise ValueError(
                "bundle_root is required when 'computation' specifies a file path"
            )
        computation_content = load_computation_file(computation, bundle_root)

    return AttestedComputation(
        runtime=runtime,
        executor=executor,
        attester=attester,
        parameters=parameters,
        computation=computation_content,
    )


def extract_computation_from_body(body: str) -> str | None:
    """从 body 中提取 ``# Computation`` 围栏代码块内容。

    Parameters
    ----------
    body:
        Markdown 正文内容。

    Returns
    -------
    str or None
        代码围栏内的内容（去除围栏标记），无匹配时返回 ``None``。
    """
    m = _COMPUTATION_FENCE_RE.search(body)
    if m is None:
        return None
    return m.group(1).rstrip()


def load_computation_file(path: str, bundle_root: Path) -> str:
    """从文件路径加载计算逻辑内容。

    Parameters
    ----------
    path:
        相对或绝对文件路径。相对路径将基于 ``bundle_root`` 拼接。
    bundle_root:
        Bundle 根目录。

    Returns
    -------
    str
        文件内容。
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = bundle_root / file_path
    return file_path.read_text(encoding="utf-8")
