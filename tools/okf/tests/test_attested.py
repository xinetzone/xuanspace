"""Attested Computation 契约解析器测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from okf.attested import (
    extract_computation_from_body,
    is_attested_computation,
    parse_attested_computation,
)
from okf.models import Concept


def _concept(**kwargs) -> Concept:
    defaults = dict(path=Path("concept.md"), type="Attested Computation")
    defaults.update(kwargs)
    return Concept(**defaults)


# ── is_attested_computation ───────────────────────────────────────────────


def test_is_attested_computation_true():
    assert is_attested_computation(_concept()) is True


def test_is_attested_computation_false():
    assert is_attested_computation(_concept(type="Metric")) is False


# ── extract_computation_from_body ─────────────────────────────────────────


def test_extract_computation_from_body():
    body = "# Computation\n```python\nprint('hello')\n```\ntrailing"
    assert extract_computation_from_body(body) == "print('hello')"


def test_extract_computation_no_match():
    assert extract_computation_from_body("no fence here at all") is None


# ── parse_attested_computation ────────────────────────────────────────────


def test_parse_attested_computation_inline():
    fm = {
        "runtime": "python3",
        "parameters": [{"name": "x", "type": "int", "required": True}],
        "executor": {"resource": "exec:1", "receipt": ["receipt-a", "receipt-b"]},
        "attester": {"resource": "attester:1"},
    }
    body = "# Computation\n```python\nprint('hello')\n```\n"
    concept = _concept(frontmatter=fm, body=body)
    result = parse_attested_computation(concept)
    assert result.runtime == "python3"
    assert result.executor.resource == "exec:1"
    assert result.executor.receipt == ["receipt-a", "receipt-b"]
    assert result.attester.resource == "attester:1"
    assert len(result.parameters) == 1
    parameter = result.parameters[0]
    assert parameter.name == "x"
    assert parameter.type == "int"
    assert parameter.required is True
    assert result.computation == "print('hello')"


def test_parse_attested_computation_file(tmp_path):
    compute_file = tmp_path / "compute.py"
    compute_file.write_text("result = x + 1", encoding="utf-8")
    fm = {
        "runtime": "python3",
        "computation": "compute.py",
        "executor": {"resource": "exec:2"},
        "attester": {"resource": "attester:2"},
    }
    concept = _concept(frontmatter=fm, body="")
    result = parse_attested_computation(concept, bundle_root=tmp_path)
    assert result.computation == "result = x + 1"


def test_parse_attested_computation_file_requires_root():
    fm = {
        "runtime": "python3",
        "computation": "compute.py",
        "executor": {},
        "attester": {},
    }
    concept = _concept(frontmatter=fm, body="")
    with pytest.raises(ValueError):
        parse_attested_computation(concept)
