"""``okf.models`` 核心数据模型的单元测试。"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from pathlib import Path

import pytest

from okf.models import (
    AttestedComputation,
    Attester,
    Bundle,
    ComputationParameter,
    Concept,
    ConformanceReport,
    Executor,
    GeneratedInfo,
    Source,
    TrustTier,
    UsageWindow,
    VerificationEvent,
)


class TestTrustTier:
    def test_members(self) -> None:
        assert TrustTier.UNVERIFIED.value == "unverified"
        assert TrustTier.MACHINE_CONFIRMED.value == "machine_confirmed"
        assert TrustTier.HUMAN_REVIEWED.value == "human_reviewed"

    def test_member_count(self) -> None:
        assert len(TrustTier) == 3


class TestConcept:
    def test_frozen(self) -> None:
        concept = Concept(path=Path("metric.md"), type="Metric")
        with pytest.raises(dataclasses.FrozenInstanceError):
            concept.title = "changed"

    def test_defaults(self) -> None:
        concept = Concept(path=Path("metric.md"), type="Metric")
        assert concept.title == ""
        assert concept.description == ""
        assert concept.resource == ""
        assert concept.tags == []
        assert concept.frontmatter == {}
        assert concept.body == ""
        assert concept.extra == {}

    def test_repr_contains_type(self) -> None:
        concept = Concept(path=Path("metric.md"), type="Metric")
        assert "type='Metric'" in repr(concept)


class TestBundle:
    def test_defaults(self) -> None:
        bundle = Bundle(root=Path("."))
        assert bundle.concepts == {}
        assert bundle.indices == []
        assert bundle.logs == []

    def test_mutable_defaults_are_isolated(self) -> None:
        first = Bundle(root=Path("."))
        second = Bundle(root=Path("."))
        assert first.concepts is not second.concepts
        assert first.indices is not second.indices
        assert first.logs is not second.logs


class TestSource:
    def test_defaults(self) -> None:
        source = Source(resource="res")
        assert source.resource == "res"
        assert source.id == ""
        assert source.title == ""
        assert source.author == ""
        assert source.usage_count == 0
        assert source.last_modified is None

    def test_no_runtime_type_enforcement(self) -> None:
        # dataclass 不做运行时类型校验（类型错误交由 mypy 等静态检查捕获）
        source = Source(resource="res", usage_count="not-a-number")  # type: ignore[arg-type]
        assert source.usage_count == "not-a-number"

    def test_last_modified_accepts_date(self) -> None:
        d = date(2024, 1, 1)
        source = Source(resource="res", last_modified=d)
        assert source.last_modified == d


class TestUsageWindow:
    def test_fields(self) -> None:
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        window = UsageWindow(from_date=start, to_date=end)
        assert window.from_date == start
        assert window.to_date == end


class TestGeneratedInfo:
    def test_fields(self) -> None:
        at = datetime(2024, 1, 15, 10, 0, 0)
        info = GeneratedInfo(by="pipeline:etl", at=at)
        assert info.by == "pipeline:etl"
        assert info.at == at


class TestVerificationEvent:
    def test_fields(self) -> None:
        at = datetime(2024, 1, 18, 9, 30, 0)
        event = VerificationEvent(by="human:alice", at=at)
        assert event.by == "human:alice"
        assert event.at == at


class TestComputationParameter:
    def test_defaults(self) -> None:
        param = ComputationParameter(name="n", type="int")
        assert param.name == "n"
        assert param.type == "int"
        assert param.required is False


class TestAttestedComputation:
    @staticmethod
    def _executor() -> Executor:
        return Executor(resource="runner")

    @staticmethod
    def _attester() -> Attester:
        return Attester(resource="signer")

    def test_computation_defaults_to_none(self) -> None:
        comp = AttestedComputation(
            runtime="python3.14",
            executor=self._executor(),
            attester=self._attester(),
        )
        assert comp.computation is None
        assert comp.parameters == []

    def test_computation_accepts_path_string(self) -> None:
        comp = AttestedComputation(
            runtime="python3.14",
            executor=self._executor(),
            attester=self._attester(),
            computation="path/to/script.py",
        )
        assert comp.computation == "path/to/script.py"


class TestConformanceReport:
    def test_defaults(self) -> None:
        report = ConformanceReport()
        assert report.errors == []
        assert report.warnings == []
        assert report.bundle is None
