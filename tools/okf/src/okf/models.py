import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


class TrustTier(enum.Enum):
    UNVERIFIED = "unverified"
    MACHINE_CONFIRMED = "machine_confirmed"
    HUMAN_REVIEWED = "human_reviewed"


@dataclass(frozen=True, slots=True)
class Concept:
    path: Path = field(doc="概念文件的绝对路径")
    type: str = field(doc="概念类型标识（type 字段）")
    title: str = field(default="", doc="概念标题")
    description: str = field(default="", doc="概念描述")
    resource: str = field(default="", doc="概念来源资源标识")
    tags: list[str] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)
    body: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Bundle:
    root: Path = field(doc="Bundle 根目录")
    concepts: dict[str, Concept] = field(default_factory=dict, doc="概念 ID 到 Concept 的映射")
    indices: list[Path] = field(default_factory=list)
    logs: list[Path] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Source:
    resource: str = field(doc="来源资源标识")
    id: str = field(default="", doc="来源 ID")
    title: str = field(default="", doc="来源标题")
    author: str = field(default="", doc="来源作者")
    usage_count: int = 0
    last_modified: date | None = None


@dataclass(frozen=True, slots=True)
class UsageWindow:
    from_date: date = field(doc="使用窗口起始日期")
    to_date: date = field(doc="使用窗口结束日期")


@dataclass(frozen=True, slots=True)
class GeneratedInfo:
    by: str
    at: datetime


@dataclass(frozen=True, slots=True)
class VerificationEvent:
    by: str
    at: datetime


@dataclass(frozen=True, slots=True)
class ComputationParameter:
    name: str
    type: str
    required: bool = False


@dataclass(frozen=True, slots=True)
class Executor:
    resource: str
    receipt: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Attester:
    resource: str


@dataclass(frozen=True, slots=True)
class AttestedComputation:
    runtime: str
    executor: Executor
    attester: Attester
    parameters: list[ComputationParameter] = field(default_factory=list)
    computation: str | None = None


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bundle: Bundle | None = None
