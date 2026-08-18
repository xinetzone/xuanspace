from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


class TrustTier(enum.Enum):
    UNVERIFIED = "unverified"
    MACHINE_CONFIRMED = "machine_confirmed"
    HUMAN_REVIEWED = "human_reviewed"


@dataclass(frozen=True)
class Concept:
    path: Path
    type: str
    title: str = ""
    description: str = ""
    resource: str = ""
    tags: list[str] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)
    body: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Bundle:
    root: Path
    concepts: dict[str, Concept] = field(default_factory=dict)
    indices: list[Path] = field(default_factory=list)
    logs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class Source:
    resource: str
    id: str = ""
    title: str = ""
    author: str = ""
    usage_count: int = 0
    last_modified: date | None = None


@dataclass(frozen=True)
class UsageWindow:
    from_date: date
    to_date: date


@dataclass(frozen=True)
class GeneratedInfo:
    by: str
    at: datetime


@dataclass(frozen=True)
class VerificationEvent:
    by: str
    at: datetime


@dataclass(frozen=True)
class ComputationParameter:
    name: str
    type: str
    required: bool = False


@dataclass(frozen=True)
class Executor:
    resource: str
    receipt: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Attester:
    resource: str


@dataclass(frozen=True)
class AttestedComputation:
    runtime: str
    executor: Executor
    attester: Attester
    parameters: list[ComputationParameter] = field(default_factory=list)
    computation: str | None = None


@dataclass(frozen=True)
class ConformanceReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bundle: Bundle | None = None
