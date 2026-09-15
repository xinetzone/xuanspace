# SPDX-License-Identifier: GPL-2.0-only
"""值对象层（翻译自 podman_compose.py 第 1625-1682、2420-2426、3991-4018 行）。

- ``ServiceDependencyCondition``：依赖条件枚举（docker 风格条件在 ``from_value``
  中映射为 podman 条件，逐行保留）。
- ``ServiceDependency``：服务依赖值对象（名称 + 条件，带相等/哈希语义）。
- ``PullImageSettings``：镜像拉取策略设置（dataclass 原样保留）。
- ``XPodmanSettingKey``：``x-podman`` 扩展设置键（上游为
  ``PodmanCompose`` 的嵌套 ``Enum``；分层后提为模块级 ``StrEnum``，
  T6 引擎层将以此为唯一事实源，差异登记 T10 差异表）。
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .logging_utils import log

__all__ = [
    "ServiceDependency",
    "ServiceDependencyCondition",
    "PullImageSettings",
    "XPodmanSettingKey",
]


class XPodmanSettingKey(StrEnum):
    DOCKER_COMPOSE_COMPAT = "docker_compose_compat"
    DEFAULT_NET_NAME_COMPAT = "default_net_name_compat"
    DEFAULT_NET_BEHAVIOR_COMPAT = "default_net_behavior_compat"
    NAME_SEPARATOR_COMPAT = "name_separator_compat"
    IN_POD = "in_pod"
    POD_ARGS = "pod_args"


class ServiceDependencyCondition(StrEnum):
    CONFIGURED = "configured"
    CREATED = "created"
    EXITED = "exited"
    HEALTHY = "healthy"
    INITIALIZED = "initialized"
    PAUSED = "paused"
    REMOVING = "removing"
    RUNNING = "running"
    STOPPED = "stopped"
    STOPPING = "stopping"
    UNHEALTHY = "unhealthy"
    SERVICE_COMPLETED_SUCCESSFULLY = "service_completed_successfully"

    @classmethod
    def from_value(cls, value: str) -> ServiceDependencyCondition:
        # Check if the value exists in the enum
        for member in cls:
            if member.value == value:
                return member

        # Check if this is a value coming from a reference
        docker_to_podman_cond = {
            "service_healthy": ServiceDependencyCondition.HEALTHY,
            "service_started": ServiceDependencyCondition.RUNNING,
            "service_completed_successfully": (
                ServiceDependencyCondition.SERVICE_COMPLETED_SUCCESSFULLY
            ),
        }
        try:
            return docker_to_podman_cond[value]
        except KeyError:
            raise ValueError(
                f"Value '{value}' is not a valid condition for a service dependency"
            ) from None


class ServiceDependency:
    def __init__(self, name: str, condition: str) -> None:
        self._name = name
        self._condition = ServiceDependencyCondition.from_value(condition)

    @property
    def name(self) -> str:
        return self._name

    @property
    def condition(self) -> ServiceDependencyCondition:
        return self._condition

    def __hash__(self) -> int:
        # Compute hash based on the frozenset of items to ensure order does not matter
        return hash(("name", self._name) + ("condition", self._condition))

    def __eq__(self, other: object) -> bool:
        # Compare equality based on dictionary content
        if isinstance(other, ServiceDependency):
            return self._name == other.name and self._condition == other.condition
        return False


@dataclass
class PullImageSettings:
    POLICY_PRIORITY: ClassVar[dict[str, int]] = {
        "always": 3,
        "newer": 2,
        "missing": 1,
        "never": 0,
        "build": 0,
    }

    image: str
    policy: str = "missing"
    quiet: bool = False

    ignore_pull_error: bool = False

    def __post_init__(self) -> None:
        if self.policy not in self.POLICY_PRIORITY:
            log.debug("Pull policy %s is not valid, using 'missing' instead", self.policy)
            self.policy = "missing"

    def update_policy(self, new_policy: str) -> None:
        if new_policy not in self.POLICY_PRIORITY:
            log.debug("Pull policy %s is not valid, ignoring it", new_policy)
            return

        if self.POLICY_PRIORITY[new_policy] > self.POLICY_PRIORITY[self.policy]:
            self.policy = new_policy
