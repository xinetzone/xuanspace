"""项目类型 → 构建后端的分派表。"""

from collections.abc import Mapping

from .backends import (
    BuildBackend,
    NativeBackend,
    OtherBackend,
    PythonBackend,
    StaticBackend,
)

_DEFAULT_BACKENDS: Mapping[str, BuildBackend] = {
    "python": PythonBackend(),
    "native": NativeBackend(),
    "static": StaticBackend(),
    "other": OtherBackend(),
}


class BuildDispatcher:
    """按项目类型返回后端；未登记类型统一落 ``other``。"""

    def __init__(self, backends: Mapping[str, BuildBackend] | None = None) -> None:
        self._backends = dict(_DEFAULT_BACKENDS if backends is None else backends)

    def select(self, project_type: str) -> BuildBackend:
        return self._backends.get(project_type, self._backends["other"])

    def registered_kinds(self) -> tuple[str, ...]:
        return tuple(self._backends)


DEFAULT_DISPATCHER = BuildDispatcher()


def dispatch(project_type: str) -> BuildBackend:
    """模块级便捷入口：等价于 ``DEFAULT_DISPATCHER.select``。"""
    return DEFAULT_DISPATCHER.select(project_type)
