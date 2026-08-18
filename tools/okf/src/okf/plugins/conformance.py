"""一致性校验插件。"""

from ..conformance import check_bundle
from ..plugin import InjectSpec, Plugin


class ConformanceChecker:
    def __call__(self, ctx, config: dict) -> None:
        try:
            bundle = ctx.get("bundle_accessor")
        except KeyError:
            return
        report = check_bundle(bundle)
        ctx.effect(lambda: [lambda: None], label="conformance_checker")
        ctx.provide("conformance_report", report)

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="conformance_checker",
            apply=instance,
            inject=[InjectSpec("bundle_accessor")],
            provide=["conformance_report"],
        )
