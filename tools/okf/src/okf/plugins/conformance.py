"""一致性校验插件。"""

from xuan_plugin.plugin import InjectSpec, Plugin

from ..conformance import check_bundle


class ConformanceChecker:
    def __call__(self, ctx, config: dict) -> None:
        try:
            bundle = ctx.get("bundle_accessor")
        except KeyError:
            return
        report = check_bundle(bundle)
        ctx.provide("conformance_report", report, label="conformance_checker")

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="conformance_checker",
            apply=instance,
            inject=[InjectSpec("bundle_accessor")],
            provide=["conformance_report"],
        )
