"""CLI 适配层插件。"""

from ..plugin import InjectSpec, Plugin


class CLIAdapter:
    def __call__(self, ctx, config: dict) -> None:
        """CLI 适配层：收集所有服务到统一接口。"""
        services = {}
        service_names = [
            "bundle_accessor",
            "conformance_report",
            "index_generator",
            "log_generator",
            "trust_analyzer",
            "link_analyzer",
        ]
        for name in service_names:
            try:
                services[name] = ctx.get(name)
            except KeyError:
                services[name] = None

        ctx.effect(lambda: [lambda: None], label="cli_adapter")
        ctx.provide("cli_services", services)

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="cli_adapter",
            apply=instance,
            inject=[
                InjectSpec("bundle_accessor"),
                InjectSpec("conformance_report"),
                InjectSpec("index_generator"),
                InjectSpec("log_generator"),
                InjectSpec("trust_analyzer"),
                InjectSpec("link_analyzer"),
            ],
            provide=["cli_services"],
        )
