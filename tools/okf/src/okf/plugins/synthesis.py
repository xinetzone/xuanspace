"""索引与日志合成插件。"""

from ..plugin import InjectSpec, Plugin
from ..synthesis import generate_index, parse_log


class IndexSynthesizer:
    def __call__(self, ctx, config: dict) -> None:
        try:
            bundle = ctx.get("bundle_accessor")
        except KeyError:
            return

        def generate():
            concepts = list(bundle.concepts.values())
            return generate_index(bundle.root, concepts)

        ctx.effect(lambda: [lambda: None], label="index_synthesizer")
        ctx.provide("index_generator", generate)

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="index_synthesizer",
            apply=instance,
            inject=[InjectSpec("bundle_accessor")],
            provide=["index_generator"],
        )


class LogSynthesizer:
    def __call__(self, ctx, config: dict) -> None:
        try:
            bundle = ctx.get("bundle_accessor")
        except KeyError:
            return

        def generate():
            log_path = bundle.root / "log.md"
            if log_path.exists():
                return parse_log(log_path)
            return {}

        ctx.effect(lambda: [lambda: None], label="log_synthesizer")
        ctx.provide("log_generator", generate)

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="log_synthesizer",
            apply=instance,
            inject=[InjectSpec("bundle_accessor")],
            provide=["log_generator"],
        )
