"""信任推导插件。"""

from ..plugin import InjectSpec, Plugin
from ..trust import derive_trust_tier, is_stale


class TrustDeriver:
    def __call__(self, ctx, config: dict) -> None:
        try:
            bundle = ctx.get("bundle_accessor")
        except KeyError:
            return

        def analyze(concept_id: str | None = None):
            if concept_id:
                concept = bundle.concepts.get(concept_id)
                if concept:
                    return {
                        "trust_tier": derive_trust_tier(concept).value,
                        "is_stale": is_stale(concept),
                    }
                return None
            return {
                cid: {
                    "trust_tier": derive_trust_tier(c).value,
                    "is_stale": is_stale(c),
                }
                for cid, c in bundle.concepts.items()
            }

        ctx.effect(lambda: [lambda: None], label="trust_deriver")
        ctx.provide("trust_analyzer", analyze)

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="trust_deriver",
            apply=instance,
            inject=[InjectSpec("bundle_accessor")],
            provide=["trust_analyzer"],
        )
