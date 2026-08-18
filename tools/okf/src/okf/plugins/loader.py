"""Bundle 加载器插件。"""

from pathlib import Path

from ..loader import load_bundle
from ..plugin import Plugin


class BundleLoader:
    """加载 Bundle 目录树，提供 bundle_accessor 服务。"""

    def __call__(self, ctx, config: dict) -> None:
        bundle_root = Path(config.get("path", "."))
        bundle = load_bundle(bundle_root)
        ctx.effect(lambda: [lambda: None], label="bundle_loader")
        ctx.provide("bundle_accessor", bundle)

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="bundle_loader",
            apply=instance,
            provide=["bundle_accessor"],
        )
