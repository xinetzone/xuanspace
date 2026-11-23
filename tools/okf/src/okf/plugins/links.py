"""链接解析插件。"""

from xuan_plugin.plugin import InjectSpec, Plugin

from ..links import check_broken_links, parse_links_with_context


class LinkResolver:
    def __call__(self, ctx, config: dict) -> None:
        try:
            bundle = ctx.get("bundle_accessor")
        except KeyError:
            return

        def analyze():
            all_links = []
            for concept in bundle.concepts.values():
                links = parse_links_with_context(concept.path, bundle.root)
                all_links.extend(links)
            broken = check_broken_links(all_links, bundle)
            return {"links": all_links, "broken": broken}

        ctx.provide("link_analyzer", analyze, label="link_resolver")

    def __new__(cls) -> Plugin:
        instance = object.__new__(cls)
        return Plugin(
            name="link_resolver",
            apply=instance,
            inject=[InjectSpec("bundle_accessor")],
            provide=["link_analyzer"],
        )
