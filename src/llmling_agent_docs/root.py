"""Root of the documentation tree."""

from __future__ import annotations

import mknodes as mk

from llmling_agent_docs import (
    advanced_features,
    cli,
    configuration,
    core_concepts,
    example_section,
    getting_started,
)


def build(project) -> mk.MkNav:
    build = Build()
    project.linkprovider.add_inv_file("https://mkdocstrings.github.io/objects.inv")
    build.on_theme(project.theme)
    return build.on_root(project.root) or project.root


class Build:
    """Class for building the documentation tree."""

    @classmethod
    def build(cls, root, theme):
        b = cls()
        b.on_theme(theme)
        return b.on_root(root)

    def on_theme(self, theme: mk.Theme) -> None:
        theme.error_page.content = mk.MkAdmonition("Page does not exist!")
        if isinstance(theme, mk.MaterialTheme):
            theme.content_area_width = 1300
            theme.tooltip_width = 800
            theme.add_status_icon("js", "fa6-brands:js", "Uses JavaScript")
            theme.add_status_icon("css", "vaadin:css", "Uses CSS")

    def on_root(self, nav: mk.MkNav) -> mk.MkNav:
        nav.page_template.announcement_bar = mk.MkMetadataBadges("websites")
        nav += getting_started.nav
        nav += core_concepts.nav
        nav += configuration.nav
        nav += example_section.nav
        nav += advanced_features.nav
        nav.add_doc(section_name="API", flatten_nav=True, recursive=True)
        nav += cli.nav
        return nav


if __name__ == "__main__":
    import mknodes as mk

    nav = mk.MkNav()
    bld = Build()
    bld.on_root(nav)
    for node in nav.iter_nodes():
        if isinstance(node, mk.MkPage):
            print(node)
