"""Examples section of the LLMling-agent documentation."""

from __future__ import annotations

import pathlib

import mknodes as mk

from llmling_agent.agent.agent import Agent
from llmling_agent_docs.examples.utils import iter_examples


nav = mk.MkNav("Examples")

INTRO = """
"This page is generated using another older work of mine, an
experimental MkDocs add-on (or almost fork) named **MkNodes**,
which focusses on programmatic website generation. You can see the source
for this homepage section below.  This system is perfectly suited for Agent usage for
documentation generation.
"""

for example in iter_examples():

    @nav.route.page(example.title, icon=example.icon, hide="toc")
    def _(page: mk.MkPage, ex=example) -> None:  # type: ignore
        """Add example page with description from its docstring."""
        if ex.files:
            link = mk.MkLink.for_pydantic_playground(ex.files)
            page += mk.MkIFrame(link.url, width=1200, height=900)
        if ex.docs:
            page += mk.MkTemplate(str(ex.docs))
        if ex.files:
            link = mk.MkLink.for_pydantic_playground(ex.files)
            page += link


@nav.route.page(
    "MkDocs Integration & Docs generation", icon="oui:documentation", hide="toc"
)
def gen_docs(page: mk.MkPage) -> None:
    """Generate docs using agents."""
    agent = Agent(model="openai:gpt-5-nano")
    content = pathlib.Path("src/llmling_agent/__init__.py")
    page += mk.MkAdmonition(INTRO)
    page += mk.MkCode(pathlib.Path(__file__).read_text(encoding="utf-8"))
    result = agent.run.sync(  # type: ignore[attr-defined]
        "Group and list the given classes. Use markdown for the group headers", content
    )
    page += result.content


if __name__ == "__main__":
    print(nav.to_markdown())
