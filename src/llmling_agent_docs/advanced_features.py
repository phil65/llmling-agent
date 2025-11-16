"""Advanced Features section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Advanced Features")


@nav.route.page("Async Operations", icon="octicon:sync-16")
def _(page: mk.MkPage) -> None:
    """Asynchronous operations and patterns."""
    page += mk.MkTemplate("docs/advanced/async.md")


@nav.route.page("Callables as Agents", icon="octicon:code-16")
def _(page: mk.MkPage) -> None:
    """Using Python callables as agents."""
    page += mk.MkTemplate("docs/advanced/callables.md")


@nav.route.page("System Prompts", icon="octicon:comment-16")
def _(page: mk.MkPage) -> None:
    """Advanced system prompt features."""
    page += mk.MkTemplate("docs/advanced/system_prompts.md")


@nav.route.page("Data Extraction", icon="octicon:file-code-16")
def _(page: mk.MkPage) -> None:
    """Extracting structured data."""
    page += mk.MkTemplate("docs/interaction/extract.md")


@nav.route.page("Decision Making", icon="octicon:git-branch-16")
def _(page: mk.MkPage) -> None:
    """Advanced decision making patterns."""
    page += mk.MkTemplate("docs/interaction/pick.md")


@nav.route.page("Generic Typing", icon="codicon:python")
def _(page: mk.MkPage) -> None:
    """Advanced type system features."""
    page += mk.MkTemplate("docs/advanced/generic_typing.md")


@nav.route.page("Injection", icon="octicon:git-merge-16")
def _(page: mk.MkPage) -> None:
    """Advanced dependency injection."""
    page += mk.MkTemplate("docs/advanced/injection.md")


@nav.route.page("Models & Providers", icon="octicon:cpu-16")
def _(page: mk.MkPage) -> None:
    """Advanced model and provider features."""
    page += mk.MkTemplate("docs/advanced/models.md")


@nav.route.page("Prompts", icon="octicon:comment-discussion-16")
def _(page: mk.MkPage) -> None:
    """Advanced prompt handling."""
    page += mk.MkTemplate("docs/advanced/prompts.md")


@nav.route.page("Syntactic Sugar", icon="octicon:code-square-16")
def _(page: mk.MkPage) -> None:
    """Syntactic conveniences and shortcuts."""
    page += mk.MkTemplate("docs/advanced/syntactic_sugar.md")


@nav.route.page("Task Monitoring", icon="octicon:pulse-16")
def _(page: mk.MkPage) -> None:
    """Advanced task monitoring features."""
    page += mk.MkTemplate("docs/advanced/task_monitoring.md")


@nav.route.page("Agent state", icon="octicon:database-16")
def _(page: mk.MkPage) -> None:
    """Agent state management."""
    page += mk.MkTemplate("docs/advanced/agent_state.md")


@nav.route.page("Document conversion", icon="octicon:project-symlink-16")
def _(page: mk.MkPage) -> None:
    """Document conversion."""
    page += mk.MkTemplate("docs/advanced/document_conversion.md")


@nav.route.page("Passing data to the agent", icon="octicon:arrow-both-16")
def _(page: mk.MkPage) -> None:
    """Passing data to the agent."""
    page += mk.MkTemplate("docs/advanced/passing_data.md")


@nav.route.page("ACP Integration", icon="octicon:desktop-download-16")
def _(page: mk.MkPage) -> None:
    """Agent Client Protocol integration for desktop applications."""
    page += mk.MkTemplate("docs/advanced/acp_integration.md")


# @nav.route.page("UPath & YAML Features", icon="octicon:file-code-16")
# def _(page: mk.MkPage):
#     """UPath integration and YAML features."""
#     page += mk.MkTemplate("docs/concepts/upath_yaml.md")

if __name__ == "__main__":
    print(nav.to_markdown())
