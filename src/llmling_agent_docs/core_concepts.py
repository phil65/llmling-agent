"""Core Concepts section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Core Concepts")


@nav.route.page("Basic Agent", icon="octicon:person-16")
def _(page: mk.MkPage) -> None:
    """Core Agent implementation and features."""
    page += mk.MkTemplate("docs/agents/basic_agent.md")


@nav.route.page("Tools and Tool Management", icon="octicon:tools-16")
def _(page: mk.MkPage) -> None:
    """Tool management and registration."""
    page += mk.MkTemplate("docs/agents/tool_manager.md")


@nav.route.page("Agent Pool", icon="octicon:database-16")
def _(page: mk.MkPage) -> None:
    """Agent pool management and features."""
    page += mk.MkTemplate("docs/concepts/pool.md")


@nav.route.page("Team", icon="octicon:people-16")
def _(page: mk.MkPage) -> None:
    """Working with agent teams and groups."""
    page += mk.MkTemplate("docs/agents/team.md")


@nav.route.page("TeamRun", icon="octicon:play-16")
def _(page: mk.MkPage) -> None:
    """Team execution and monitoring."""
    page += mk.MkTemplate("docs/concepts/team_run.md")


@nav.route.page("Run interface", icon="codicon:run-all")
def _(page: mk.MkPage) -> None:
    """Agent / Team run interface."""
    page += mk.MkTemplate("docs/concepts/run_methods.md")


@nav.route.page("Talk & TeamTalk", icon="octicon:comment-discussion-16")
def _(page: mk.MkPage) -> None:
    """Agent communication system."""
    page += mk.MkTemplate("docs/concepts/talk.md")


@nav.route.page("Routing", icon="octicon:git-branch-16")
def _(page: mk.MkPage) -> None:
    """Agent communication system."""
    page += mk.MkTemplate("docs/concepts/routing.md")


@nav.route.page("Tasks", icon="octicon:tasklist-16")
def _(page: mk.MkPage) -> None:
    """Task definition and execution."""
    page += mk.MkTemplate("docs/concepts/tasks.md")


@nav.route.page("Events", icon="octicon:broadcast-16")
def _(page: mk.MkPage) -> None:
    """Event system and handlers."""
    page += mk.MkTemplate("docs/concepts/events.md")


@nav.route.page("Storage System", icon="octicon:database-16")
def _(page: mk.MkPage) -> None:
    """Storage and persistence system."""
    page += mk.MkTemplate("docs/concepts/storage.md")


@nav.route.page("Knowledge System", icon="octicon:book-16")
def _(page: mk.MkPage) -> None:
    """Knowledge management and access."""
    page += mk.MkTemplate("docs/concepts/knowledge.md")


@nav.route.page("Capabilities", icon="octicon:shield-check-16")
def _(page: mk.MkPage) -> None:
    """Agent capabilities system."""
    page += mk.MkTemplate("docs/concepts/capabilities.md")


@nav.route.page("Agent Context", icon="octicon:project-16")
def _(page: mk.MkPage) -> None:
    """Agent context and state management."""
    page += mk.MkTemplate("docs/concepts/agent_context.md")


@nav.route.page("Conversation Manager", icon="octicon:history-16")
def _(page: mk.MkPage) -> None:
    """Conversation history and management."""
    page += mk.MkTemplate("docs/concepts/conversation_manager.md")


@nav.route.page("Signal System", icon="octicon:broadcast-16")
def _(page: mk.MkPage) -> None:
    """Signal and event system."""
    page += mk.MkTemplate("docs/concepts/signals.md")


@nav.route.page("Messages and Responses", icon="octicon:comment-discussion-16")
def _(page: mk.MkPage) -> None:
    """Message types and response handling."""
    page += mk.MkTemplate("docs/concepts/messages.md")


@nav.route.page("MCP Servers", icon="octicon:server-16")
def _(page: mk.MkPage) -> None:
    """MCP server integration."""
    page += mk.MkTemplate("docs/concepts/mcp.md")


# @nav.route.page("Human Provider", icon="octicon:person-16")
# def _(page: mk.MkPage):
#     """Using the human provider for interaction."""
#     page += mk.MkTemplate("docs/interaction/human_provider.md")


if __name__ == "__main__":
    print(nav.to_markdown())
