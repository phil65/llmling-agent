"""Resource management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession


LIST_RESOURCES_HELP = """\
Display all resources available to the agent.

Shows:
- Resource names and descriptions
- Resource types and URIs
- Whether parameters are supported
- MIME types

Resource types can be:
- path: Files or URLs
- text: Raw text content
- cli: Command line tools
- source: Python source code
- callable: Python functions
- image: Image files

Use /show-resource for detailed information about specific resources.
"""

SHOW_RESOURCES_HELP = """\
Display detailed information and content of a specific resource.

Shows:
- Resource metadata (type, URI, description)
- MIME type information
- Parameter support status
- Resource content (if loadable)

For resources that support parameters:
- Pass parameters as --param arguments
- Parameters are passed to resource loader\

Examples:
  /show-resource config.yml               # Show configuration file
  /show-resource template --date today    # Template with parameters
  /show-resource image.png               # Show image details
  /show-resource api --key value         # API with parameters

Note: Some resources might require parameters to be viewed.
"""


async def list_resources(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """List available resources."""
    try:
        resources = ctx.data._agent.runtime.get_resources()

        sections = ["# Available Resources\n"]
        for resource in resources:
            desc = f": {resource.description}" if resource.description else ""
            sections.append(f"- **{resource.name}**{desc}")
            sections.append(f"  Type: {resource.type}")
            if resource.uri:
                sections.append(f"  URI: `{resource.uri}`")

            # Show if resource is templated
            if resource.is_templated():
                sections.append("  *Supports parameters*")

        await ctx.output.print("\n".join(sections))
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to list resources: {e}")


async def show_resource(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Show details or content of a resource."""
    if not args:
        await ctx.output.print(
            "Usage: /show-resource <name> [--param1 value1] [--param2 value2]"
        )
        return

    name = args[0]
    try:
        # First get resource info
        resources = ctx.data._agent.runtime.get_resources()
        resource_info = next((r for r in resources if r.name == name), None)
        if not resource_info:
            await ctx.output.print(f"Resource '{name}' not found")
            return
        sections = [f"# Resource: {name}\n", f"Type: {resource_info.type}"]
        if resource_info.uri:
            sections.append(f"URI: `{resource_info.uri}`")
        if resource_info.description:
            sections.append(f"Description: {resource_info.description}")
        if resource_info.is_templated():
            sections.append("\nParameters supported")
        sections.append(f"MIME Type: {resource_info.mime_type}")

        # Try to load content with provided parameters
        try:
            content = await ctx.data._agent.runtime.load_resource(name, **kwargs)
            sections.extend(["\n# Content:", "```", str(content), "```"])
        except Exception as e:  # noqa: BLE001
            sections.append(f"\nFailed to load content: {e}")

        await ctx.output.print("\n".join(sections))
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Error accessing resource: {e}")


list_resources_cmd = Command(
    name="list-resources",
    description="List available resources",
    execute_func=list_resources,
    help_text=LIST_RESOURCES_HELP,
    category="resources",
)

show_resource_cmd = Command(
    name="show-resource",
    description="Show details and content of a resource",
    execute_func=show_resource,
    usage="<name> [--param1 value1] [--param2 value2]",
    help_text=SHOW_RESOURCES_HELP,
    category="resources",
)