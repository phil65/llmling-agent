"""Command for reading file content into conversations."""

from __future__ import annotations

from slashed import Command, CommandContext, CommandError  # noqa: TC002
from slashed.completers import PathCompleter

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger


logger = get_logger(__name__)


READ_HELP = """\
Read content from files or URLs into the conversation.

By default reads raw content, but can convert supported formats to markdown
with the --convert-to-md flag.

Supported formats for conversion:
- PDF documents
- Office files (Word, Excel, PowerPoint)
- Images (with metadata)
- Audio files (metadata)
- HTML pages
- Text formats (CSV, JSON, XML)

Examples:
  /read document.txt               # Read raw text
  /read document.pdf --convert-to-md   # Convert PDF to markdown
  /read https://example.com/doc.docx --convert-to-md
  /read presentation.pptx --convert-to-md
"""


async def read_command(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Read file content into conversation.

    Args:
        ctx: Command context
        args: Command arguments (path)
        kwargs: Command options
    """
    if not args:
        await ctx.output.print("**Usage:** `/read <path> [--convert-to-md]`")
        return

    path = args[0]
    convert_to_md = kwargs.get("convert_to_md", "").lower() in {"true", "1", "yes"}

    try:
        agent = ctx.context.agent
        await agent.conversation.add_context_from_path(path, convert_to_md=convert_to_md)
        await ctx.output.print(
            f"📄 **Added content from** {path!r} **to next message as context**"
        )
    except Exception as e:
        msg = f"Unexpected error reading {path}: {e}"
        logger.exception(msg)
        raise CommandError(msg) from e


read_cmd = Command(
    name="read",
    description="Read file or URL content into conversation",
    execute_func=read_command,
    usage="<path_or_url> [--convert-to-md]",
    help_text=READ_HELP,
    category="content",
    completer=PathCompleter(),
)
