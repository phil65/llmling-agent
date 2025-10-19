"""Command for reading file content into conversations."""

from __future__ import annotations

from slashed import CommandContext, CommandError, SlashedCommand  # noqa: TC002
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


class ReadCommand(SlashedCommand):
    """Read content from files or URLs into the conversation.

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

    name = "read"
    category = "content"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        path: str,
        *,
        convert_to_md: bool = False,
    ):
        """Read file content into conversation.

        Args:
            ctx: Command context
            path: Path or URL to read
            convert_to_md: Whether to convert to markdown format
        """
        try:
            agent = ctx.context.agent
            await agent.conversation.add_context_from_path(
                path, convert_to_md=convert_to_md
            )
            await ctx.output.print(
                f"📄 **Added content from** {path!r} **to next message as context**"
            )
        except Exception as e:
            msg = f"Unexpected error reading {path}: {e}"
            logger.exception(msg)
            raise CommandError(msg) from e

    def get_completer(self):
        """Get completer for file paths."""
        return PathCompleter()
