"""Prepare LLM output to be in proper shape and executable."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from schemez import ToolsetCodeGenerator

    from llmling_agent.tools import Tool


def fix_code(python_code: str) -> str:
    """Fix code to be executable."""
    # Simplified execution: require main() function pattern
    if "async def main(" not in python_code:
        # Auto-wrap code in main function, ensuring last expression is returned
        lines = python_code.strip().splitlines()
        if lines:
            # Check if last line is an expression (not a statement)
            last_line = lines[-1].strip()
            if last_line and not any(
                last_line.startswith(kw)
                for kw in [
                    "import ",
                    "from ",
                    "def ",
                    "class ",
                    "if ",
                    "for ",
                    "while ",
                    "try ",
                    "with ",
                    "async def ",
                ]
            ):
                # Last line looks like an expression, add return
                lines[-1] = f"    return {last_line}"
                indented_lines = [f"    {line}" for line in lines[:-1]] + [lines[-1]]
            else:
                indented_lines = [f"    {line}" for line in lines]
            python_code = "async def main():\n" + "\n".join(indented_lines)
        else:
            python_code = "async def main():\n    pass"
    return python_code


def tools_to_codegen(
    tools: Sequence[Tool],
    include_signatures: bool = True,
    include_docstrings: bool = True,
) -> ToolsetCodeGenerator:
    """Create a ToolsetCodeGenerator from a sequence of Tools.

    Args:
        tools: Tools to generate code for
        include_signatures: Include function signatures in documentation
        include_docstrings: Include function docstrings in documentation

    Returns:
        ToolsetCodeGenerator instance
    """
    from pydantic_ai import RunContext
    from schemez import ToolsetCodeGenerator, create_schema
    from schemez.code_generation.tool_code_generator import ToolCodeGenerator

    from llmling_agent.agent.context import AgentContext

    generators = [
        ToolCodeGenerator(
            schema=create_schema(
                t.callable,
                name_override=t.name,
                description_override=t.description,
                mode="openai",
                exclude_types=[AgentContext, RunContext],
            ),
            callable=t.callable,
            name_override=t.name,
        )
        for t in tools
    ]
    return ToolsetCodeGenerator(generators, include_signatures, include_docstrings)
