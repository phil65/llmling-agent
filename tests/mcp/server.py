"""Compact FastMCP server demonstrating sampling and elicitation in one workflow."""

import asyncio

from fastmcp import Context, FastMCP
from mcp.types import ModelHint, ModelPreferences


mcp = FastMCP("Test Server")


@mcp.tool
async def sample_test(ctx: Context, message: str) -> str:
    """Analyze code, ask user which issues to fix, then return improved code."""
    # Step 1: Use sampling to check if there are issues (yes/no)
    prefs = ModelPreferences(hints=[ModelHint(name="gpt-5-nano")])
    has_issues_result = await ctx.sample(
        message,
        max_tokens=500,
        system_prompt="only 'yes' or 'no'.",
        model_preferences=prefs,
    )
    return str(has_issues_result)


@mcp.tool
async def test_elicitation(ctx: Context, message: str):
    """Use this to test the elicitation feature!."""
    return await ctx.elicit(message, response_type=bool)


@mcp.tool
async def test_progress(ctx: Context, message: str) -> str:
    """Test progress reporting with the given message."""
    await ctx.report_progress(0, 100, "first step")
    await asyncio.sleep(0.1)
    await ctx.report_progress(50, 100, "second step")
    await asyncio.sleep(0.1)
    await ctx.report_progress(99, 100, "third step")
    await asyncio.sleep(0.1)
    return f"Progress test completed with message: {message}"


if __name__ == "__main__":
    mcp.run(show_banner=False)
