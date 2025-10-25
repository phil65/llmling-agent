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
    return await ctx.elicit(message, response_type=bool)  # type: ignore


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


from pydantic import BaseModel


class WeatherData(BaseModel):
    temperature: float
    humidity: int


class StructuredResponse(BaseModel):
    result: str
    data: WeatherData
    timestamp: str


@mcp.tool
async def test_return_types(
    ctx: Context, return_type: str
) -> str | StructuredResponse | list:
    """Test different PydanticAI return types based on the return_type parameter.

    Args:
        return_type: One of 'text', 'structured', 'mixed' to test different return scenarios
    """
    if return_type == "text":
        # Simple text return
        return "Simple text response for PydanticAI"
    if return_type == "structured":
        # Structured data that should be returned directly
        return StructuredResponse(
            result="success",
            data=WeatherData(temperature=23.5, humidity=65),
            timestamp="2024-01-15T14:30:00Z",
        )
    if return_type == "mixed":
        # This will have both structured output and rich content
        # FastMCP will handle the conversion to appropriate formats
        return [
            {"type": "data", "value": 42},
            {"type": "message", "text": "Rich content example"},
            {"type": "metadata", "info": "Additional context"},
        ]
    return f"Unknown return_type: {return_type}. Use 'text', 'structured', or 'mixed'"


@mcp.tool
async def test_rich_content(ctx: Context, content_type: str):
    """Test FastMCP rich content types that should be converted to PydanticAI types.

    Args:
        content_type: One of 'image', 'audio', 'file', 'mixed'
    """
    from fastmcp.tools.tool import ToolResult
    from fastmcp.utilities.types import Audio, File, Image
    from mcp.types import TextContent

    if content_type == "image":
        # Return FastMCP Image - should convert to PydanticAI ImageUrl/BinaryContent
        return Image(data=b"fake_image_data", format="png")
    if content_type == "audio":
        # Return FastMCP Audio - should convert to PydanticAI AudioUrl/BinaryContent
        return Audio(data=b"fake_audio_data", format="wav")
    if content_type == "file":
        # Return FastMCP File - should convert to PydanticAI DocumentUrl/BinaryContent
        return File(data=b"fake_file_data", format="pdf", name="test.pdf")
    if content_type == "mixed":
        # Return mixed content with both structured data and rich content
        return ToolResult(
            content=[
                TextContent(type="text", text="Text description"),
                Image(data=b"fake_image", format="png").to_image_content(),
            ],
            structured_content={"structured_data": "additional info", "count": 42},
        )
    return (
        f"Unknown content_type: {content_type}. Use 'image', 'audio', 'file', or 'mixed'"
    )


if __name__ == "__main__":
    mcp.run(show_banner=False)
