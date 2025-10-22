"""Provider for code formatting and linting tools."""

from __future__ import annotations

from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


async def format_code(code: str, language: str | None = None) -> str:
    """Format and lint code, returning a concise summary.

    Args:
        code: Source code to format and lint
        language: Programming language (auto-detected if not provided)

    Returns:
        Short status message about formatting/linting results
    """
    from anyenv.language_formatters import FormatterRegistry

    registry = FormatterRegistry("subprocess")
    registry.register_default_formatters()

    # Get formatter by language or try to detect
    formatter = None
    if language:
        formatter = registry.get_formatter_by_language(language)

    if not formatter:
        # Try to detect from content
        detected = registry.detect_language_from_content(code)
        if detected:
            formatter = registry.get_formatter_by_language(detected)

    if not formatter:
        return f"❌ Unsupported language: {language or 'unknown'}"

    try:
        result = await formatter.format_and_lint_string(code, fix=True)

        if result.success:
            changes = "formatted" if result.format_result.formatted else "no changes"
            lint_status = "clean" if result.lint_result.success else "has issues"
            duration = f"{result.total_duration:.2f}s"
            return f"✅ Code {changes}, {lint_status} ({duration})"
        errors = []
        if not result.format_result.success:
            errors.append(f"format: {result.format_result.error_type}")
        if not result.lint_result.success:
            errors.append(f"lint: {result.lint_result.error_type}")
        return f"❌ Failed: {', '.join(errors)}"

    except Exception as e:  # noqa: BLE001
        return f"❌ Error: {type(e).__name__}"


def create_code_tools() -> list[Tool]:
    """Create tools for code formatting and linting."""
    return [
        Tool.from_callable(
            format_code,
            source="builtin",
            category="execute",
        ),
    ]


class CodeTools(StaticResourceProvider):
    """Provider for code formatting and linting tools."""

    def __init__(self, name: str = "code"):
        super().__init__(name=name, tools=create_code_tools())
