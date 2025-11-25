"""Provider for code formatting and linting tools."""

from __future__ import annotations

import importlib.util
import re
from typing import Any

from llmling_agent.resource_providers import StaticResourceProvider
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

    registry = FormatterRegistry("local")
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


def _substitute_metavars(match, fix_pattern: str, source_code: str) -> str:
    """Substitute $METAVARS and $$$METAVARS in fix pattern with captured values."""
    result = fix_pattern

    # Handle $$$ multi-match metavars first (greedy match)
    for metavar in re.findall(r"\$\$\$([A-Z_][A-Z0-9_]*)", fix_pattern):
        captured_list = match.get_multiple_matches(metavar)
        if captured_list:
            # Extract original text span to preserve formatting
            first = captured_list[0]
            last = captured_list[-1]
            start_idx = first.range().start.index
            end_idx = last.range().end.index
            original_span = source_code[start_idx:end_idx]
            result = result.replace(f"$$${metavar}", original_span)

    # Handle single $ metavars
    for metavar in re.findall(r"(?<!\$)\$([A-Z_][A-Z0-9_]*)", fix_pattern):
        captured = match.get_match(metavar)
        if captured:
            result = result.replace(f"${metavar}", captured.text())

    return result


async def ast_grep(
    code: str,
    language: str,
    rule: dict[str, Any],
    fix: str | None = None,
) -> dict[str, Any]:
    """Search or transform code using AST patterns.

    Uses ast-grep for structural code search and rewriting based on abstract syntax trees.
    More precise than regex - understands code structure.

    Args:
        code: Source code to analyze
        language: Programming language (python, javascript, typescript, rust, go, etc.)
        rule: AST matching rule dict (see examples below)
        fix: Optional replacement pattern using $METAVARS from the rule

    Returns:
        Dict with matches and optionally transformed code

    ## Pattern Syntax

    - `$NAME` - captures single node (identifier, expression, etc.)
    - `$$$ITEMS` - captures multiple nodes (arguments, statements, etc.)
    - Patterns match structurally, not textually

    ## Rule Keys

    | Key | Description | Example |
    |-----|-------------|---------|
    | pattern | Code pattern with metavars | `"print($MSG)"` |
    | kind | AST node type | `"function_definition"` |
    | regex | Regex on node text | `"^test_"` |
    | inside | Must be inside matching node | `{"kind": "class_definition"}` |
    | has | Must contain matching node | `{"pattern": "return"}` |
    | all | All rules must match | `[{"kind": "call"}, {"has": ...}]` |
    | any | Any rule must match | `[{"pattern": "print"}, {"pattern": "log"}]` |

    ## Examples

    **Find all print calls:**
    ```
    rule={"pattern": "print($MSG)"}
    ```

    **Find and replace console.log:**
    ```
    rule={"pattern": "console.log($MSG)"}
    fix="logger.info($MSG)"
    ```

    **Find functions containing await:**
    ```
    rule={
        "kind": "function_definition",
        "has": {"pattern": "await $EXPR"}
    }
    ```

    **Find unused imports (no references):**
    ```
    rule={
        "pattern": "import $MOD",
        "not": {"precedes": {"pattern": "$MOD"}}
    }
    ```

    **Rename function arguments:**
    ```
    rule={"pattern": "def $FN($$$ARGS): $$$BODY"}
    fix="def $FN($$$ARGS) -> None: $$$BODY"
    ```
    """
    from ast_grep_py import SgRoot

    root = SgRoot(code, language)
    node = root.root()

    matches = node.find_all(**rule)

    result: dict[str, Any] = {
        "match_count": len(matches),
        "matches": [
            {
                "text": m.text(),
                "range": {
                    "start": {"line": m.range().start.line, "column": m.range().start.column},
                    "end": {"line": m.range().end.line, "column": m.range().end.column},
                },
                "kind": m.kind(),
            }
            for m in matches
        ],
    }

    if fix and matches:
        edits = [m.replace(_substitute_metavars(m, fix, code)) for m in matches]
        result["fixed_code"] = node.commit_edits(edits)

    return result


class CodeTools(StaticResourceProvider):
    """Provider for code formatting and linting tools."""

    def __init__(self, name: str = "code") -> None:
        tools = [Tool.from_callable(format_code, source="builtin", category="execute")]

        if importlib.util.find_spec("ast_grep_py"):
            tools.append(
                Tool.from_callable(
                    ast_grep,
                    source="builtin",
                    category="search",
                    description_override=(
                        "Search or transform code using AST patterns. "
                        "More precise than regex - understands code structure."
                    ),
                )
            )

        super().__init__(name=name, tools=tools)
