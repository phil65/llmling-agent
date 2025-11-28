"""Provider for code formatting and linting tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from fsspec import AbstractFileSystem  # type: ignore[import-untyped]

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    import fsspec

    from llmling_agent.tools.base import Tool

from anyenv.code_execution.base import ExecutionEnvironment


logger = get_logger(__name__)

# Map file extensions to ast-grep language identifiers
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".swift": "swift",
    ".lua": "lua",
    ".cs": "csharp",
}


def _substitute_metavars(match: Any, fix_pattern: str, source_code: str) -> str:
    """Substitute $METAVARS and $$$METAVARS in fix pattern with captured values."""
    result = fix_pattern

    # Handle $$$ multi-match metavars first (greedy match)
    for metavar in re.findall(r"\$\$\$([A-Z_][A-Z0-9_]*)", fix_pattern):
        captured_list = match.get_multiple_matches(metavar)
        if captured_list:
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


def _detect_language(path: str) -> str | None:
    """Detect ast-grep language from file extension."""
    suffix = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix)


class CodeTools(ResourceProvider):
    """Provider for code analysis and transformation tools."""

    def __init__(
        self,
        source: fsspec.AbstractFileSystem | ExecutionEnvironment | None = None,
        name: str = "code",
        cwd: str | None = None,
    ) -> None:
        """Initialize with an fsspec filesystem or execution environment.

        Args:
            source: Filesystem or execution environment to operate on
            name: Name for this toolset provider
            cwd: Optional cwd to resolve relative paths against
        """
        from fsspec.asyn import AsyncFileSystem  # type: ignore[import-untyped]
        from fsspec.implementations.asyn_wrapper import (  # type: ignore[import-untyped]
            AsyncFileSystemWrapper,
        )
        from morefs.asyn_local import AsyncLocalFileSystem

        super().__init__(name=name)

        if isinstance(source, ExecutionEnvironment):
            self.execution_env: ExecutionEnvironment | None = source
            fs = source.get_fs()
        else:
            self.execution_env = None
            fs = source

        match fs:
            case AsyncFileSystem():
                self.fs = fs
            case AbstractFileSystem():
                self.fs = AsyncFileSystemWrapper(fs)
            case None:
                self.fs = AsyncLocalFileSystem()
        self.cwd = cwd
        self._tools: list[Tool] | None = None

    def _resolve_path(self, path: str) -> str:
        """Resolve a potentially relative path to an absolute path."""
        if self.cwd and not (path.startswith("/") or (len(path) > 1 and path[1] == ":")):
            return str(Path(self.cwd) / path)
        return path

    async def get_tools(self) -> list[Tool]:
        """Get code analysis tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [self.create_tool(self.format_code, category="execute")]
        if importlib.util.find_spec("ast_grep_py"):
            self._tools.append(self.create_tool(self.ast_grep, category="search"))
        return self._tools

    async def format_code(self, path: str, language: str | None = None) -> str:
        """Format and lint a code file, returning a concise summary.

        Args:
            path: Path to the file to format
            language: Programming language (auto-detected from extension if not provided)

        Returns:
            Short status message about formatting/linting results
        """
        from anyenv.language_formatters import FormatterRegistry

        resolved = self._resolve_path(path)
        try:
            content = await self.fs._cat_file(resolved)
            code = content.decode("utf-8") if isinstance(content, bytes) else content
        except FileNotFoundError:
            return f"❌ File not found: {path}"

        registry = FormatterRegistry("local")
        registry.register_default_formatters()
        # Get formatter by language or detect from extension/content
        formatter = None
        if language:
            formatter = registry.get_formatter_by_language(language)
        if not formatter:
            detected = _detect_language(path)
            if detected:
                formatter = registry.get_formatter_by_language(detected)
        if not formatter:
            detected = registry.detect_language_from_content(code)
            if detected:
                formatter = registry.get_formatter_by_language(detected)
        if not formatter:
            return f"❌ Unsupported language: {language or 'unknown'}"

        try:
            result = await formatter.format_and_lint_string(code, fix=True)

            if result.success:
                # Write back if formatted
                if result.format_result.formatted and result.format_result.output:
                    await self.fs._pipe_file(resolved, result.format_result.output.encode("utf-8"))
                    changes = "formatted and saved"
                else:
                    changes = "no changes needed"
                lint_status = "clean" if result.lint_result.success else "has issues"
                duration = f"{result.total_duration:.2f}s"
                return f"✅ {path}: {changes}, {lint_status} ({duration})"

            errors = []
            if not result.format_result.success:
                errors.append(f"format: {result.format_result.error_type}")
            if not result.lint_result.success:
                errors.append(f"lint: {result.lint_result.error_type}")
            return f"❌ Failed: {', '.join(errors)}"

        except Exception as e:  # noqa: BLE001
            return f"❌ Error: {type(e).__name__}: {e}"

    async def ast_grep(
        self,
        path: str,
        rule: dict[str, Any],
        fix: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Search or transform code in a file using AST patterns.

        Uses ast-grep for structural code search and rewriting based on abstract
        syntax trees. More precise than regex - understands code structure.

        Args:
            path: Path to the file to analyze
            rule: AST matching rule dict (see examples below)
            fix: Optional replacement pattern using $METAVARS from the rule
            dry_run: If True, show changes without applying. If False, write changes.

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
        """
        from ast_grep_py import SgRoot

        resolved = self._resolve_path(path)

        # Detect language from extension
        language = _detect_language(path)
        if not language:
            return {"error": f"Cannot detect language for: {path}"}

        # Read file
        try:
            content = await self.fs._cat_file(resolved)
            code = content.decode("utf-8") if isinstance(content, bytes) else content
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}

        root = SgRoot(code, language)
        node = root.root()
        matches = node.find_all(**rule)

        result: dict[str, Any] = {
            "path": path,
            "language": language,
            "match_count": len(matches),
            "matches": [
                {
                    "text": m.text(),
                    "range": {
                        "start": {
                            "line": m.range().start.line,
                            "column": m.range().start.column,
                        },
                        "end": {
                            "line": m.range().end.line,
                            "column": m.range().end.column,
                        },
                    },
                    "kind": m.kind(),
                }
                for m in matches
            ],
        }

        if fix and matches:
            edits = [m.replace(_substitute_metavars(m, fix, code)) for m in matches]
            fixed_code = node.commit_edits(edits)
            result["fixed_code"] = fixed_code
            result["dry_run"] = dry_run

            if not dry_run:
                await self.fs._pipe_file(resolved, fixed_code.encode("utf-8"))
                result["written"] = True

        return result
