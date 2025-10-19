"""Language formatters with dependency-injected command execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
import tempfile


@dataclass
class LintResult:
    """Result of a linting operation."""

    success: bool
    output: str
    errors: str
    fixed_issues: int = 0
    remaining_issues: int = 0


@dataclass
class FormatResult:
    """Result of a formatting operation."""

    success: bool
    output: str
    errors: str
    formatted: bool = False


@dataclass
class FormatAndLintResult:
    """Combined result of format and lint operations."""

    format_result: FormatResult
    lint_result: LintResult

    @property
    def success(self) -> bool:
        """Overall success if both operations succeeded."""
        return self.format_result.success and self.lint_result.success


# Type alias for command handler
CommandHandler = Callable[[list[str]], Awaitable[tuple[int, str, str]]]


async def default_command_handler(cmd: list[str]) -> tuple[int, str, str]:
    """Default subprocess-based command handler."""
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return (
        process.returncode or 0,
        stdout.decode() if stdout else "",
        stderr.decode() if stderr else "",
    )


class LanguageFormatter(ABC):
    """Abstract base class for language-specific formatters."""

    def __init__(self, command_handler: CommandHandler = default_command_handler):
        self.command_handler = command_handler

    @property
    @abstractmethod
    def name(self) -> str:
        """Language name (e.g., 'Python', 'TOML')."""

    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """File extensions this formatter handles (e.g., ['.py', '.pyi'])."""

    @property
    @abstractmethod
    def pygments_lexers(self) -> list[str]:
        """Pygments lexer names for this language (e.g., ['python', 'python3'])."""

    @abstractmethod
    async def format(self, path: Path) -> FormatResult:
        """Format a file."""

    @abstractmethod
    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        """Lint a file, optionally fixing issues."""

    async def format_and_lint(self, path: Path, fix: bool = False) -> FormatAndLintResult:
        """Format and then lint a file."""
        format_result = await self.format(path)
        lint_result = await self.lint(path, fix=fix)
        return FormatAndLintResult(format_result, lint_result)

    def can_handle(self, path: Path) -> bool:
        """Check if this formatter can handle the given file."""
        return path.suffix.lower() in self.extensions

    def can_handle_language(self, language: str) -> bool:
        """Check if this formatter can handle the given language name."""
        return language.lower() in [lexer.lower() for lexer in self.pygments_lexers]

    async def format_string(
        self, content: str, language: str | None = None
    ) -> FormatResult:
        """Format a string by creating a temporary file.

        Args:
            content: String content to format
            language: Language name (pygments lexer name) if extension can't be determined

        Returns:
            FormatResult with formatted content in output field
        """
        # Use primary extension for temp file
        extension = self.extensions[0] if self.extensions else ".txt"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=extension, delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        try:
            result = await self.format(temp_path)
            if result.success:
                # Read the formatted content back
                formatted_content = temp_path.read_text()
                result.output = formatted_content
            return result
        finally:
            temp_path.unlink(missing_ok=True)

    async def lint_string(
        self, content: str, language: str | None = None, fix: bool = False
    ) -> LintResult:
        """Lint a string by creating a temporary file.

        Args:
            content: String content to lint
            language: Language name (pygments lexer name) if extension can't be determined
            fix: Whether to apply fixes

        Returns:
            LintResult with any fixes applied to output field
        """
        # Use primary extension for temp file
        extension = self.extensions[0] if self.extensions else ".txt"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=extension, delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        try:
            result = await self.lint(temp_path, fix=fix)
            if result.success and fix:
                # Read the potentially modified content back
                modified_content = temp_path.read_text()
                result.output = modified_content
            return result
        finally:
            temp_path.unlink(missing_ok=True)

    async def format_and_lint_string(
        self, content: str, language: str | None = None, fix: bool = False
    ) -> FormatAndLintResult:
        """Format and lint a string."""
        format_result = await self.format_string(content, language)
        content_to_lint = format_result.output if format_result.success else content
        lint_result = await self.lint_string(content_to_lint, language, fix)
        return FormatAndLintResult(format_result, lint_result)


class PythonFormatter(LanguageFormatter):
    """Python formatter using ruff."""

    @property
    def name(self) -> str:
        return "Python"

    @property
    def extensions(self) -> list[str]:
        return [".py", ".pyi"]

    @property
    def pygments_lexers(self) -> list[str]:
        return ["python", "python3", "py"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["uv", "run", "ruff", "format", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)

        return FormatResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            formatted=return_code == 0,  # ruff format returns 0 on success
        )

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        cmd = ["uv", "run", "ruff", "check"]
        if fix:
            cmd.extend(["--fix", "--unsafe-fixes"])
        cmd.append(str(path))

        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,  # Could parse output to count fixes
            remaining_issues=0,  # Could parse output to count remaining issues
        )


class TOMLFormatter(LanguageFormatter):
    """TOML formatter using tombi."""

    @property
    def name(self) -> str:
        return "TOML"

    @property
    def extensions(self) -> list[str]:
        return [".toml"]

    @property
    def pygments_lexers(self) -> list[str]:
        return ["toml"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["uv", "run", "tombi", "format", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)
        ok = return_code == 0
        return FormatResult(success=ok, output=stdout, errors=stderr, formatted=ok)

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        # tombi lint doesn't seem to have a --fix option, so we ignore the fix parameter
        cmd = ["uv", "run", "tombi", "lint", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,
            remaining_issues=0,
        )


class TypeScriptFormatter(LanguageFormatter):
    """TypeScript/JavaScript formatter using biome."""

    @property
    def name(self) -> str:
        return "TypeScript"

    @property
    def extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx", ".json"]

    @property
    def pygments_lexers(self) -> list[str]:
        return ["typescript", "ts", "javascript", "js", "jsx", "tsx", "json"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["biome", "format", "--write", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)
        ok = return_code == 0
        return FormatResult(success=ok, output=stdout, errors=stderr, formatted=ok)

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        cmd = ["biome", "lint"]
        if fix:
            cmd.append("--write")
        cmd.append(str(path))

        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,  # Could parse output to count fixes
            remaining_issues=0,  # Could parse output to count remaining issues
        )


class RustFormatter(LanguageFormatter):
    """Rust formatter using rustfmt and clippy."""

    @property
    def name(self) -> str:
        return "Rust"

    @property
    def extensions(self) -> list[str]:
        return [".rs"]

    @property
    def pygments_lexers(self) -> list[str]:
        return ["rust", "rs"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["rustfmt", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)
        ok = return_code == 0
        return FormatResult(success=ok, output=stdout, errors=stderr, formatted=ok)

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        cmd = ["cargo", "clippy"]
        if fix:
            cmd.append("--fix")
        cmd.extend(["--", "--", str(path)])

        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,  # Could parse output to count fixes
            remaining_issues=0,  # Could parse output to count remaining issues
        )


class FormatterRegistry:
    """Registry for language formatters."""

    def __init__(self):
        self.formatters: list[LanguageFormatter] = []

    def register(self, formatter: LanguageFormatter) -> None:
        """Register a formatter."""
        self.formatters.append(formatter)

    def get_formatter(self, path: Path) -> LanguageFormatter | None:
        """Get formatter for given file path."""
        return next((f for f in self.formatters if f.can_handle(path)), None)

    def get_formatter_by_language(self, language: str) -> LanguageFormatter | None:
        """Get formatter for given language name (pygments lexer)."""
        return next((f for f in self.formatters if f.can_handle_language(language)), None)

    def detect_language_from_content(self, content: str) -> str | None:
        """Detect language from content using pygments (if available)."""
        try:
            from pygments.lexers import guess_lexer

            lexer = guess_lexer(content)
            return lexer.name.lower()
        except ImportError:
            return None
        except Exception:  # noqa: BLE001
            # Pygments couldn't detect the language
            return None

    def get_supported_extensions(self) -> list[str]:
        """Get all supported file extensions."""
        return sorted({e for formatter in self.formatters for e in formatter.extensions})


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Create registry and register formatters
        registry = FormatterRegistry()
        registry.register(PythonFormatter())
        registry.register(TOMLFormatter())
        registry.register(TypeScriptFormatter())
        registry.register(RustFormatter())

        print(f"Supported extensions: {registry.get_supported_extensions()}")

        # Create test files
        test_py = Path("test.py")
        test_toml = Path("test.toml")

        test_py.write_text("def hello( ):\n    print('world')")
        test_toml.write_text('[section]\nkey="value"')

        try:
            # Test using registry
            py_formatter = registry.get_formatter(test_py)
            if py_formatter:
                result = await py_formatter.format_and_lint(test_py, fix=True)
                print(f"Python format success: {result.format_result.success}")
                print(f"Python lint success: {result.lint_result.success}")

            toml_formatter = registry.get_formatter(test_toml)
            if toml_formatter:
                result = await toml_formatter.format_and_lint(test_toml)
                print(f"TOML format success: {result.format_result.success}")
                print(f"TOML lint success: {result.lint_result.success}")

        finally:
            # Cleanup
            test_py.unlink(missing_ok=True)
            test_toml.unlink(missing_ok=True)

    asyncio.run(main())
