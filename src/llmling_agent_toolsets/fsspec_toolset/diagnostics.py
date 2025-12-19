"""LSP diagnostics manager for file operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anyenv.lsp_servers import DiagnosticRunner
from anyenv.os_commands import get_os_command_provider


if TYPE_CHECKING:
    from anyenv.lsp_servers import Diagnostic, LSPServerInfo
    from exxec import ExecutionEnvironment


class DiagnosticsManager:
    """Manages LSP diagnostics for file operations.

    Lazily checks server availability and caches results.
    """

    def __init__(self, env: ExecutionEnvironment | None = None) -> None:
        """Initialize diagnostics manager.

        Args:
            env: Execution environment for running diagnostic commands.
                 If None, diagnostics will be disabled.
        """
        self._env = env
        self._runner: DiagnosticRunner | None = None
        self._server_availability: dict[str, bool] = {}

    @property
    def enabled(self) -> bool:
        """Check if diagnostics are enabled (have an execution environment)."""
        return self._env is not None

    def _get_runner(self) -> DiagnosticRunner:
        """Get or create the diagnostic runner."""
        if self._runner is None:
            executor = self._env.execute_command if self._env else None
            self._runner = DiagnosticRunner(executor=executor)
            self._runner.register_defaults()
        return self._runner

    async def _is_server_available(self, server: LSPServerInfo) -> bool:
        """Check if a server's CLI command is available (cached).

        Args:
            server: The LSP server info to check

        Returns:
            True if the server's command is available
        """
        if not self._env or not server.cli_diagnostics:
            return False

        if server.id not in self._server_availability:
            provider = get_os_command_provider(self._env.os_type)
            which_cmd = provider.get_command("which")
            cmd = which_cmd.create_command(server.cli_diagnostics.command)
            result = await self._env.execute_command(cmd)
            is_available = (
                which_cmd.parse_command(result.stdout or "", result.exit_code or 0) is not None
            )
            self._server_availability[server.id] = is_available

        return self._server_availability[server.id]

    async def run_for_file(self, path: str) -> list[Diagnostic]:
        """Run all applicable diagnostics on a file.

        Args:
            path: File path to check

        Returns:
            List of diagnostics from all available servers
        """
        if not self.enabled:
            return []

        runner = self._get_runner()
        all_diagnostics: list[Diagnostic] = []

        for server in runner.get_servers_for_file(path):
            if await self._is_server_available(server):
                result = await runner.run(server, [path])
                all_diagnostics.extend(result.diagnostics)

        return all_diagnostics

    def format_diagnostics(self, diagnostics: list[Diagnostic]) -> str:
        """Format diagnostics as a Markdown table.

        Args:
            diagnostics: List of diagnostics to format

        Returns:
            Markdown table with Severity, Location, Code, Description columns
        """
        if not diagnostics:
            return ""

        lines: list[str] = [
            "| Severity | Location | Code | Description |",
            "|----------|----------|------|-------------|",
        ]
        for d in diagnostics:
            loc = f"{d.file}:{d.line}:{d.column}"
            code = d.code or ""
            # Escape pipe characters in message
            msg = d.message.replace("|", "\\|")
            lines.append(f"| {d.severity.upper()} | {loc} | {code} | {msg} |")

        return "\n".join(lines)
