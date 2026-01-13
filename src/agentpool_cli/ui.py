"""UI commands for launching interactive interfaces."""

from __future__ import annotations

import signal
import socket
import subprocess
import time
from typing import Annotated

import typer as t

from agentpool_cli import log


logger = log.get_logger(__name__)

# Create UI subcommand group
ui_app = t.Typer(help="Launch interactive user interfaces")


@ui_app.command("opencode")
def opencode_ui_command(
    config: Annotated[
        str | None,
        t.Argument(help="Path to agent configuration (optional, not used with --attach)"),
    ] = None,
    host: Annotated[
        str,
        t.Option("--host", "-h", help="Host to bind/connect to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        t.Option("--port", "-p", help="Port for server to listen on / connect to"),
    ] = 4096,
    agent: Annotated[
        str | None,
        t.Option(
            "--agent",
            help="Name of specific agent to use (not used with --attach)",
        ),
    ] = None,
    attach: Annotated[
        bool,
        t.Option("--attach", help="Only attach TUI to existing server (don't start server)"),
    ] = False,
) -> None:
    """Launch OpenCode TUI with integrated server or attach to existing one.

    By default, starts an OpenCode-compatible server in the background and
    automatically attaches the OpenCode TUI to it. When you exit the TUI,
    the server is automatically shut down.

    With --attach, only launches the TUI and connects to an existing server
    (useful when running the server separately or connecting from multiple clients).

    Examples:
        # Start server + TUI
        agentpool ui opencode

        # Use specific config and agent
        agentpool ui opencode agents.yml --agent myagent

        # Custom port
        agentpool ui opencode --port 8080

        # Attach to existing server (no server startup)
        agentpool ui opencode --attach
        agentpool ui opencode --attach --port 8080
    """
    url = f"http://{host}:{port}"

    # Attach-only mode: just launch TUI
    if attach:
        logger.info("Attaching to existing OpenCode server", url=url)

        # Clear screen for clean TUI
        import os

        os.system("clear" if os.name != "nt" else "cls")

        result = subprocess.run(["opencode", "attach", url], check=False)
        if result.returncode != 0 and result.returncode != 130:  # 130 = Ctrl+C
            logger.warning("OpenCode TUI exited with non-zero status", code=result.returncode)
        return

    # Build server command
    server_cmd = [
        "agentpool",
        "serve-opencode",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if config:
        server_cmd.append(config)
    if agent:
        server_cmd.extend(["--agent", agent])

    logger.info("Starting OpenCode server", url=url)

    # Start server in background with suppressed output
    server = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server to be ready with retry
        max_retries = 30
        for i in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                sock.connect((host, port))
                sock.close()
                logger.info("Server is ready", url=url)
                break
            except (TimeoutError, ConnectionRefusedError, OSError):
                if i == max_retries - 1:
                    msg = f"Server failed to start after {max_retries} attempts"
                    raise RuntimeError(msg)
                time.sleep(0.5)

        # Give HTTP layer a moment to be fully ready
        time.sleep(0.5)

        # Clear screen before launching TUI
        import os

        os.system("clear" if os.name != "nt" else "cls")

        # Attach TUI
        result = subprocess.run(["opencode", "attach", url], check=False)
        if result.returncode != 0:
            logger.warning("OpenCode TUI exited with non-zero status", code=result.returncode)

    except KeyboardInterrupt:
        logger.info("UI interrupted by user")
    except Exception as e:
        logger.exception("Error running OpenCode UI")
        raise t.Exit(1) from e
    finally:
        # Clean up server
        logger.info("Shutting down server")
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not shut down gracefully, killing")
            server.kill()


@ui_app.command("toad")
def toad_ui_command(
    config: Annotated[
        str | None,
        t.Argument(help="Path to agent configuration (optional)"),
    ] = None,
    websocket: Annotated[
        bool,
        t.Option("--websocket", "-w", help="Use WebSocket transport (otherwise stdio)"),
    ] = False,
    port: Annotated[
        int,
        t.Option("--port", "-p", help="Port for WebSocket server (only with --websocket)"),
    ] = 8765,
) -> None:
    """Launch Toad TUI for ACP agents.

    By default uses stdio transport where Toad spawns the agentpool server.
    With --websocket, starts a WebSocket ACP server in the background first.

    Examples:
        # Direct stdio (Toad spawns server)
        agentpool ui toad

        # Use specific config
        agentpool ui toad agents.yml

        # WebSocket transport
        agentpool ui toad --websocket

        # WebSocket with custom port
        agentpool ui toad --websocket --port 9000
    """
    if websocket:
        _run_toad_websocket(config, port)
    else:
        _run_toad_stdio(config)


def _run_toad_stdio(config: str | None) -> None:
    """Run Toad with stdio transport (Toad spawns server)."""
    # Build agentpool command that Toad will spawn
    agentpool_cmd = "agentpool serve-acp"
    if config:
        agentpool_cmd += f" {config}"

    # Clear screen for clean TUI
    import os

    os.system("clear" if os.name != "nt" else "cls")

    # Run toad with agentpool as subprocess
    result = subprocess.run(
        ["uvx", "--from", "batrachian-toad@latest", "toad", "acp", agentpool_cmd],
        check=False,
    )

    if result.returncode != 0 and result.returncode != 130:  # 130 = Ctrl+C
        logger.warning("Toad TUI exited with non-zero status", code=result.returncode)


def _run_toad_websocket(config: str | None, port: int) -> None:
    """Run Toad with WebSocket transport."""
    url = f"ws://localhost:{port}"

    # Build server command
    server_cmd = [
        "agentpool",
        "serve-acp",
        "--transport",
        "websocket",
        "--ws-port",
        str(port),
    ]
    if config:
        server_cmd.append(config)

    logger.info("Starting ACP WebSocket server", url=url)

    # Start server in background
    server = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server startup
        time.sleep(1.5)

        # Clear screen for clean TUI
        import os

        os.system("clear" if os.name != "nt" else "cls")

        # Run toad with mcp-ws client
        result = subprocess.run(
            ["uvx", "--from", "batrachian-toad@latest", "toad", "acp", f"uvx mcp-ws {url}"],
            check=False,
        )

        if result.returncode != 0 and result.returncode != 130:  # 130 = Ctrl+C
            logger.warning("Toad TUI exited with non-zero status", code=result.returncode)

    except KeyboardInterrupt:
        logger.info("UI interrupted by user")
    except Exception as e:
        logger.exception("Error running Toad UI")
        raise t.Exit(1) from e
    finally:
        # Clean up server
        logger.info("Shutting down server")
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not shut down gracefully, killing")
            server.kill()


if __name__ == "__main__":
    t.run(ui_app)
