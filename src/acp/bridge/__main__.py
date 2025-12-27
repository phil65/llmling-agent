"""CLI entry point for running the ACP bridge.

Usage:
    # HTTP bridge (default)
    uv run -m acp.bridge <command> [args...]
    uv run -m acp.bridge --port 8080 -- your-agent-command --arg1 value1

    # WebSocket bridge
    uv run -m acp.bridge --transport websocket <command> [args...]
    uv run -m acp.bridge -t ws --port 8765 -- uv run agentpool serve-acp
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys

import anyio


def main() -> None:
    """Run the ACP bridge from command line."""
    parser = argparse.ArgumentParser(
        description="Bridge a stdio ACP agent to HTTP or WebSocket transport.",
        epilog=(
            "Examples:\n"
            "  # HTTP bridge\n"
            "  acp-bridge your-agent-command\n"
            "  acp-bridge --port 8080 -- your-agent --config config.yml\n"
            "\n"
            "  # WebSocket bridge\n"
            "  acp-bridge -t ws -- uv run agentpool serve-acp\n"
            "  acp-bridge --transport websocket --port 8765 -- uv run my-agent\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("command", help="Command to spawn the ACP agent.")
    parser.add_argument("args", nargs="*", help="Arguments for the agent command.")
    parser.add_argument(
        "-t",
        "--transport",
        choices=["http", "websocket", "ws"],
        default="http",
        help="Transport type: http (default) or websocket/ws",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=None,
        help="Port to serve on. Default: 8080 (HTTP) or 8765 (WebSocket)",
    )
    parser.add_argument(
        "-H",
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level. Default: INFO",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Allowed CORS origins (HTTP only). Can be specified multiple times.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for the agent subprocess.",
    )

    parsed = parser.parse_args()

    if not parsed.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, parsed.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    use_websocket = parsed.transport in ("websocket", "ws")

    if use_websocket:
        # Reduce websockets noise
        logging.getLogger("websockets").setLevel(logging.WARNING)

        from acp.bridge.ws_server import ACPWebSocketServer

        port = parsed.port or 8765
        server = ACPWebSocketServer(
            command=parsed.command,
            args=parsed.args,
            host=parsed.host,
            port=port,
            cwd=parsed.cwd,
        )

        print(f"🔗 ACP WebSocket bridge starting on ws://{parsed.host}:{port}")
        print(f"   Agent command: {parsed.command} {' '.join(parsed.args)}")

        with contextlib.suppress(KeyboardInterrupt):
            anyio.run(server.serve)
    else:
        from acp.bridge import ACPBridge, BridgeSettings

        port = parsed.port or 8080
        settings = BridgeSettings(
            host=parsed.host,
            port=port,
            log_level=parsed.log_level,
            allow_origins=parsed.allow_origin if parsed.allow_origin else None,
        )

        bridge = ACPBridge(
            command=parsed.command, args=parsed.args, cwd=parsed.cwd, settings=settings
        )
        with contextlib.suppress(KeyboardInterrupt):
            anyio.run(bridge.run)


if __name__ == "__main__":
    main()
