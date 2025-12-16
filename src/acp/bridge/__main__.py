"""CLI entry point for running the ACP bridge.

Usage:
    uv run -m acp_bridge <command> [args...]
    uv run -m acp_bridge --port 8080 -- your-agent-command --arg1 value1
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys


def main() -> None:
    """Run the ACP bridge from command line."""
    parser = argparse.ArgumentParser(
        description="Bridge a stdio ACP agent to streamable HTTP transport.",
        epilog=(
            "Examples:\n"
            "  acp-bridge your-agent-command\n"
            "  acp-bridge --port 8080 -- your-agent --config config.yml\n"
            "  acp-bridge -H 0.0.0.0 -p 9000 -- uv run my-agent\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("command", help="Command to spawn the ACP agent.")
    parser.add_argument("args", nargs="*", help="Arguments for the agent command.")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Port to serve the HTTP endpoint on. Default: 8080",
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
        help="Allowed CORS origins. Can be specified multiple times.",
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
        format="[%(levelname)1.1s %(asctime)s.%(msecs)03d %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from acp.bridge import ACPBridge, BridgeSettings

    settings = BridgeSettings(
        host=parsed.host,
        port=parsed.port,
        log_level=parsed.log_level,
        allow_origins=parsed.allow_origin if parsed.allow_origin else None,
    )

    bridge = ACPBridge(command=parsed.command, args=parsed.args, cwd=parsed.cwd, settings=settings)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
