"""CLI for ACP Debug Server.

Simple command-line interface for the ACP debug server that combines
ACP protocol server with FastAPI web interface for testing.
"""

import argparse
import asyncio
import logging
import sys


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACP Debug Server - Combined ACP + FastAPI testing server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (ACP on stdio, web UI on localhost:8000)
  python -m acp.cli

  # Custom port for web interface
  python -m acp.cli --port 9000

  # Custom host and port
  python -m acp.cli --host 0.0.0.0 --port 8080

  # Test ACP connection
  echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}' | python -m acp

  # Open web interface at http://localhost:8000 to manually send notifications
        """,  # noqa: E501
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7777,
        help="Port for FastAPI web interface (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for FastAPI web interface (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )
    from acp.agent.implementations.debug_server.debug_server import ACPDebugServer

    args = parser.parse_args()

    # Configure logging
    level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Create and run debug server
    server = ACPDebugServer(fastapi_port=args.port, fastapi_host=args.host)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nDebug server interrupted", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"Debug server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
