"""ACP package main entry point.

Allows running the ACP debug server with:
    python -m acp
"""

from acp.agent.implementations.debug_server.cli import main


if __name__ == "__main__":
    main()
