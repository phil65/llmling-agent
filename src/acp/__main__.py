"""ACP package main entry point.

Allows running the ACP debug server with:
    python -m acp
"""

if __name__ == "__main__":
    from acp.agent.implementations.debug_server.cli import main

    main()
