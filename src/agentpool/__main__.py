"""CLI entry point - forwards to agentpool_cli for faster startup."""

from agentpool_cli.__main__ import cli


if __name__ == "__main__":
    cli()
