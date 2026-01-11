"""OpenCode-compatible API server.

This module provides a FastAPI-based server that implements the OpenCode API,
allowing OpenCode SDK clients to interact with AgentPool agents.

Example usage:

    from agentpool_server.opencode_server import OpenCodeServer

    server = OpenCodeServer(port=4096)
    server.run()

Or programmatically:

    from agentpool_server.opencode_server import create_app

    app = create_app(working_dir="/path/to/project")
    # Use with uvicorn or other ASGI server
"""
