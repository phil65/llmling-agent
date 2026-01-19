"""OpenCode-compatible API server.

This module provides a FastAPI-based server that implements the OpenCode API,
allowing OpenCode SDK clients to interact with AgentPool agents.

Example usage:

    from agentpool import AgentPool
    from agentpool_server.opencode_server import OpenCodeServer

    async with AgentPool("config.yml") as pool:
        server = OpenCodeServer(pool.main_agent, port=4096)
        await server.run_async()

Or programmatically:

    from agentpool_server.opencode_server import create_app

    app = create_app(agent=my_agent, working_dir="/path/to/project")
    # Use with uvicorn or other ASGI server
"""
