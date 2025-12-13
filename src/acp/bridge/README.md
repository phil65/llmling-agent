# ACP Bridge

A stdio-to-HTTP bridge for ACP (Agent Client Protocol) agents.

## Overview

The ACP Bridge allows you to expose stdio-based ACP agents via a streamable HTTP endpoint. This is useful when you want to run an agent as a subprocess but communicate with it over HTTP instead of stdio.

## Features

- **Stdio to HTTP**: Spawns an ACP agent subprocess and exposes it via HTTP
- **Streamable HTTP**: Uses streamable HTTP transport for efficient communication
- **CORS Support**: Optional CORS configuration for web clients
- **Health Endpoint**: Built-in `/status` endpoint for monitoring
- **Type-Safe**: Full type hints and Pydantic validation

## Installation

The bridge is included in the `acp` package:

```python
from acp.bridge import ACPBridge, BridgeSettings
```

## Usage

### Programmatic Usage

```python
import asyncio
from acp.bridge import ACPBridge, BridgeSettings

settings = BridgeSettings(
    host="127.0.0.1",
    port=8080,
    log_level="INFO",
    allow_origins=["*"],  # Optional CORS
)

bridge = ACPBridge(
    command="uv",
    args=["run", "my-agent"],
    settings=settings,
)

asyncio.run(bridge.run())
```

### Command Line Usage

```bash
# Basic usage
acp-bridge my-agent-command

# With custom port and host
acp-bridge --port 8080 --host 0.0.0.0 -- uv run my-agent

# With debug logging
acp-bridge --log-level DEBUG my-agent

# With CORS enabled
acp-bridge --allow-origin "*" my-agent
```

## Endpoints

### POST /acp

The main endpoint for ACP JSON-RPC requests. Send ACP protocol messages here.

```bash
curl -X POST http://localhost:8080/acp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}'
```

### GET /status

Health check endpoint that returns the bridge status.

```bash
curl http://localhost:8080/status
```

Response:
```json
{
  "status": "connected",
  "command": "my-agent",
  "args": ["--config", "config.yml"]
}
```

## Configuration

### BridgeSettings

- `host` (str): Host to bind the server to. Default: `"127.0.0.1"`
- `port` (int): Port to serve on. Default: `8080`
- `log_level` (str): Logging level. Default: `"INFO"`
- `allow_origins` (list[str] | None): CORS allowed origins. Default: `None`

### ACPBridge

- `command` (str): Command to spawn the ACP agent
- `args` (list[str] | None): Arguments for the command
- `env` (Mapping[str, str] | None): Environment variables
- `cwd` (str | Path | None): Working directory
- `settings` (BridgeSettings | None): Server settings

## Example: Bridging an Existing Agent

```python
from acp.bridge import ACPBridge

# Bridge the debug server
bridge = ACPBridge(
    command="uv",
    args=["run", "python", "-m", "acp.debug_server"],
)

await bridge.run()
```

## Architecture

The bridge works by:

1. Spawning the specified command as a subprocess with stdio pipes
2. Creating a `ClientSideConnection` to communicate with the agent via stdio
3. Exposing an HTTP server that accepts JSON-RPC requests
4. Forwarding HTTP requests to the stdio agent and returning responses

```
HTTP Client → HTTP Server → ClientSideConnection → stdio → Agent Process
```

## Supported Methods

The bridge supports all standard ACP protocol methods:

- `initialize`
- `session/new`, `session/load`, `session/list`
- `session/fork`, `session/resume`
- `session/prompt`, `session/cancel`
- `session/set_mode`, `session/set_model`
- `authenticate`
- Extension methods (prefixed with `_`)

## Error Handling

The bridge returns standard JSON-RPC error responses:

- `-32700`: Parse error (invalid JSON)
- `-32600`: Invalid Request (missing method)
- `-32603`: Internal error (agent communication failure)
- `503`: Service unavailable (agent not connected)

## See Also

- [ACP Protocol Documentation](../README.md)
- [MCP Proxy](https://github.com/modelcontextprotocol/mcp-proxy) - Similar concept for MCP