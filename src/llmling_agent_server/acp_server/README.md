# llmling-agent ACP Integration

ACP (Agent Client Protocol) integration for llmling-agent, enabling seamless interoperability with desktop applications through JSON-RPC 2.0 communication over stdio streams.

## Overview

This package provides a bridge between llmling-agent's powerful agent system and the Agent Client Protocol, allowing you to:

- Expose llmling agents as ACP-compatible services
- Enable bidirectional JSON-RPC 2.0 communication over stdio
- Support session management and conversation history
- Integrate with desktop applications requiring agent capabilities
- Handle file system operations with permission management
- Support terminal integration for command execution
- Stream responses with content blocks (text, image, audio, resources)
- Seamless MCP (Model Context Protocol) server integration

## Installation

```bash
# Install with ACP support
pip install llmling-agent[acp]

# Or install from source
cd llmling-agent
pip install -e .[acp]
```

## Quick Start


ACP-specific settings are controlled via CLI parameters:

```bash
# Basic ACP server
llmling-agent acp agents.yml

# With file system access
llmling-agent acp agents.yml --file-access

# With full capabilities
llmling-agent acp agents.yml --file-access --terminal-access

# With debugging
llmling-agent acp agents.yml --file-access --show-messages --log-level DEBUG
```

### CLI Usage

```bash
# Run agents from config as ACP server
llmling-agent acp agents.yml

# With file system permissions
llmling-agent acp agents.yml --file-access

# Test with manual JSON-RPC (example)
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}' | llmling-agent acp agents.yml
```
